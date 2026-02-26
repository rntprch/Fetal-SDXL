# Fetal Ultrasound Generation

This folder contains a full pipeline for:
- generating captions for fetal ultrasound dataset images
- converting JSONL captions into training `.txt` files
- training SDXL LoRA with `kohya-ss/sd-scripts`
- generating synthetic images with the trained LoRA
- calculating KID and MS-SSIM metrics

## Environment Installation (`sd-scripts`)

Reference repo: https://github.com/kohya-ss/sd-scripts

```bash
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
source venv/bin/activate

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt
pip install xformers --index-url https://download.pytorch.org/whl/cu124
accelerate config
```

### `accelerate config` answers

- This machine
- No distributed training
- NO
- NO
- NO
- all
- bf16

## Download Weights

Base SDXL checkpoint:

```bash
mkdir -p sdxl_lora/weights
wget -O sdxl_lora/weights/sd_xl_base_1.0.safetensors \
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true"
```

## Dataset Structure (Training Input)

Expected image/caption layout for LoRA training:

```text
fetal_ultrasound/
  img/
    1_thalamic/
      *.png|jpg
      *.txt
    2_cerebellum/
      *.png|jpg
      *.txt
    3_ventricular/
      *.png|jpg
      *.txt
  log/
  output/
```

## Full Pipeline

Run commands from the repository root (the folder where this repo is stored) unless noted otherwise.

### 1. Generate Captions (CSV + JSONL)

This runs the caption extraction/verification pipeline and writes a CSV plus JSONL.

Requirements:
- Hugging Face token (`HF_TOKEN`)
- model access for the configured `--model-id` (default in script is `lingshu-medical-mllm/Lingshu-32B`)

```bash
export HF_TOKEN=your_token_here

python captions/generate_captions.py \
  --csv ./FETAL_PLANES_DB_data.csv \
  --image-root ./FetalPlanes/Images \
  --prompts ./captions/promts.json \
  --landmarks ./captions/landmarks.json \
  --output ./captions/brain_descriptions_full.csv \
  --output-jsonl ./FetalPlanes/brain_descriptions_full.jsonl
```

Notes:
- `--limit 100` is useful for a quick smoke test.
- Output JSONL is used in the next step.

### 2. Create Full Description `.txt` Files for Training

This script:
- reads the JSONL captions
- builds full textual descriptions
- moves images into plane folders (`1_thalamic`, `2_cerebellum`, `3_ventricular`)
- writes `.txt` files next to each image
- deletes images that have no description in JSONL

```bash

python captions/create_txt.py \
  --jsonl ./captions/brain_descriptions_full.jsonl \
  --img-dir ../fetal_ultrasound/img
```

### 3. Train SDXL LoRA

`train.sh` launches `sd-scripts` training using `clean_code/sdxl_lora/config.json`.

Before running:
- place/verify the base checkpoint path in `clean_code/sdxl_lora/config.json`
- ensure `SD_SCRIPTS_DIR` points to your `sd-scripts` clone

```bash
cd ./clean_code/sdxl_lora

export SD_SCRIPTS_DIR=/path/to/sd-scripts
bash train.sh ./config.json
```

Typical outputs:
- LoRA weights: `clean_code/sdxl_lora/fetal_ultrasound/output/*.safetensors`
- Training state dirs: `clean_code/sdxl_lora/fetal_ultrasound/output/*-state`

### 4. Generate Images with Trained LoRA

`generate.sh` creates images for 3 prompts and stores them in plane-named folders:
- `trans_thalamic`
- `trans_cerebellum`
- `trans_ventricular`

```bash
cd ./clean_code/sdxl_lora

export SD_SCRIPTS_DIR=/path/to/sd-scripts

bash generate.sh \
  ../fetal_ultrasound/output/sdxl_lora_r16_a16_e8-state \
  ./config.json \
  ../fetal_ultrasound/generated/sdxl_lora_r16_a16_e8
```

Optional generation controls (env vars):
- `SAMPLES_PER_PROMPT=500`
- `BATCH_SIZE=4`
- `CFG_SCALE=6.0`
- `STEPS=28`
- `LORA_MUL=1.0`
- `BASE_SEED=1234`

### 5. Calculate Metrics (KID + MS-SSIM)

`metrics.py` expects:
- real images in plane folders named:
  - `1_plane_trans_thalamic`
  - `2_plane_trans_cerebellum`
  - `3_plane_trans_ventricular`
- synthetic images in class folders named:
  - `trans_thalamic`
  - `trans_cerebellum`
  - `trans_ventricular`

Synthetic root should be the folder that directly contains those 3 synthetic class directories (for example the output from `generate.sh`).

Install extra metric dependencies if needed:

```bash
pip install torchmetrics torch-fidelity pytorch-msssim pillow tqdm
```

Example:

```bash

python calculate_metrics/metrics.py \
  --real_root ../fetal_ultrasound/img \
  --synthetic_root ../fetal_ultrasound/generated/sdxl_lora_r16_a16_e8 \
  --output_json ../fetal_ultrasound/generated/sdxl_lora_r16_a16_e8/metrics_report.json
```

## Dataset

You can access demo dataset via this [link](https://www.dropbox.com/scl/fi/iej4i1e7pltikkyf338ex/generated_fetal_brain.zip?rlkey=qtp92jnqeqpw4rfkt72u9pyl7&st=e0hchm6y&dl=0)