#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD_SCRIPTS_DIR="${SD_SCRIPTS_DIR:-SCRIPT_DIR/sd-scripts}"

STATE_DIR="${1:-SCRIPT_DIR/fetal_ultrasound/output/sdxl_lora_r16_a16_e8-state}"
TRAIN_CONFIG_JSON="${2:-$SCRIPT_DIR/sdxl_config.json}"
OUTDIR="${3:-$SCRIPT_DIR/fetal_ultrasound/generated/sdxl_lora_r16_a16_e8}"

LORA_MUL="${LORA_MUL:-1.0}"       
CFG_SCALE="${CFG_SCALE:-6.0}"    
STEPS="${STEPS:-28}"              
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-500}" # Number of samples per promt
BATCH_SIZE="${BATCH_SIZE:-4}"     
BASE_SEED="${BASE_SEED:-1234}"
SAMPLER="${SAMPLER:-euler_a}"

if [ ! -d "$STATE_DIR" ] && [ ! -f "${STATE_DIR%.*}" ]; then
  if [ -f "$STATE_DIR" ]; then
     echo "Input is a file, assuming weights."
  else
     echo "State dir or weights file not found: $STATE_DIR" >&2
     exit 1
  fi
fi

if [ ! -f "$TRAIN_CONFIG_JSON" ]; then
  echo "Train config JSON not found: $TRAIN_CONFIG_JSON" >&2
  exit 1
fi

if [ ! -f "$SD_SCRIPTS_DIR/sdxl_gen_img.py" ]; then
  echo "sd-scripts not found or incomplete in: $SD_SCRIPTS_DIR" >&2
  exit 1
fi


LORA_WEIGHTS_CANDIDATE="${STATE_DIR%-state}.safetensors"
LORA_WEIGHTS_INSIDE="$STATE_DIR/model.safetensors"

if [ -f "$LORA_WEIGHTS_CANDIDATE" ]; then
  LORA_WEIGHTS="$LORA_WEIGHTS_CANDIDATE"
elif [ -f "$LORA_WEIGHTS_INSIDE" ]; then
  LORA_WEIGHTS="$LORA_WEIGHTS_INSIDE"
elif [ -f "$STATE_DIR" ] && [[ "$STATE_DIR" == *.safetensors ]]; then
  LORA_WEIGHTS="$STATE_DIR"
else
  echo "No LoRA weights found!" >&2
  echo "Checked: $LORA_WEIGHTS_CANDIDATE"
  echo "Checked: $LORA_WEIGHTS_INSIDE"
  exit 1
fi

echo "Using LoRA weights: $LORA_WEIGHTS"
mkdir -p "$OUTDIR"

readarray -t CFG_LINES < <(
  python3 - "$TRAIN_CONFIG_JSON" <<'PY'
import json
import sys

def is_true(value):
    if isinstance(value, bool): return value
    if isinstance(value, str): return value.strip().lower() in {"1", "true", "yes", "on"}
    return False

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)

ckpt = str(cfg.get("pretrained_model_name_or_path", "")).strip()
resolution = str(cfg.get("max_resolution", "1024,1024"))
seed = str(cfg.get("seed", "1234"))
mixed_precision = str(cfg.get("mixed_precision", "fp16")).strip().lower()
no_half_vae = "1" if is_true(cfg.get("no_half_vae")) else "0"

parts = [p.strip() for p in resolution.split(",")]
if len(parts) == 2 and all(parts):
    w, h = parts[0], parts[1]
else:
    w, h = "1024", "1024"

print(ckpt)
print(w)
print(h)
print(seed)
print(mixed_precision)
print(no_half_vae)
PY
)

BASE_CKPT="${CFG_LINES[0]:-}"
IMG_W="${CFG_LINES[1]:-1024}"
IMG_H="${CFG_LINES[2]:-1024}"
SEED_FROM_CFG="${CFG_LINES[3]:-1234}"
MIXED_PRECISION="${CFG_LINES[4]:-fp16}"
NO_HALF_VAE="${CFG_LINES[5]:-0}"


PRECISION_FLAG=()
MIXED_PRECISION="bf16" 

case "$MIXED_PRECISION" in
  bf16)
    PRECISION_FLAG=(--bf16)
    ;;
  fp16|"")
    PRECISION_FLAG=(--fp16)
    ;;
  *)
    PRECISION_FLAG=(--fp16)
    ;;
esac

VAE_FLAGS=()
if [ "$NO_HALF_VAE" = "1" ]; then
  VAE_FLAGS+=(--no_half_vae)
fi

PROMPTS=(
  "trans thalamic plane, fetal ultrasound scan, fetal brain, quality good, artifacts speckle noise, calvarium rim present, midline echo present, thalami present, csp present, lateral ventricle margins partial, third ventricle not clearly seen, posterior fossa not clearly seen, limitations posterior fossa obscured by shadowing"
  "trans cerebellum plane, fetal ultrasound scan, fetal brain, quality good, artifacts speckle noise, calvarium rim present, midline echo present, cerebellum present, posterior fossa not clearly seen, cisterna magna not clearly seen, csp not clearly seen, thalami not clearly seen"
  "trans ventricular plane, fetal ultrasound scan, fetal brain, quality good, artifacts speckle noise, calvarium rim present, midline echo present, lateral ventricle present, atrium region present, choroid plexus present, csp present, posterior fossa not clearly seen, limitations posterior fossa obscured by acoustic shadow"
)

if [ "${BASE_SEED}" = "1234" ] && [ -n "${SEED_FROM_CFG}" ]; then
  BASE_SEED="${SEED_FROM_CFG}"
fi

echo "------------------------------------------------"
echo "Starting Inference"
echo "Base Checkpoint: $BASE_CKPT"
echo "LoRA: $LORA_WEIGHTS (Strength: $LORA_MUL)"
echo "Resolution: ${IMG_W}x${IMG_H}"
echo "Precision: ${PRECISION_FLAG[*]}"
echo "Images per prompt: $SAMPLES_PER_PROMPT"
echo "------------------------------------------------"

for i in "${!PROMPTS[@]}"; do
  prompt="${PROMPTS[$i]}"
  plane_name="$(printf '%s' "$prompt" | sed -E 's/[[:space:]]+plane,.*$//' | tr '[:upper:]' '[:lower:]' | tr ' ' '_')"
  if [ -z "$plane_name" ]; then
    plane_name="prompt_$((i + 1))"
  fi
  prompt_outdir="$OUTDIR/$plane_name"
  mkdir -p "$prompt_outdir"
  
  echo "Generating for $plane_name: ${prompt:0:50}..."

  (
    cd "$SD_SCRIPTS_DIR"
    
    python3 sdxl_gen_img.py \
      --ckpt "$BASE_CKPT" \
      --prompt "$prompt" \
      --outdir "$prompt_outdir" \
      --images_per_prompt "$SAMPLES_PER_PROMPT" \
      --batch_size "$BATCH_SIZE" \
      --W "$IMG_W" \
      --H "$IMG_H" \
      --seed "$BASE_SEED" \
      "${PRECISION_FLAG[@]}" \
      "${VAE_FLAGS[@]}" \
      --sampler "$SAMPLER" \
      --steps "$STEPS" \
      --scale "$CFG_SCALE" \
      --network_module networks.lora \
      --network_weights "$LORA_WEIGHTS" \
      --network_mul "$LORA_MUL"
  )
done

echo "Done. Generated images saved to: $OUTDIR"
