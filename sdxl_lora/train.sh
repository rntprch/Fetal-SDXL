#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_JSON="${1:-$SCRIPT_DIR/config.json}"
SD_SCRIPTS_DIR="${SD_SCRIPTS_DIR:-$SCRIPT_DIR/sd-scripts}"

if [ $# -gt 0 ]; then
  shift
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

python3 - "$CONFIG_JSON" "$SD_SCRIPTS_DIR" "$@" <<'PY'
import json
import subprocess
import sys
import shlex


def is_true(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def is_empty(value):
    return value is None or (isinstance(value, str) and value.strip() == "")


def to_cli_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


config_path = sys.argv[1]
sd_scripts_dir = sys.argv[2]
extra_args = sys.argv[3:]

with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

is_sdxl = is_true(cfg.get("sdxl"))

train_py = "sdxl_train_network.py" if is_sdxl else "train_network.py"

cmd = ["accelerate", "launch"]

threads = cfg.get("num_cpu_threads_per_process")
if not is_empty(threads):
    cmd.extend(["--num_cpu_threads_per_process", to_cli_scalar(threads)])

cmd.append(f"{sd_scripts_dir}/{train_py}")

def add_opt(flag, key=None):
    key = key or flag.lstrip("-").replace("-", "_")
    value = cfg.get(key)
    if not is_empty(value):
        cmd.extend([flag, to_cli_scalar(value)])


def add_bool_flag(flag, key=None):
    key = key or flag.lstrip("-").replace("-", "_")
    if is_true(cfg.get(key)):
        cmd.append(flag)


def add_multi_opt(flag, key):
    """
    For args like --optimizer_args that accept multiple key=value tokens.
    Accepts:
      - list in JSON: ["a=b","c=d"]
      - string in JSON: "a=b c=d" (whitespace split, shell-like)
    """
    value = cfg.get(key)
    if is_empty(value):
        return
    if isinstance(value, list):
        parts = [to_cli_scalar(v) for v in value if not is_empty(v)]
    else:
        # shlex handles quoted strings safely if you ever need them
        parts = shlex.split(str(value))
    if parts:
        cmd.append(flag)
        cmd.extend(parts)

add_opt("--pretrained_model_name_or_path")
add_opt("--logging_dir")
add_opt("--train_data_dir")
add_opt("--reg_data_dir")
add_opt("--output_dir")
add_opt("--resolution", "max_resolution")

add_opt("--learning_rate")
add_opt("--lr_scheduler")
add_opt("--lr_warmup_steps", "lr_warmup")

add_opt("--train_batch_size")
add_opt("--gradient_accumulation_steps")
add_opt("--max_train_epochs")
add_opt("--save_every_n_epochs")

add_opt("--mixed_precision")
add_opt("--save_precision")
add_opt("--seed")

add_bool_flag("--cache_latents")
add_opt("--caption_extension")
add_bool_flag("--enable_bucket")
add_opt("--bucket_reso_steps")
add_bool_flag("--bucket_no_upscale")

add_bool_flag("--gradient_checkpointing")

add_opt("--save_model_as")
add_bool_flag("--save_state")
add_opt("--resume")

add_opt("--text_encoder_lr")
add_opt("--unet_lr")

add_opt("--network_dim")
add_opt("--network_alpha")
add_opt("--network_weights", "lora_network_weights")

add_opt("--output_name")
add_opt("--max_data_loader_n_workers")

add_opt("--optimizer_type", "optimizer")
add_multi_opt("--optimizer_args", "optimizer_args")

cmd.extend(["--network_module", "networks.lora"])

# SD1/2-only
if not is_sdxl:
    add_bool_flag("--v2")
    add_bool_flag("--v_parameterization")
    add_opt("--clip_skip")

# SDXL-only
if is_sdxl:
    add_bool_flag("--network_train_unet_only")
    add_bool_flag("--cache_text_encoder_outputs")
    add_bool_flag("--cache_text_encoder_outputs_to_disk")
    add_bool_flag("--no_half_vae")

# Fallback for configs that only define "epoch"
if "--max_train_epochs" not in cmd and not is_empty(cfg.get("epoch")):
    cmd.extend(["--max_train_epochs", to_cli_scalar(cfg["epoch"])])

print("Launching:", " ".join(cmd + extra_args))
subprocess.run(cmd + extra_args, check=True, cwd=sd_scripts_dir)
PY