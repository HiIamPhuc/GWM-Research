#!/bin/bash

set -euo pipefail

# Expected environment variables (can be exported by job.slurm):
# - PROJECT_ROOT
# - CONFIG_PATH
# - RAW_DATA_DIR
# - DATASET_NAME
# - RUN_PREPROCESS, RUN_CONTEXT, RUN_TRAIN, RUN_EVAL
# - CACHE_ROOT

PROJECT_ROOT="${PROJECT_ROOT:-}"
CONFIG_PATH="${CONFIG_PATH:-configs/wn18rr_finetune_server.yaml}"
RAW_DATA_DIR="${RAW_DATA_DIR:-}"
DATASET_NAME="${DATASET_NAME:-wn18rr}"

RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
RUN_CONTEXT="${RUN_CONTEXT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

if [[ -z "$PROJECT_ROOT" ]]; then
    echo "Error: PROJECT_ROOT is not set."
    exit 1
fi

if [[ ! -f "$PROJECT_ROOT/$CONFIG_PATH" ]]; then
    echo "Error: config file not found at $PROJECT_ROOT/$CONFIG_PATH"
    exit 1
fi

if [[ "$RUN_PREPROCESS" == "1" ]] && [[ -z "$RAW_DATA_DIR" ]]; then
    echo "Error: RAW_DATA_DIR must be set when RUN_PREPROCESS=1"
    exit 1
fi

export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
mkdir -p "$TORCH_HOME" "$HF_HOME"

cd "$PROJECT_ROOT"

read_yaml_key() {
    local config_file="$1"
    local key="$2"
    python - "$config_file" "$key" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
key = sys.argv[2]
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}
value = cfg.get(key, "")
if isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
PY
}

DATA_DIR="$(read_yaml_key "$CONFIG_PATH" "data_dir")"
OUTPUT_DIR="$(read_yaml_key "$CONFIG_PATH" "output_dir")"

if [[ -z "$DATA_DIR" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: data_dir/output_dir must be set in $CONFIG_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "GWM training pipeline"
echo "project_root : $PROJECT_ROOT"
echo "config       : $CONFIG_PATH"
echo "raw_data_dir : $RAW_DATA_DIR"
echo "data_dir     : $DATA_DIR"
echo "output_dir   : $OUTPUT_DIR"
echo "dataset_name : $DATASET_NAME"
echo "========================================"

if [[ "$RUN_PREPROCESS" == "1" ]]; then
    echo "[1/4] Preprocessing raw dataset"
    python utils/preprocess_data.py \
        --data_dir "$RAW_DATA_DIR" \
        --output_dir "$DATA_DIR" \
        --dataset "$DATASET_NAME"
else
    echo "[1/4] Skipped preprocessing"
fi

if [[ "$RUN_CONTEXT" == "1" ]]; then
    echo "[2/4] Computing context_ids.pt from config"
    python utils/compute_context.py \
        --config "$CONFIG_PATH" \
        --data_dir "$DATA_DIR"
else
    echo "[2/4] Skipped context computation"
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
    echo "[3/4] Training"
    python train.py \
        --config "$CONFIG_PATH" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR"
else
    echo "[3/4] Skipped training"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
    echo "[4/4] Evaluating"
    python evaluate.py \
        --config "$CONFIG_PATH" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR"
else
    echo "[4/4] Skipped evaluation"
fi

echo "Pipeline finished successfully."

