#!/bin/bash
#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export CONFIG_PATH="configs/wn18rr_finetune_server.yaml"
export RAW_DATA_DIR="data/WN18RR"
export DATASET_NAME="wn18rr"

export RUN_PREPROCESS=1
export RUN_CONTEXT=1
export RUN_STRUCTURAL=1
export RUN_TRAIN=1
export RUN_EVAL=1

export STRUCT_MODEL="RotatE"
export STRUCT_EPOCHS=50

bash "$PROJECT_ROOT/scripts/run.sh"
