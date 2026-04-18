#!/bin/bash
#SBATCH --job-name=gwm-wn18rr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --output=media/vhviet03/logs/gwm/slurm-%x-%j.out
#SBATCH --error=media/vhviet03/logs/gwm/slurm-%x-%j.err
#SBATCH --mail-user=ngphuc222@clc.fitus.edu.vn
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --qos=short

set -euo pipefail

# Optional: uncomment and adjust if your cluster requires modules.
spack load cuda@11.8.0

# Configure these for your school cluster.
export PROJECT_ROOT="GWM-Research"
export CONFIG_PATH="GWM-Research/configs/wn18rr_finetune_server.yaml"
export RAW_DATA_DIR="GWM-Research/data/WN18RR"
export DATASET_NAME="wn18rr"
export CONDA_ENV_NAME="gwm"

# Optional stage toggles: set to 0 to skip a stage.
export RUN_PREPROCESS=1
export RUN_CONTEXT=1
export RUN_TRAIN=1
export RUN_EVAL=1

# Conda activation.
source /media/vhviet03/miniconda/bin/activate
conda activate "$CONDA_ENV_NAME"

cd /media/vhviet03
bash "$PROJECT_ROOT/scripts/run.sh"
