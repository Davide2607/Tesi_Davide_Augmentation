#!/bin/bash
#SBATCH --job-name=inspect_h5
#SBATCH --output=logs/inspect_h5_%j.out
#SBATCH --error=logs/inspect_h5_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

module purge || true

# Activate your conda env (adjust if needed)
source ~/.bashrc
conda activate fer_augmentation

DATASET_PATH=${DATASET_PATH:-$HOME/data/dataset_unito.h5}
SAMPLE=${SAMPLE:-2000}

mkdir -p logs

python -u scripts/inspect_h5_dataset.py --path "$DATASET_PATH" --sample "$SAMPLE"
