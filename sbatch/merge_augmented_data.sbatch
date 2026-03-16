#!/bin/bash
#SBATCH --job-name=merge_aug_data
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s324381@studenti.polito.it
#SBATCH --partition=cpu_skylake
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/dravida/models/FER-Augmentation/out_err/merge_aug_data_%j.out
#SBATCH --error=/home/dravida/models/FER-Augmentation/out_err/merge_aug_data_%j.err

set -eo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fer_augmentation
set -u

cd /home/dravida/Tesi_Davide_Augmentation || exit 1
mkdir -p /home/dravida/models/FER-Augmentation/out_err

MAX_PER_DIR=${MAX_PER_DIR:-0}
OUT_PATH=${OUT_PATH:-/home/dravida/data/dataset_augmented.h5}
SEED=${SEED:-42}

EXTRA_ARGS=()
if [ "$MAX_PER_DIR" -gt 0 ]; then
  EXTRA_ARGS+=(--max-per-dir "$MAX_PER_DIR")
fi

python scripts/merge_augmented_data.py \
  --orig-path /home/dravida/data/dataset_original_backup.h5 \
  --synthetic-dir /home/dravida/models/STYLE_GAN_augmentation_disgust \
  --synthetic-dir /home/dravida/models/STYLE_GAN_augmentation_surprise \
  --out-path "$OUT_PATH" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"
