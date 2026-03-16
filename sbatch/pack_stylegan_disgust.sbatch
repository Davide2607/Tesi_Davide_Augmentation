#!/bin/bash
#SBATCH -p gpu_a40
#SBATCH -J pack_stylegan_disgust
#SBATCH -o /home/dravida/models/FER-Augmentation/out_err/pack_disgust_%j.out
#SBATCH -e /home/dravida/models/FER-Augmentation/out_err/pack_disgust_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00

set -eo pipefail

# Load Python environment
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fer_augmentation
set -u
PY=$(which python)

# Directories
cd /home/dravida/Tesi_Davide_Augmentation || exit 1

# CONFIG
INPUT_DIR=~/models/stylegan2_generated_disgust        # cartella con PNG generate DISGUST
TARGET_CLASS=1                                        # 1=DISGUST
OUTPUT_DIR=~/models/STYLE_GAN_augmentation_disgust    # output .npy separati DISGUST
MAX_IMAGES=${MAX_IMAGES:-5000}                        # override: MAX_IMAGES=10000 sbatch ...
RANDOM_SEED=${RANDOM_SEED:-42}                        # override: RANDOM_SEED=123 sbatch ...

echo "[INFO] Python: $PY"
"$PY" -V
echo "[INFO] INPUT_DIR=$INPUT_DIR"
[[ -d "$INPUT_DIR" ]] || { echo "[ERROR] Input dir not found: $INPUT_DIR"; exit 1; }
echo "[INFO] PNG count: $(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.png' | wc -l)"

$PY pack_stylegan_images.py \
  --input-dir "$INPUT_DIR" \
  --target-class "$TARGET_CLASS" \
  --max-images "$MAX_IMAGES" \
  --random-seed "$RANDOM_SEED" \
  --file-pattern "seed*.png" \
  --output-dir "$OUTPUT_DIR"

echo "=== Pack complete (DISGUST) ==="
echo "Saved to: $OUTPUT_DIR"
