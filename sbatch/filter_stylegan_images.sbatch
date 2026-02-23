#!/bin/bash
#SBATCH -p gpu_a40
#SBATCH -J pack_stylegan
#SBATCH -o /home/dravida/models/FER-Augmentation/out_err/filter_%j.out
#SBATCH -e /home/dravida/models/FER-Augmentation/out_err/filter_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00

set -euo pipefail

# Load Python environment
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fer_augmentation
PY=$(which python)

# Directories
cd /home/dravida/Tesi_Davide_Augmentation || exit 1

# CONFIGURAZIONE - Modifica questi parametri
INPUT_DIR=~/models/stylegan2_generated_disgust        # cartella con PNG generati
TARGET_CLASS=1                                        # 1=DISGUST (0=ANGRY, 1=DISGUST, 2=FEAR, etc.)
OUTPUT_DIR=~/models/STYLE_GAN_augmentation_disgust    # output compatibile con MERGE_AUGMENTED_DATA.ipynb
MAX_IMAGES=5000                                       # usa solo 5000 immagini sintetiche
RANDOM_SEED=42                                        # campionamento riproducibile

echo "[INFO] Python: $PY"
"$PY" -V
echo "[INFO] INPUT_DIR=$INPUT_DIR"
[[ -d "$INPUT_DIR" ]] || { echo "[ERROR] Input dir not found: $INPUT_DIR"; exit 1; }
echo "[INFO] PNG count: $(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.png' | wc -l)"

# Pack no-filter (usa tutte le immagini generate)
$PY pack_stylegan_images.py \
  --input-dir "$INPUT_DIR" \
  --target-class "$TARGET_CLASS" \
  --max-images "$MAX_IMAGES" \
  --random-seed "$RANDOM_SEED" \
  --output-dir "$OUTPUT_DIR"

echo "=== Pack complete ===" 
echo "All generated images saved to: $OUTPUT_DIR"
echo "Use MERGE_AUGMENTED_DATA.ipynb to merge with original dataset"
