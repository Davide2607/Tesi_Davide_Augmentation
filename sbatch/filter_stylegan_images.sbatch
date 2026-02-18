#!/bin/bash
#SBATCH -p gpu_a40
#SBATCH -J filter_stylegan
#SBATCH -o /home/dravida/models/FER-Augmentation/out_err/filter_%j.out
#SBATCH -e /home/dravida/models/FER-Augmentation/out_err/filter_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00

# Load Python environment
module load miniconda3/3.13.25
source activate stylegan2  # o il tuo env con tensorflow
PY=/home/dravida/.conda/envs/stylegan2/bin/python

# Directories
cd /home/dravida/Tesi_Davide_Augmentation || exit 1

# CONFIGURAZIONE - Modifica questi parametri
INPUT_DIR=~/models/stylegan2_generated_disgust        # cartella con PNG generati
MODEL_PATH=~/models/best_model_efficientnet.h5        # modello FER allenato
MODEL_NAME=EfficientNetB1                             # nome backbone
TARGET_CLASS=1                                        # 1=DISGUST (0=ANGRY, 1=DISGUST, 2=FEAR, etc.)
CONFIDENCE_THRESHOLD=0.7                              # soglia minima (0.6-0.8 consigliato)
OUTPUT_DIR=~/models/STYLE_GAN_augmentation            # output per immagini filtrate

# Esegui filtro
$PY filter_stylegan_images.py \
  --input-dir "$INPUT_DIR" \
  --model-path "$MODEL_PATH" \
  --model-name "$MODEL_NAME" \
  --target-class "$TARGET_CLASS" \
  --confidence-threshold "$CONFIDENCE_THRESHOLD" \
  --output-dir "$OUTPUT_DIR"

echo "=== Filter complete ===" 
echo "Filtered images saved to: $OUTPUT_DIR"
echo "Use MERGE_AUGMENTED_DATA.ipynb to merge with original dataset"
