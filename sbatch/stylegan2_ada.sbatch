#!/bin/bash
#SBATCH -p gpu_a40
#SBATCH -J stylegan2_ada
#SBATCH -o ~/models/FER-Augmentation/out_err/stylegan2_%j.out
#SBATCH -e ~/models/FER-Augmentation/out_err/stylegan2_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 12:00:00

# Attiva conda (aggiorna il path se diverso)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fer_augmentation

# Cartella progetto
echo "CD to project"
cd ~/Tesi_Davide || exit 1

# Parametri
DATA_ZIP=~/data/DISGUST.zip            # zip del dataset di UNA classe (usa --source-dir se manca)
SOURCE_DIR=                            # opzionale: es. ~/data/DISGUST per creare lo zip se non esiste
OUTDIR=~/models/stylegan2_runs         # output training
SEEDS=0-199                            # intervallo seed per generate
KIMG=300                               # durata training
GEN_OUT=~/models/stylegan2_generated   # output immagini generate
RESUME=                                # opzionale: snapshot .pkl da cui riprendere (default ffhq)
GPUS=1
AUG=ada
CFG=auto

# Costruisci comando
CMD=(python StyleGAN2_ADA_FER.py \
  --data-zip "$DATA_ZIP" \
  --outdir "$OUTDIR" \
  --gpus "$GPUS" \
  --kimg "$KIMG" \
  --aug "$AUG" \
  --cfg "$CFG" \
  --seeds "$SEEDS" \
  --generate-out "$GEN_OUT")

if [[ -n "$SOURCE_DIR" ]]; then
  CMD+=(--source-dir "$SOURCE_DIR")
fi
if [[ -n "$RESUME" ]]; then
  CMD+=(--resume "$RESUME")
fi

set -x
"${CMD[@]}"
