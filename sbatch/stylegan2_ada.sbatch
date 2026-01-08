#!/bin/bash
#SBATCH -p gpu_a40
#SBATCH -J stylegan2_ada
#SBATCH -o /home/dravida/models/FER-Augmentation/out_err/stylegan2_%j.out
#SBATCH -e /home/dravida/models/FER-Augmentation/out_err/stylegan2_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 12:00:00

# Usa l'interprete Python dell'env
PY=/home/dravida/.conda/envs/fer_augmentation/bin/python

# Cartella progetto
mkdir -p /home/dravida/models/FER-Augmentation/out_err
cd /home/dravida/Tesi_Davide_Augmentation || exit 1

# Parametri preconfigurati (DISGUST)
DATA_ZIP=~/data/DISGUST.zip                 # zip del dataset di UNA classe
SOURCE_DIR=~/data/DISGUST                   # usato se il zip non esiste (png estratti da export_h5_class)
OUTDIR=~/models/stylegan2_runs_disgust      # output training
SEEDS=0-999                                 # intervallo seed per generate
KIMG=600                                    # durata training in kimg
GEN_OUT=~/models/stylegan2_generated_disgust# output immagini generate
RESUME=                                     # opzionale: snapshot .pkl da cui riprendere (default ffhq)
GPUS=1
AUG=ada
CFG=auto

# Comando
CMD=($PY StyleGAN2_ADA_FER.py \
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
