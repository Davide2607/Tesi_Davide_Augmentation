#!/bin/bash
#SBATCH -p gpu_a40
#SBATCH -J stylegan2_ada
#SBATCH -o /home/dravida/models/FER-Augmentation/out_err/stylegan2_%j.out
#SBATCH -e /home/dravida/models/FER-Augmentation/out_err/stylegan2_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 24:00:00

set -euo pipefail

# Usa l'interprete Python dell'env corretto
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stylegan2
python -m pip install -q ninja
PY=$(which python)

# Cartella progetto
mkdir -p /home/dravida/models/FER-Augmentation/out_err
cd /home/dravida/Tesi_Davide_Augmentation || exit 1
# Isola e pulisce la cache delle estensioni Torch per questo job
export TORCH_EXTENSIONS_DIR=/home/dravida/.cache/torch_extensions/stylegan2
rm -rf "$TORCH_EXTENSIONS_DIR"

# Parametri preconfigurati (SURPRISE)
DATA_ZIP=~/data/SURPRISE_stylegan2.zip      # zip in formato StyleGAN2 (creato via dataset_tool)
SOURCE_DIR=~/data/SURPRISE                  # usato se il zip non esiste (png estratti da export_h5_class)
OUTDIR=~/models/stylegan2_runs_surprise     # output training
SEEDS=0-4999                                # intervallo seed per generate (5000 immagini)
KIMG=600                                    # durata training in kimg
GEN_OUT=~/models/stylegan2_generated_surprise # output immagini generate
RESUME=auto                                 # auto | none | /path/to/network-snapshot-xxxxxx.pkl
GPUS=1
AUG=noaug
CFG=auto
WORKERS=2

# Comando
CMD=($PY StyleGAN2_ADA_FER.py \
  --data-zip "$DATA_ZIP" \
  --outdir "$OUTDIR" \
  --gpus "$GPUS" \
  --kimg "$KIMG" \
  --aug "$AUG" \
  --cfg "$CFG" \
  --workers "$WORKERS" \
  --seeds "$SEEDS" \
  --generate-out "$GEN_OUT")

if [[ -n "$SOURCE_DIR" ]]; then
  CMD+=(--source-dir "$SOURCE_DIR")
fi

if [[ "$RESUME" == "none" ]]; then
  CMD+=(--no-resume)
elif [[ "$RESUME" == "auto" ]]; then
  if [[ -d "$OUTDIR" ]]; then
    LATEST_SNAPSHOT=$(find "$OUTDIR" -type f -name 'network-snapshot-*.pkl' | sort | tail -n 1 || true)
    if [[ -n "$LATEST_SNAPSHOT" ]]; then
      CMD+=(--resume "$LATEST_SNAPSHOT")
    fi
  fi
elif [[ -n "$RESUME" ]]; then
  CMD+=(--resume "$RESUME")
fi

set -x
"${CMD[@]}"
