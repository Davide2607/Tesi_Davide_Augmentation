#!/bin/bash
#SBATCH -p gpu_a40
#SBATCH -J stylegan2_ada_fear
#SBATCH -o /home/dravida/models/FER-Augmentation/out_err/stylegan2_fear_%j.out
#SBATCH -e /home/dravida/models/FER-Augmentation/out_err/stylegan2_fear_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 24:00:00

set -euo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stylegan2
python -m pip install -q ninja
PY=$(which python)

mkdir -p /home/dravida/models/FER-Augmentation/out_err
cd /home/dravida/Tesi_Davide_Augmentation || exit 1

SCRIPT_PATH=""
for CANDIDATE in \
  "$PWD/StyleGAN2_ADA_FER.py" \
  "$PWD/stylegan2_ada_fer.py" \
  "$PWD/neaural_networks/StyleGAN2_ADA_FER.py" \
  "$PWD/neaural_networks/stylegan2_ada_fer.py" \
  "$PWD/scripts/StyleGAN2_ADA_FER.py" \
  "$PWD/scripts/stylegan2_ada_fer.py"
do
  if [[ -f "$CANDIDATE" ]]; then
    SCRIPT_PATH="$CANDIDATE"
    break
  fi
done

if [[ -z "$SCRIPT_PATH" ]]; then
  echo "[FATAL] Script StyleGAN2_ADA_FER.py non trovato in $PWD"
  exit 1
fi

export TORCH_EXTENSIONS_DIR=/home/dravida/.cache/torch_extensions/stylegan2
rm -rf "$TORCH_EXTENSIONS_DIR"

# Parametri per FEAR
DATA_ZIP=${DATA_ZIP:-~/data/FEAR_stylegan2.zip}
SOURCE_DIR=${SOURCE_DIR:-~/data/FEAR}
OUTDIR=${OUTDIR:-~/models/stylegan2_runs_fear}
SEEDS=${SEEDS:-0-2999}
KIMG=${KIMG:-50}
GEN_OUT=${GEN_OUT:-~/Immagini/stylegan2_generated_fear}
RESUME=${RESUME:-none}
GPUS=${GPUS:-1}
AUG=${AUG:-noaug}
CFG=${CFG:-stylegan2}
WORKERS=${WORKERS:-2}

CMD=($PY "$SCRIPT_PATH" \
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
