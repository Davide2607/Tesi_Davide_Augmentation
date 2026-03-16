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
# Risolvi il path dello script in modo robusto (case-sensitive su Linux)
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
  echo "[FATAL] Verifica che il file esista nel repository sul cluster."
  exit 1
fi
# Isola e pulisce la cache delle estensioni Torch per questo job
export TORCH_EXTENSIONS_DIR=/home/dravida/.cache/torch_extensions/stylegan2
rm -rf "$TORCH_EXTENSIONS_DIR"

# Parametri preconfigurati (SURPRISE)
DATA_ZIP=${DATA_ZIP:-~/data/SURPRISE_stylegan2.zip}            # override per classe/dataset
SOURCE_DIR=${SOURCE_DIR:-~/data/SURPRISE}                      # override opzionale
OUTDIR=${OUTDIR:-~/models/stylegan2_runs_surprise}             # output training
SEEDS=${SEEDS:-0-4999}                                          # es. 5000-9999 per secondo batch
KIMG=${KIMG:-600}                                               # es. 0/1 se vuoi solo generate
GEN_OUT=${GEN_OUT:-~/models/stylegan2_generated_surprise}       # output immagini generate
RESUME=${RESUME:-none}                                          # auto | none | /path/to/network-snapshot-xxxxxx.pkl
GPUS=${GPUS:-1}
AUG=${AUG:-noaug}
CFG=${CFG:-auto}
WORKERS=${WORKERS:-2}

# Comando
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
