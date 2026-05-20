#!/bin/bash
#SBATCH --job-name=build_unito_gan_filt
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s324381@studenti.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem=16384
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/dravida/models/FER-Augmentation/out_err/build_unito_gan_filt_%j.out
#SBATCH --error=/home/dravida/models/FER-Augmentation/out_err/build_unito_gan_filt_%j.err

set -eo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fer_augmentation
set -u

cd /home/dravida/Tesi_Davide_Augmentation || exit 1
mkdir -p /home/dravida/models/FER-Augmentation/out_err
export PYTHONPATH=/home/dravida/Tesi_Davide_Augmentation/scripts:/home/dravida/Tesi_Davide_Augmentation:${PYTHONPATH:-}
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CONDA_PREFIX}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

# Base dataset (REAL merged) to augment
ORIG_H5=${ORIG_H5:-/home/dravida/data/dataset_unito.h5}

# Model used as a "quality filter" (should be a decent NO-GAN model)
# Prefer *_weights.h5 when available (more robust than .keras deserialization)
MODEL_PATH=${MODEL_PATH:-/home/dravida/Tesi_Davide_Augmentation/model/pretrained_ConvNeXt_final_layers_weights.h5}
MODEL_NAME=${MODEL_NAME:-ConvNeXt}

# StyleGAN generated image folders
DISGUST_IN=${DISGUST_IN:-/home/dravida/models/stylegan2_generated_disgust}
SURPRISE_IN=${SURPRISE_IN:-/home/dravida/models/stylegan2_generated_surprise}

# Filtering threshold (higher = stricter)
CONF_THRESH=${CONF_THRESH:-0.85}

# Where to write filtered synthetic npy bundles
DISGUST_OUT=${DISGUST_OUT:-/home/dravida/models/STYLE_GAN_filtered_disgust_thr${CONF_THRESH}}
SURPRISE_OUT=${SURPRISE_OUT:-/home/dravida/models/STYLE_GAN_filtered_surprise_thr${CONF_THRESH}}

# Cap the amount of synthetic data included from each synthetic-dir (per class here)
MAX_PER_DIR=${MAX_PER_DIR:-500}

# Output H5
OUT_H5=${OUT_H5:-/home/dravida/data/dataset_unito_con_GAN_filtered_cap500.h5}
SEED=${SEED:-42}

echo "[INFO] ORIG_H5=${ORIG_H5}"
echo "[INFO] MODEL_PATH=${MODEL_PATH} (MODEL_NAME=${MODEL_NAME})"
echo "[INFO] DISGUST_IN=${DISGUST_IN} -> ${DISGUST_OUT}"
echo "[INFO] SURPRISE_IN=${SURPRISE_IN} -> ${SURPRISE_OUT}"
echo "[INFO] CONF_THRESH=${CONF_THRESH} MAX_PER_DIR=${MAX_PER_DIR}"
echo "[INFO] OUT_H5=${OUT_H5}"

# Quick checks
python -c "import os; import tensorflow as tf; print('[GPU]', tf.config.list_physical_devices('GPU')); print('[ORIG exists]', os.path.exists(os.path.expanduser('${ORIG_H5}'))); print('[MODEL exists]', os.path.exists(os.path.expanduser('${MODEL_PATH}')))"

# Class indices follow training order: 0=ANGRY 1=DISGUST 2=FEAR 3=HAPPY 4=NEUTRAL 5=SAD 6=SURPRISE
python -u scripts/filter_stylegan_images.py \
  --input-dir "${DISGUST_IN}" \
  --model-path "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --target-class 1 \
  --confidence-threshold "${CONF_THRESH}" \
  --output-dir "${DISGUST_OUT}"

python - <<PY
import numpy as np
from pathlib import Path
path = Path("${DISGUST_OUT}") / "synthetic_images.npy"
arr = np.load(path) if path.exists() else np.zeros((0,))
print(f"[CHECK] DISGUST accepted: {arr.shape[0] if hasattr(arr, 'shape') else 0}")
if not hasattr(arr, 'shape') or arr.shape[0] == 0:
    raise SystemExit("[FATAL] DISGUST filtering produced 0 images. Aborting.")
PY

python -u scripts/filter_stylegan_images.py \
  --input-dir "${SURPRISE_IN}" \
  --model-path "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --target-class 6 \
  --confidence-threshold "${CONF_THRESH}" \
  --output-dir "${SURPRISE_OUT}"

python - <<PY
import numpy as np
from pathlib import Path
path = Path("${SURPRISE_OUT}") / "synthetic_images.npy"
arr = np.load(path) if path.exists() else np.zeros((0,))
print(f"[CHECK] SURPRISE accepted: {arr.shape[0] if hasattr(arr, 'shape') else 0}")
if not hasattr(arr, 'shape') or arr.shape[0] == 0:
    raise SystemExit("[FATAL] SURPRISE filtering produced 0 images. Aborting.")
PY

EXTRA_ARGS=()
if [ "${MAX_PER_DIR}" -gt 0 ]; then
  EXTRA_ARGS+=(--max-per-dir "${MAX_PER_DIR}")
fi

python -u scripts/merge_augmented_data.py \
  --orig-path "${ORIG_H5}" \
  --synthetic-dir "${DISGUST_OUT}" \
  --synthetic-dir "${SURPRISE_OUT}" \
  --out-path "${OUT_H5}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"

echo "[DONE] Built ${OUT_H5}"
