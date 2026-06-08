#!/bin/bash
#SBATCH -p gpu_a40
#SBATCH -J convnext_opt_ft
#SBATCH -o /home/dravida/models/FER-Augmentation/out_err/convnext_opt_ft_%j.out
#SBATCH -e /home/dravida/models/FER-Augmentation/out_err/convnext_opt_ft_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 24:00:00

set -eo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fer_augmentation
set -u

mkdir -p /home/dravida/models/FER-Augmentation/out_err
cd /home/dravida/Tesi_Davide_Augmentation || exit 1
export DATASET_H5="${DATASET_H5:-/home/dravida/data/database_primo_metodo.h5}"
export TEST_DATASET_H5="${TEST_DATASET_H5:-/home/dravida/data/KDEF.h5}"
export MODEL_DIR="${MODEL_DIR:-model}"
export PYTHONPATH=/home/dravida/Tesi_Davide_Augmentation/scripts:/home/dravida/Tesi_Davide_Augmentation:${PYTHONPATH:-}
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CONDA_PREFIX}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

echo "[INFO] DATASET_H5=${DATASET_H5}"
echo "[INFO] TEST_DATASET_H5=${TEST_DATASET_H5}"
echo "[INFO] MODEL_DIR=${MODEL_DIR}"

# How many backbone layers to unfreeze (ConvNeXt only).
export FT_UNFREEZE="${FT_UNFREEZE:-241}"
echo "[INFO] FT_UNFREEZE=${FT_UNFREEZE}"

# Finetuning optimizer loads the final-layers model from MODEL_DIR.
if [ ! -d "${MODEL_DIR}/pretrained_ConvNeXt_final_layers" ] && [ ! -f "${MODEL_DIR}/pretrained_ConvNeXt_final_layers.keras" ] && [ ! -f "${MODEL_DIR}/pretrained_ConvNeXt_final_layers.h5" ] && [ ! -f "${MODEL_DIR}/pretrained_ConvNeXt_final_layers_weights.h5" ]; then
	echo "[FATAL] Pretrained final-layers model not found in MODEL_DIR=${MODEL_DIR}."
	echo "        Expected one of:"
	echo "          - ${MODEL_DIR}/pretrained_ConvNeXt_final_layers (SavedModel dir)"
	echo "          - ${MODEL_DIR}/pretrained_ConvNeXt_final_layers.keras"
	echo "          - ${MODEL_DIR}/pretrained_ConvNeXt_final_layers.h5"
	echo "          - ${MODEL_DIR}/pretrained_ConvNeXt_final_layers_weights.h5"
	echo "        Run sbatch/convnext_final.sbatch first with the same MODEL_DIR."
	exit 2
fi

echo "[GPU CHECK] nvidia-smi"
nvidia-smi || { echo "[FATAL] nvidia-smi non disponibile"; exit 1; }

python - <<'PY'
import sys
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"[GPU CHECK] TensorFlow GPUs: {gpus}")
if not gpus:
	print("[FATAL] TensorFlow non vede GPU. Interrompo il job per evitare tuning su CPU.")
	sys.exit(1)
PY

python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("bayes_opt") is None:
	print("[SETUP] Installing bayesian-optimization...")
	subprocess.check_call([sys.executable, "-m", "pip", "install", "bayesian-optimization"])
else:
	print("[SETUP] bayes_opt already available.")
PY

python -u scripts/optimizer_finetuning.py --model_name ConvNeXt --learning_rate 0.0004809590540741948 --trial_epochs 8 --init_points 2 --n_iter 15
