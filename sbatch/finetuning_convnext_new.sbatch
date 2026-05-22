#!/bin/bash
#SBATCH --job-name=convnext_finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s324381@studenti.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=20480
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/dravida/models/FER-Augmentation/out_err/convnext_ft_%j.out
#SBATCH --error=/home/dravida/models/FER-Augmentation/out_err/convnext_ft_%j.err

set -eo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fer_augmentation
set -u

cd /home/dravida/Tesi_Davide_Augmentation || exit 1
mkdir -p /home/dravida/models/FER-Augmentation/out_err
export DATASET_H5="${DATASET_H5:-/home/dravida/data/database_secondo_metodo.h5}"
export MODEL_DIR="${MODEL_DIR:-model}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export FIT_WORKERS="${FIT_WORKERS:-4}"
export FIT_MULTIPROCESSING="${FIT_MULTIPROCESSING:-1}"
export FIT_MAX_QUEUE_SIZE="${FIT_MAX_QUEUE_SIZE:-16}"

echo "[INFO] DATASET_H5=${DATASET_H5}"
echo "[INFO] MODEL_DIR=${MODEL_DIR}"

# Finetuning loads the final-layers model from MODEL_DIR.
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

FT_LR="${FT_LR:-0.000288331559788701}"
FT_DROPOUT="${FT_DROPOUT:-0.1624074561769746}"
FT_L2="${FT_L2:-0.016443457513284063}"
echo "[INFO] FT_LR=${FT_LR} FT_DROPOUT=${FT_DROPOUT} FT_L2=${FT_L2}"

# Finetuning-only augmentation overrides (train only).
# These env vars are read by scripts/loading_data.py.
# You can still override them at submission time.
export AUG_ROTATION_RANGE="${AUG_ROTATION_RANGE:-8}"
export AUG_WIDTH_SHIFT_RANGE="${AUG_WIDTH_SHIFT_RANGE:-0.08}"
export AUG_SHEAR_RANGE="${AUG_SHEAR_RANGE:-0.08}"
export AUG_HORIZONTAL_FLIP="${AUG_HORIZONTAL_FLIP:-1}"
export AUG_FILL_MODE="${AUG_FILL_MODE:-nearest}"
SEED="${SEED:-42}"
export PYTHONPATH=/home/dravida/Tesi_Davide_Augmentation/scripts:/home/dravida/Tesi_Davide_Augmentation:${PYTHONPATH:-}
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CONDA_PREFIX}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

echo "[GPU CHECK] nvidia-smi"
nvidia-smi || { echo "[FATAL] nvidia-smi non disponibile"; exit 1; }

python - <<'PY'
import sys
import os
import ctypes
import tensorflow as tf

libs_to_test = ["libcudart.so.12", "libcudnn.so.8", "libcublas.so.12"]
for lib_name in libs_to_test:
	try:
		ctypes.CDLL(lib_name)
		print(f"[GPU CHECK] OK load {lib_name}")
	except OSError as e:
		print(f"[GPU CHECK] FAIL load {lib_name}: {e}")

print(f"[GPU CHECK] CONDA_PREFIX={os.environ.get('CONDA_PREFIX')}")
print(f"[GPU CHECK] LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')}")
gpus = tf.config.list_physical_devices('GPU')
print(f"[GPU CHECK] TensorFlow GPUs: {gpus}")
if not gpus:
	print("[FATAL] TensorFlow non vede GPU. Interrompo il job per evitare training su CPU.")
	sys.exit(1)
PY

python scripts/finetuning.py --model_name ConvNeXt --FT_EPOCH 300 --dropout_rate "${FT_DROPOUT}" --learning_rate "${FT_LR}" --l2_reg "${FT_L2}" --seed "$SEED"