#!/bin/bash
#SBATCH --job-name=convnext_final
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s324381@studenti.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=20480
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/dravida/models/FER-Augmentation/out_err/convnext_final_%j.out
#SBATCH --error=/home/dravida/models/FER-Augmentation/out_err/convnext_final_%j.err

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

python scripts/train_final_layers.py --model_name ConvNeXt --TRAIN_EPOCH 300 --dropout_rate 0.5 --learning_rate 0.00024041677639819288 --l2_reg 0.016443457513284063 --seed "$SEED"