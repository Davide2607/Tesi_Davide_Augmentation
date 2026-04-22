#!/bin/bash
#SBATCH -p gpu_a40
#SBATCH -J convnext_opt_final
#SBATCH -o /home/dravida/models/FER-Augmentation/out_err/convnext_opt_final_%j.out
#SBATCH -e /home/dravida/models/FER-Augmentation/out_err/convnext_opt_final_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=20G
#SBATCH -t 12:00:00

set -eo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fer_augmentation
set -u

mkdir -p /home/dravida/models/FER-Augmentation/out_err
cd /home/dravida/Tesi_Davide_Augmentation || exit 1
export DATASET_H5="${DATASET_H5:-/home/dravida/data/dataset.h5}"
export PYTHONPATH=/home/dravida/Tesi_Davide_Augmentation/scripts:/home/dravida/Tesi_Davide_Augmentation:${PYTHONPATH:-}
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CONDA_PREFIX}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

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

python scripts/optimizer_final_layers.py --model_name ConvNeXt --trial_epochs 8 --init_points 4 --n_iter 20
