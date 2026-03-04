#!/bin/bash
#SBATCH --job-name=convnext_gpu_preflight
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s324381@studenti.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/dravida/models/FER-Augmentation/out_err/convnext_gpu_preflight_%j.out
#SBATCH --error=/home/dravida/models/FER-Augmentation/out_err/convnext_gpu_preflight_%j.err

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

echo "[GPU CHECK] nvidia-smi"
nvidia-smi || { echo "[FATAL] nvidia-smi non disponibile"; exit 1; }

python - <<'PY'
import os
import sys
import ctypes
import tensorflow as tf

print("[GPU CHECK] Python:", sys.version.split()[0])
print("[GPU CHECK] TensorFlow:", tf.__version__)
print("[GPU CHECK] Build with CUDA:", tf.test.is_built_with_cuda())
print(f"[GPU CHECK] CONDA_PREFIX={os.environ.get('CONDA_PREFIX')}")
print(f"[GPU CHECK] LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')}")

libs_to_test = ["libcudart.so.12", "libcudnn.so.8", "libcublas.so.12"]
for lib_name in libs_to_test:
    try:
        ctypes.CDLL(lib_name)
        print(f"[GPU CHECK] OK load {lib_name}")
    except OSError as exc:
        print(f"[GPU CHECK] FAIL load {lib_name}: {exc}")

gpus = tf.config.list_physical_devices('GPU')
print(f"[GPU CHECK] TensorFlow GPUs: {gpus}")
if not gpus:
    print("[FATAL] TensorFlow non vede GPU.")
    sys.exit(1)

print("[OK] Compatibilità GPU TensorFlow verificata.")
PY
