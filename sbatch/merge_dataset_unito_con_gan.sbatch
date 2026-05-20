#!/bin/bash
#SBATCH --job-name=merge_unito_gan
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s324381@studenti.polito.it
#SBATCH --partition=cpu_skylake
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/dravida/models/FER-Augmentation/out_err/merge_unito_gan_%j.out
#SBATCH --error=/home/dravida/models/FER-Augmentation/out_err/merge_unito_gan_%j.err

set -eo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fer_augmentation
set -u

cd /home/dravida/Tesi_Davide_Augmentation || exit 1
mkdir -p /home/dravida/models/FER-Augmentation/out_err
export PYTHONPATH=/home/dravida/Tesi_Davide_Augmentation/scripts:/home/dravida/Tesi_Davide_Augmentation:${PYTHONPATH:-}

A_PATH=${A_PATH:-/home/dravida/data/dataset.h5}
B_PATH=${B_PATH:-/home/dravida/data/database_zero.h5}
OUT_PATH=${OUT_PATH:-/home/dravida/data/dataset_unito_con_GAN.h5}
SEED=${SEED:-42}
CHUNK_SIZE=${CHUNK_SIZE:-2048}
NO_SHUFFLE=${NO_SHUFFLE:-1}
B_MAX_PER_CLASS=${B_MAX_PER_CLASS:-0}
B_MAX_TOTAL=${B_MAX_TOTAL:-0}

echo "[INFO] A_PATH=${A_PATH}"
echo "[INFO] B_PATH=${B_PATH}"
echo "[INFO] OUT_PATH=${OUT_PATH}"
echo "[INFO] B_MAX_PER_CLASS=${B_MAX_PER_CLASS} B_MAX_TOTAL=${B_MAX_TOTAL}"

EXTRA_ARGS=()
if [ "$NO_SHUFFLE" -eq 1 ]; then
  EXTRA_ARGS+=(--no-shuffle)
fi

if [ "$B_MAX_PER_CLASS" -gt 0 ]; then
  EXTRA_ARGS+=(--b-max-per-class "$B_MAX_PER_CLASS")
fi

if [ "$B_MAX_TOTAL" -gt 0 ]; then
  EXTRA_ARGS+=(--b-max-total "$B_MAX_TOTAL")
fi

python -u scripts/merge_h5_datasets.py \
  --a "${A_PATH}" \
  --b "${B_PATH}" \
  --out "${OUT_PATH}" \
  --seed "${SEED}" \
  --chunk-size "${CHUNK_SIZE}" \
  "${EXTRA_ARGS[@]}"
