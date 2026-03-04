#!/bin/bash
#SBATCH --job-name=convnext_final
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s324381@studenti.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=51200
#SBATCH --ntasks-per-node=8
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
export PYTHONPATH=/home/dravida/Tesi_Davide_Augmentation/scripts:/home/dravida/Tesi_Davide_Augmentation:${PYTHONPATH:-}

echo "[GPU CHECK] nvidia-smi"
nvidia-smi || { echo "[FATAL] nvidia-smi non disponibile"; exit 1; }

python - <<'PY'
import sys
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print(f"[GPU CHECK] TensorFlow GPUs: {gpus}")
if not gpus:
	print("[FATAL] TensorFlow non vede GPU. Interrompo il job per evitare training su CPU.")
	sys.exit(1)
PY

python scripts/train_final_layers.py --model_name ConvNeXt --TRAIN_EPOCH 300 --dropout_rate 0.3793651429086242 --learning_rate 0.0004809590540741948 --l2_reg 0.00200041712955451