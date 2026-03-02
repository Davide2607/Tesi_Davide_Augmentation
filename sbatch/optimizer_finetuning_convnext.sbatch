#!/bin/bash
#SBATCH -p gpu_a40
#SBATCH -J convnext_opt_ft
#SBATCH -o /home/dravida/models/FER-Augmentation/out_err/convnext_opt_ft_%j.out
#SBATCH -e /home/dravida/models/FER-Augmentation/out_err/convnext_opt_ft_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=50G
#SBATCH -t 24:00:00

set -eo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fer_augmentation
set -u

mkdir -p /home/dravida/models/FER-Augmentation/out_err
cd /home/dravida/Tesi_Davide_Augmentation || exit 1
export PYTHONPATH=/home/dravida/Tesi_Davide_Augmentation/scripts:/home/dravida/Tesi_Davide_Augmentation:${PYTHONPATH:-}

python scripts/optimizer_finetuning.py --model_name ConvNeXt --learning_rate 0.0004809590540741948
