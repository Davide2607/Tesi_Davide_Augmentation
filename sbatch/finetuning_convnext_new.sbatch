#!/bin/bash
#SBATCH --job-name=convnext_finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s324381@studenti.polito.it
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=51200
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --output=err_log/convnext_ft_%j.log
#SBATCH --error=err_log/convnext_ft_%j.err

set -eo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fer_augmentation
set -u

cd /home/dravida/Tesi_Davide_Augmentation || exit 1
mkdir -p err_log
export PYTHONPATH=/home/dravida/Tesi_Davide_Augmentation/scripts:/home/dravida/Tesi_Davide_Augmentation:${PYTHONPATH:-}

python scripts/finetuning.py --model_name ConvNeXt --FT_EPOCH 300 --dropout_rate 0.5 --learning_rate 0.0001 --l2_reg 0.00200041712955451