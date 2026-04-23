#!/bin/bash
#SBATCH --job-name=h5_counts
#SBATCH --output=/home/dravida/models/FER-Augmentation/out_err/h5_counts_%j.out
#SBATCH --error=/home/dravida/models/FER-Augmentation/out_err/h5_counts_%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -eo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fer_augmentation
set -u

cd /home/dravida/Tesi_Davide_Augmentation || exit 1
export PYTHONPATH=/home/dravida/Tesi_Davide_Augmentation/scripts:/home/dravida/Tesi_Davide_Augmentation:${PYTHONPATH:-}

A_PATH=${A_PATH:-$HOME/data/dataset.h5}
B_PATH=${B_PATH:-$HOME/data/dataset_augmented.h5}

python -u scripts/compare_h5_class_counts.py --a "$A_PATH" --b "$B_PATH"
