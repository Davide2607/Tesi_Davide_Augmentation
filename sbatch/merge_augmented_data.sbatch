#!/bin/bash
#SBATCH --job-name=merge_aug_data
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s324381@studenti.polito.it
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/dravida/models/FER-Augmentation/out_err/merge_aug_data_%j.out
#SBATCH --error=/home/dravida/models/FER-Augmentation/out_err/merge_aug_data_%j.err

set -eo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fer_augmentation
set -u

cd /home/dravida/Tesi_Davide_Augmentation || exit 1
mkdir -p /home/dravida/models/FER-Augmentation/out_err

python scripts/merge_augmented_data.py \
  --orig-path /home/dravida/data/dataset.h5 \
  --synthetic-dir /home/dravida/models/STYLE_GAN_augmentation_disgust \
  --synthetic-dir /home/dravida/models/STYLE_GAN_augmentation_surprise \
  --out-path /home/dravida/data/dataset_augmented.h5 \
  --seed 42
