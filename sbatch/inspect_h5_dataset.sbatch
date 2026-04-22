#!/bin/bash
#SBATCH --job-name=inspect_h5
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s324381@studenti.polito.it
#SBATCH --output=/home/dravida/models/FER-Augmentation/out_err/inspect_h5_%j.out
#SBATCH --error=/home/dravida/models/FER-Augmentation/out_err/inspect_h5_%j.err
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G

set -eo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fer_augmentation
set -u

DATASET_PATH=${DATASET_PATH:-$HOME/data/dataset_unito.h5}
SAMPLE=${SAMPLE:-2000}

cd /home/dravida/Tesi_Davide_Augmentation || exit 1
mkdir -p /home/dravida/models/FER-Augmentation/out_err

export PYTHONPATH=/home/dravida/Tesi_Davide_Augmentation/scripts:/home/dravida/Tesi_Davide_Augmentation:${PYTHONPATH:-}

python -u scripts/inspect_h5_dataset.py --path "$DATASET_PATH" --sample "$SAMPLE"
