#!/bin/bash
#SBATCH -p gpu_a40
#SBATCH -J stylegan2_ada_train_fear
#SBATCH -o /home/dravida/models/FER-Augmentation/out_err/stylegan2_train_fear_%j.out
#SBATCH -e /home/dravida/models/FER-Augmentation/out_err/stylegan2_train_fear_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 24:00:00

set -euo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stylegan2
python -m pip install -q ninja
PY=$(which python)

mkdir -p /home/dravida/models/FER-Augmentation/out_err
cd /home/dravida/Tesi_Davide_Augmentation/stylegan2-ada-pytorch || exit 1

# Parametri per FEAR
DATA_ZIP=/home/dravida/Tesi_Davide_Augmentation/data/FEAR_stylegan2.zip
OUTDIR=/home/dravida/Tesi_Davide_Augmentation/models/stylegan2_runs_fear
RESUME=/home/dravida/Tesi_Davide_Augmentation/stylegan2-ada-pytorch/ffhq.pkl
KIMG=50
GPUS=1
CFG=stylegan2
AUG=noaug

$PY train.py \
  --outdir="$OUTDIR" \
  --data="$DATA_ZIP" \
  --gpus=$GPUS \
  --cfg=$CFG \
  --resume="$RESUME" \
  --kimg=$KIMG \
  --aug=$AUG
