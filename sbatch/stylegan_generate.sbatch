#!/bin/bash
#SBATCH -p gpu_a40
#SBATCH -J stylegan_gen
#SBATCH -o /home/dravida/models/FER-Augmentation/out_err/stylegan_gen_%j.out
#SBATCH -e /home/dravida/models/FER-Augmentation/out_err/stylegan_gen_%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 2:00:00

set -eo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stylegan2

cd /home/dravida/Tesi_Davide_Augmentation/stylegan2-ada-pytorch

# Parse arguments or use environment variables
EMOTION="${1:-}"
START_SEED="${2:-0}"
END_SEED="${3:-499}"

if [ -z "$EMOTION" ]; then
  SEEDS="${SEEDS:-0-499}"
  OUTDIR="${OUTDIR:-/tmp/stylegan_gen}"
else
  EMOTION_LOWER=$(echo "$EMOTION" | tr '[:upper:]' '[:lower:]')
  SEEDS="$START_SEED-$END_SEED"
  OUTDIR="/home/dravida/models/stylegan2_generated_$EMOTION_LOWER"
fi

NETWORK="${NETWORK:-ffhq.pkl}"

mkdir -p "$OUTDIR"

echo "[INFO] Seeds: $SEEDS"
echo "[INFO] Output: $OUTDIR"
echo "[INFO] Network: $NETWORK"

python generate.py \
  --network="$NETWORK" \
  --seeds="$SEEDS" \
  --outdir="$OUTDIR"

echo "[SUCCESS] Generated in $OUTDIR"
ls -la "$OUTDIR" | head -20
