#!/bin/bash
#SBATCH --job-name=wgan_gp_training
#SBATCH --partition=gpu_a40
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=davideravida498@gmail.com

# Setup logging
LOGDIR="$HOME/models/FER-Augmentation/out_err"
mkdir -p "$LOGDIR"
JOB_ID="${SLURM_JOB_ID:-$$}"
exec > "$LOGDIR/wgan_gp_${JOB_ID}.log" 2>&1

echo "=========================================================================="
echo "Job started at $(date)"
echo "Job ID: $JOB_ID"
echo "=========================================================================="

# Attiva l'ambiente conda
source /home/dravida/.bashrc
eval "$(/home/dravida/.conda/bin/conda shell.bash hook)"
conda activate fer_augmentation

echo "=========================================================================="
echo "Python executable: $(which python)"
python --version
echo "=========================================================================="

# Esegui il training
cd /home/dravida/Tesi_Davide_Augmentation
/home/dravida/.conda/envs/fer_augmentation/bin/python WGAN_GP_DATA_AUGMENTATION.py

echo "=========================================================================="
echo "Job finished at $(date)"
echo "=========================================================================="
