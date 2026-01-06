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

# Check GPU availability
echo "=========================================================================="
echo "GPU Info:"
nvidia-smi || echo "nvidia-smi not available on login node"
echo "=========================================================================="

# Setup environment variables for CUDA from conda env
export CONDA_ENV_PATH="/home/dravida/.conda/envs/fer_augmentation"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_ENV_PATH"
export PATH="$CONDA_ENV_PATH/bin:$PATH"

echo "=========================================================================="
echo "Environment setup:"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Python: $CONDA_ENV_PATH/bin/python"
$CONDA_ENV_PATH/bin/python --version
echo "=========================================================================="

# Esegui il training
cd /home/dravida/Tesi_Davide_Augmentation
$CONDA_ENV_PATH/bin/python WGAN_GP_DATA_AUGMENTATION.py

echo "=========================================================================="
echo "Job finished at $(date)"
echo "=========================================================================="
