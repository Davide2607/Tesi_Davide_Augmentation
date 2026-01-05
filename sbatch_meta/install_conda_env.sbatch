#!/bin/bash
#SBATCH --job-name=${USER}_FER_install
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davideravida498@gmail.com
#SBATCH --partition=cpu_skylake
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem=100M
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# Setup robust logging: create log directory and redirect stdout/stderr to per-job log in $HOME
LOGDIR="$HOME/models/FER-Augmentation/out_err"
mkdir -p "$LOGDIR"
JOB_ID="${SLURM_JOB_ID:-$$}"
exec > "$LOGDIR/train_${JOB_ID}.log" 2>&1

module load miniconda3/3.13.25

# confirm conda is found
which conda
conda --version

# accept Anaconda channels' ToS (replace CHANNEL with the shown URLs)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


ENV_NAME="fer_augmentation"
# Create environment only if it doesn't already exist (idempotent)
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "=========================================================================="
  echo "Environment $ENV_NAME already exists, skipping creation."
  echo "=========================================================================="
else
  conda create -n "$ENV_NAME" python=3.11 -y
  echo "=========================================================================="
  echo "Created venv!"
  echo "=========================================================================="
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "=========================================================================="
echo "Activated venv!"
python -c "import sys; print('Python executable:', sys.executable); print('Environment:', sys.prefix)"
echo "=========================================================================="


python -m pip install --upgrade pip setuptools wheel
echo "=========================================================================="
echo "Upgraded pip!"
echo "=========================================================================="

conda install -y nvidia/label/cuda-12.2.0::cuda-toolkit
echo "=========================================================================="
echo "Installed cudatoolkit 12.2"
echo "=========================================================================="

conda install -y cudnn=8.9 -c conda-forge
echo "=========================================================================="
echo "Installed cudnn 8.9"
echo "=========================================================================="

python -m pip install tensorflow-io-gcs-filesystem==0.31.0
echo "=========================================================================="
echo "Installed tf io-gs-fs!"
echo "=========================================================================="

python -m pip install tensorflow==2.15.0
echo "=========================================================================="
echo "Installed tf!"
echo "=========================================================================="

python -m pip install h5py "numpy>=1.23.5,<2.0.0" Pillow scikit-learn ultralytics gdown
echo "=========================================================================="
echo "Installed other dependencies! (h5, np, PIL, skl, ultralytics)"
echo "=========================================================================="


python -c "import sys; print('Python executable:', sys.executable); print('Environment:', sys.prefix)"
python -c "import tensorflow as tf; print('tf',tf.__version__); print(tf.sysconfig.get_build_info()); print('gpus:', tf.config.list_physical_devices('GPU'))"