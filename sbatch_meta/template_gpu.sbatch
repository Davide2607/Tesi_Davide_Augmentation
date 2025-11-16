#!/bin/bash
#SBATCH --job-name=${USER}_FER_Augment
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davideravida498@gmail.com
#SBATCH --partition=gpu_v100
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=1G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=$HOME/models/FER-Augmentation/out_err/train_%j.log
#SBATCH --error=$HOME/models/FER-Augmentation/out_err/train_%j.log


# =====================================================================================
# ============================= Preliminary Setups ====================================
# =====================================================================================

module load miniconda3/3.13.25
ENV_NAME="fer_augmentation"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Export library paths for CUDA/cuDNN
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"


echo "=========================================================================="
echo "Activated venv!"
python -c "import sys; print('Python executable:', sys.executable); print('Environment:', sys.prefix)"
python -c "import tensorflow as tf; print('tf',tf.__version__); print(tf.sysconfig.get_build_info()); print('gpus:', tf.config.list_physical_devices('GPU'))"
echo "=========================================================================="

cd $HOME/Tesi_Davide || exit 1
pwd



