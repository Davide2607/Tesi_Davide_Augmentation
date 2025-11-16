#!/bin/bash
#SBATCH --job-name=${USER}_FER_Augment
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davideravida498@gmail.com
#SBATCH --partition=cpu_skylake
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=100M
#SBATCH --ntasks-per-node=1
#SBATCH --output=$HOME/models/FER-Augmentation/out_err/train_%j.log
#SBATCH --error=$HOME/models/FER-Augmentation/out_err/train_%j.log



# =====================================================================================
# ============================= Preliminary Setups ====================================
# =====================================================================================


module load miniconda3/3.13.25
ENV_NAME="fer_augmentation"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=========================================================================="
echo "Activated venv!"
python -c "import sys; print('Python executable:', sys.executable); print('Environment:', sys.prefix)"
python -c "import tensorflow as tf; print('tf',tf.__version__); print(tf.sysconfig.get_build_info()); print('gpus:', tf.config.list_physical_devices('GPU'))"
echo "=========================================================================="

cd $HOME/Tesi_Davide || exit 1
pwd

