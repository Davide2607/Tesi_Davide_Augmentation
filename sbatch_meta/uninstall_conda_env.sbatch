#!/bin/bash
#SBATCH --job-name=${USER}_FER_uninstall
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davideravida498@gmail.com
#SBATCH --partition=cpu_skylake
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=100M
#SBATCH --ntasks-per-node=1
#SBATCH --output=$HOME/models/FER-Augmentation/out_err/train_%j.log
#SBATCH --error=$HOME/models/FER-Augmentation/out_err/train_%j.log

ENV_NAME="fer_augmentation"   # change if your env has a different name

# Initialize conda in non-interactive shells
eval "$(conda shell.bash hook)"
conda --version

echo "Removing conda env: $ENV_NAME"
if conda env list | grep -q "^$ENV_NAME\b"; then
	conda env remove -n "$ENV_NAME" -y
else
	echo "Conda env $ENV_NAME does not exist (skipping remove)"
fi

# If env still exists on disk (legacy), remove the env folder under $HOME
if [ -d "$HOME/.conda/envs/$ENV_NAME" ]; then
	echo "Removing leftover env folder: $HOME/.conda/envs/$ENV_NAME"
	rm -rf "$HOME/.conda/envs/$ENV_NAME"
fi
echo "=========================================================================="
echo "Activated venv!"
python -c "import sys; print('Python executable:', sys.executable); print('Environment:', sys.prefix)"
python -c "import tensorflow as tf; print('tf',tf.__version__); print(tf.sysconfig.get_build_info()); print('gpus:', tf.config.list_physical_devices('GPU'))"
python -c "import torch; print('PyTorch GPUs available:' + str(torch.cuda.device_count()) if torch.cuda.is_available() else 'No GPUs available')"
echo "=========================================================================="

cd $HOME/Tesi_Davide || exit 1
pwd

