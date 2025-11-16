#!/bin/bash
#SBATCH --job-name=${USER}_GPU_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davideravida498@gmail.com
#SBATCH --partition=gpu_v100
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/models/FER-Augmentation/out_err/test_gpu_%j.log
#SBATCH --error=/home/%u/models/FER-Augmentation/out_err/test_gpu_%j.log


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
echo "Environment activated!"
python -c "import sys; print('Python:', sys.executable); print('Env:', sys.prefix)"
echo "=========================================================================="

echo "=========================================================================="
echo "Testing TensorFlow GPU detection:"
python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPUs detected:', tf.config.list_physical_devices('GPU')); print('CUDA available:', tf.test.is_built_with_cuda()); print('GPU available:', tf.test.is_gpu_available(build_info=False))"
echo "=========================================================================="

echo "=========================================================================="
echo "Testing PyTorch GPU detection:"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('Current device:', torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "=========================================================================="

echo "=========================================================================="
echo "GPU test completed successfully!"
echo "=========================================================================="
