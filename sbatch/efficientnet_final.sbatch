#!/bin/bash
#SBATCH --job-name=famato/tesi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s310275@studenti.polito.it
#SBATCH --partition=cuda
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --mem=51200
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --output=err_log/train_%j.log
#SBATCH --error=err_log/train_%j.err
module load intel/python/3/2019.4.088
module load nvidia/cudasdk/11.6
source $HOME/myvenv/bin/activate
cd $HOME/final_scripts/scripts
python  train_final_new.py --model_name EfficientNetB1 --TRAIN_EPOCH 300  --dropout_rate 0.14413654697781123 --learning_rate 0.0006667288552864503 --l2_reg 0.0015379870915489774