#!/bin/bash
#SBATCH -p gpu_a40
#SBATCH -J export_h5
#SBATCH -o /home/dravida/models/FER-Augmentation/out_err/export_h5_%j.out
#SBATCH -e /home/dravida/models/FER-Augmentation/out_err/export_h5_%j.err
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 02:00:00

# Attiva conda (aggiorna il path se diverso)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fer_augmentation

# Ensure log directory exists before Slurm writes files
mkdir -p /home/dravida/models/FER-Augmentation/out_err

# Move into the project directory on the cluster
cd ~/Tesi_Davide_Augmentation || exit 1

# Download dataset da Google Drive se non esiste
GDRIVE_ID="1RqqDESsSLoCS3NsTYUfrMzEVM7cV8RAF"
mkdir -p ~/data
if [[ ! -f ~/data/dataset.h5 ]]; then
  echo "Scarico dataset da Google Drive..."
  gdown "https://drive.google.com/uc?id=$GDRIVE_ID" -O ~/data/dataset.h5
fi

# Parametri da modificare
H5=~/data/dataset.h5           # path del dataset h5
CLASS_NAME=DISGUST             # nome classe (oppure lascia vuoto e usa CLASS_IDX)
CLASS_IDX=                     # indice classe (se non usi CLASS_NAME)
OUT_DIR=~/data/DISGUST         # cartella di output PNG
MAKE_ZIP=1                     # 1 per creare zip, 0 per no

CMD=(python scripts/export_h5_class.py --h5 "$H5" --out-dir "$OUT_DIR")

if [[ -n "$CLASS_NAME" ]]; then
  CMD+=(--class-name "$CLASS_NAME")
elif [[ -n "$CLASS_IDX" ]]; then
  CMD+=(--class-idx "$CLASS_IDX")
else
  echo "Devi impostare CLASS_NAME o CLASS_IDX" >&2
  exit 1
fi

if [[ "$MAKE_ZIP" == "1" ]]; then
  CMD+=(--zip)
fi

set -x
"${CMD[@]}"
