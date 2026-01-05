#!/bin/bash
#SBATCH --job-name=download_dataset
#SBATCH --partition=cpu_skylake
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x_%j.log

echo "Scaricamento dataset da Google Drive..."

# Attiva ambiente conda
eval "$(/home/dravida/.conda/bin/conda shell.bash hook)"
conda activate fer_augmentation

# Crea cartella dati
mkdir -p ~/data

# Scarica il dataset da Google Drive
GDRIVE_FILE_ID="1LLW2MGp-JDIqYdt0QPAz65DpGhlAbcGC"
OUTPUT_PATH="$HOME/data/dataset.h5"

/home/dravida/.conda/envs/fer_augmentation/bin/gdown "$GDRIVE_FILE_ID" -O "$OUTPUT_PATH"

# Verifica il file
if [ -f "$OUTPUT_PATH" ]; then
    echo "✓ Dataset scaricato con successo!"
    ls -lh "$OUTPUT_PATH"
    
    # Mostra info HDF5
    /home/dravida/.conda/envs/fer_augmentation/bin/python -c "
import h5py
with h5py.File('$OUTPUT_PATH', 'r') as f:
    print('Keys:', list(f.keys()))
    for k in f.keys():
        print(f'{k}: {f[k].shape}')
"
else
    echo "✗ Errore: Dataset non scaricato!"
    exit 1
fi
