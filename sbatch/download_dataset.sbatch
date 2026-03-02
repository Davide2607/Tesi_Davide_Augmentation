#!/bin/bash
#SBATCH --job-name=download_dataset
#SBATCH --partition=cpu_skylake
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x_%j.log

echo "Scaricamento dataset da Google Drive..."

# Crea cartella dati
mkdir -p ~/data

# ID file Google Drive
GDRIVE_TRAIN_ID="1RqqDESsSLoCS3NsTYUfrMzEVM7cV8RAF"
GDRIVE_TEST_ID="1E9qvRJZNyZYL0S1K_G14LmX1n5lOxZWE"

TRAIN_PATH="$HOME/data/dataset.h5"
TEST_PATH="$HOME/data/test_data_adele.h5"

echo "Scarico dataset train/val..."
/home/dravida/.conda/envs/fer_augmentation/bin/gdown "https://drive.google.com/uc?id=$GDRIVE_TRAIN_ID" -O "$TRAIN_PATH"

echo "Scarico dataset test..."
/home/dravida/.conda/envs/fer_augmentation/bin/gdown "https://drive.google.com/uc?id=$GDRIVE_TEST_ID" -O "$TEST_PATH"

# Verifica i file
if [ -f "$TRAIN_PATH" ] && [ -f "$TEST_PATH" ]; then
    echo "✓ Dataset train e test scaricati con successo!"
    ls -lh "$TRAIN_PATH" "$TEST_PATH"
    
    # Mostra info HDF5 train
    /home/dravida/.conda/envs/fer_augmentation/bin/python -c "
import h5py
print('--- TRAIN/VAL dataset.h5 ---')
with h5py.File('$TRAIN_PATH', 'r') as f:
    print('Keys:', list(f.keys()))
    for k in f.keys():
        print(f'{k}: {f[k].shape}')
print('\n--- TEST test_data_adele.h5 ---')
with h5py.File('$TEST_PATH', 'r') as f:
    print('Keys:', list(f.keys()))
    for k in f.keys():
        print(f'{k}: {f[k].shape}')
"
else
    echo "✗ Errore: Dataset non scaricati correttamente!"
    exit 1
fi
