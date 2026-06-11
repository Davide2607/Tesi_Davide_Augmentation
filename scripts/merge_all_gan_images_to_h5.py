#!/usr/bin/env python
"""
Merge database_zero + immagini sintetiche PNG (7 classi) → database_zero_all_augmented.h5
Uso:
  python merge_all_gan_images_to_h5.py \
    --orig-h5 /path/database_zero.h5 \
    --gen-base-dir /path/stylegan2_generated \
    --classes ANGRY DISGUST FEAR HAPPY NEUTRAL SAD SURPRISE \
    --max-per-class 500 \
    --output-h5 /path/database_zero_all_augmented.h5
"""

import argparse
import collections
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, List
from PIL import Image

def load_original_h5(h5_path: str):
    """Leggi il dataset originale."""
    with h5py.File(h5_path, 'r') as f:
        X_train = np.array(f['X_train'])
        y_train = np.array(f['y_train'], dtype=np.int32)
        X_val = np.array(f['X_val'])
        y_val = np.array(f['y_val'], dtype=np.int32)
        class_names = [c.decode('utf-8') if isinstance(c, bytes) else c for c in f['class_names']]
    return X_train, y_train, X_val, y_val, class_names

def load_png_images(directory: str, target_size: int = 128, max_images: Optional[int] = None) -> np.ndarray:
    """Leggi tutte le immagini PNG da directory e ridimensiona."""
    png_files = sorted(Path(directory).glob('seed*.png'))
    
    if max_images:
        png_files = png_files[:max_images]
    
    if not png_files:
        print(f"[WARNING] Nessun file seed*.png trovato in {directory}")
        return np.array([], dtype=np.uint8).reshape(0, target_size, target_size, 3)
    
    images = []
    for i, png_file in enumerate(png_files):
        try:
            img = Image.open(png_file).convert('RGB')
            img = img.resize((target_size, target_size), Image.LANCZOS)
            img_array = np.array(img, dtype=np.uint8)
            images.append(img_array)
            
            if (i + 1) % 100 == 0:
                print(f"[INFO] Caricati {i + 1}/{len(png_files)} immagini da {Path(directory).name}")
        except Exception as e:
            print(f"[WARNING] Errore caricamento {png_file}: {e}")
            continue
    
    return np.array(images, dtype=np.uint8)

def normalize_class_name(name: str) -> str:
    """Normalizza il nome della classe."""
    if isinstance(name, bytes):
        name = name.decode('utf-8')
    return name.upper().strip()

def merge_and_save(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    class_names: list,
    synth_images_dict: dict,  # {class_idx: X_synth}
    output_h5: str
):
    """Unisci i dataset e salva in H5."""
    
    # Concat immagini sintetiche
    X_synth_list = []
    y_synth_list = []
    
    for class_idx in sorted(synth_images_dict.keys()):
        X_synth = synth_images_dict[class_idx]
        if len(X_synth) > 0:
            y_synth = np.full(len(X_synth), class_idx, dtype=np.int32)
            X_synth_list.append(X_synth)
            y_synth_list.append(y_synth)
            print(f"[INFO] Classe {class_idx} ({class_names[class_idx]}): {len(X_synth)} immagini sintetiche")
    
    if X_synth_list:
        X_synth_all = np.concatenate(X_synth_list, axis=0)
        y_synth_all = np.concatenate(y_synth_list, axis=0)
    else:
        print("[ERROR] Nessuna immagine sintetica caricata!")
        return
    
    # Concat con training
    X_train_merged = np.concatenate([X_train, X_synth_all], axis=0)
    y_train_merged = np.concatenate([y_train, y_synth_all], axis=0)
    
    # Salva in H5
    print(f"\n[INFO] Salvataggio in {output_h5}")
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('X_train', data=X_train_merged, compression='gzip', compression_opts=4)
        f.create_dataset('y_train', data=y_train_merged, compression='gzip', compression_opts=4)
        f.create_dataset('X_val', data=X_val, compression='gzip', compression_opts=4)
        f.create_dataset('y_val', data=y_val, compression='gzip', compression_opts=4)
        f.create_dataset('class_names', data=np.array([c.encode('utf-8') for c in class_names]))
    
    print(f"[SUCCESS] Dataset salvato!")
    print(f"  X_train shape: {X_train_merged.shape}")
    print(f"  y_train shape: {y_train_merged.shape}")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  y_val shape: {y_val.shape}")
    
    # Stampa distribuzione classi
    counter = collections.Counter(y_train_merged.tolist())
    print(f"\n[INFO] Class distribution (train):")
    for idx, name in enumerate(class_names):
        count = counter.get(idx, 0)
        delta = len(synth_images_dict.get(idx, np.array([])))
        print(f"  {name:12s}: {count:6d} (synthetic: +{delta})")

def main():
    p = argparse.ArgumentParser(description="Merge database_zero + 7 GAN images → database_zero_all_augmented")
    p.add_argument("--orig-h5", required=True, help="Path database_zero.h5")
    p.add_argument("--gen-base-dir", required=True, help="Base directory per stylegan2_generated_* (es. /home/.../models)")
    p.add_argument("--classes", nargs='+', default=['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE'], help="Classi da caricare")
    p.add_argument("--max-per-class", type=int, default=500, help="Max immagini per classe")
    p.add_argument("--output-h5", required=True, help="Output database_zero_all_augmented.h5")
    
    args = p.parse_args()
    
    # Carica dataset originale
    print(f"[INFO] Caricamento {args.orig_h5}...")
    X_train, y_train, X_val, y_val, class_names = load_original_h5(args.orig_h5)
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, classes: {class_names}")
    
    # Carica immagini sintetiche per ogni classe
    synth_images_dict = {}
    for class_name in args.classes:
        class_idx = [i for i, cn in enumerate(class_names) if normalize_class_name(cn) == normalize_class_name(class_name)]
        if not class_idx:
            print(f"[WARNING] Classe {class_name} non trovata, skip")
            continue
        class_idx = class_idx[0]
        
        # Directory naming: stylegan2_generated_lowercase
        class_dir = Path(args.gen_base_dir) / f"stylegan2_generated_{class_name.lower()}"
        if not class_dir.exists():
            print(f"[WARNING] Directory non trovata: {class_dir}")
            synth_images_dict[class_idx] = np.array([], dtype=np.uint8).reshape(0, 128, 128, 3)
            continue
        
        print(f"\n[INFO] Caricamento {class_name} da {class_dir}...")
        X_synth = load_png_images(str(class_dir), target_size=128, max_images=args.max_per_class)
        synth_images_dict[class_idx] = X_synth
    
    # Merge e salva
    merge_and_save(X_train, y_train, X_val, y_val, class_names, synth_images_dict, args.output_h5)

if __name__ == "__main__":
    main()
