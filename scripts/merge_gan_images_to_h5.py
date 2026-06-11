#!/usr/bin/env python
"""
Merge database_zero + immagini sintetiche PNG (SAD + FEAR) → database_zero_augmented.h5
Uso:
  python merge_gan_images_to_h5.py \
    --orig-h5 /path/database_zero.h5 \
    --sad-dir /path/stylegan2_generated_sad \
    --fear-dir /path/stylegan2_generated_fear \
    --max-per-class 500 \
    --output-h5 /path/database_zero_augmented.h5
"""

import argparse
import collections
import h5py
import numpy as np
from pathlib import Path
from typing import Optional
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
    """Leggi tutte le immagini PNG da directory (seed0000.png, seed0001.png, ...) e ridimensiona."""
    png_files = sorted(Path(directory).glob('seed*.png'))
    
    if max_images:
        png_files = png_files[:max_images]
    
    images = []
    for i, png_file in enumerate(png_files):
        # Leggi con PIL e converti a RGB
        img = Image.open(png_file).convert('RGB')
        # Ridimensiona a target_size
        img = img.resize((target_size, target_size), Image.LANCZOS)
        img_array = np.array(img, dtype=np.uint8)
        images.append(img_array)
        
        if (i + 1) % 100 == 0:
            print(f"[INFO] Caricati {i + 1}/{len(png_files)} immagini da {Path(directory).name}")
    
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
    X_synth_sad: np.ndarray, X_synth_fear: np.ndarray,
    output_h5: str
):
    """Unisci i dataset e salva in H5."""
    
    # Trova gli indici delle classi SAD e FEAR
    class_names_norm = [normalize_class_name(c) for c in class_names]
    sad_idx = class_names_norm.index('SAD')
    fear_idx = class_names_norm.index('FEAR')
    
    # Crea label array per immagini sintetiche
    y_synth_sad = np.full(len(X_synth_sad), sad_idx, dtype=np.int32)
    y_synth_fear = np.full(len(X_synth_fear), fear_idx, dtype=np.int32)
    
    # Concat con training
    X_train_merged = np.concatenate([X_train, X_synth_sad, X_synth_fear], axis=0)
    y_train_merged = np.concatenate([y_train, y_synth_sad, y_synth_fear], axis=0)
    
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
        delta = 0
        if idx == sad_idx:
            delta = len(X_synth_sad)
        elif idx == fear_idx:
            delta = len(X_synth_fear)
        print(f"  {name:12s}: {count:6d} (synthetic: +{delta})")

def main():
    p = argparse.ArgumentParser(description="Merge database_zero + GAN images → database_zero_augmented")
    p.add_argument("--orig-h5", required=True, help="Path database_zero.h5")
    p.add_argument("--sad-dir", required=True, help="Directory con seed*.png (SAD)")
    p.add_argument("--fear-dir", required=True, help="Directory con seed*.png (FEAR)")
    p.add_argument("--max-per-class", type=int, default=500, help="Max immagini per classe")
    p.add_argument("--output-h5", required=True, help="Output database_zero_augmented.h5")
    
    args = p.parse_args()
    
    # Carica dataset originale
    print(f"[INFO] Caricamento {args.orig_h5}...")
    X_train, y_train, X_val, y_val, class_names = load_original_h5(args.orig_h5)
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, classes: {class_names}")
    
    # Carica immagini sintetiche
    print(f"\n[INFO] Caricamento SAD da {args.sad_dir}...")
    X_sad = load_png_images(args.sad_dir, max_images=args.max_per_class)
    print(f"  Caricate {len(X_sad)} immagini")
    
    print(f"\n[INFO] Caricamento FEAR da {args.fear_dir}...")
    X_fear = load_png_images(args.fear_dir, max_images=args.max_per_class)
    print(f"  Caricate {len(X_fear)} immagini")
    
    # Merge e salva
    merge_and_save(X_train, y_train, X_val, y_val, class_names, X_sad, X_fear, args.output_h5)

if __name__ == "__main__":
    main()
