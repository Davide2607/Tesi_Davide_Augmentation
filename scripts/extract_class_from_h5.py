#!/usr/bin/env python
"""
Estrae immagini di una classe da H5 e crea zip per StyleGAN2.
Uso:
  python extract_class_from_h5.py --h5-file /path/to/database.h5 --class-name SAD --output-dir /tmp/SAD
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
import cv2
import subprocess
import sys
from typing import Optional

def normalize_class_name(name: str) -> str:
    """Normalizza il nome della classe a uppercase."""
    if isinstance(name, bytes):
        name = name.decode('utf-8')
    return name.upper().strip()

def extract_class(h5_file: str, class_name: str, output_dir: str, max_images: Optional[int] = None):
    """Estrae immagini di una classe da H5 file e le salva come PNG."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Apertura H5: {h5_file}")
    with h5py.File(h5_file, 'r') as db:
        # Leggi class_names e normalizza
        class_names_bytes = db['class_names'][:]
        class_names = [normalize_class_name(cn) for cn in class_names_bytes]
        print(f"[INFO] Classi disponibili: {class_names}")
        
        # Trova indice della classe
        class_target = normalize_class_name(class_name)
        if class_target not in class_names:
            raise ValueError(f"Classe {class_target} non trovata. Disponibili: {class_names}")
        class_idx = class_names.index(class_target)
        
        # Leggi train e val
        X_train = db['X_train'][:]
        y_train = db['y_train'][:]
        X_val = db['X_val'][:]
        y_val = db['y_val'][:]
        
        # Estrai immagini della classe
        train_mask = y_train == class_idx
        val_mask = y_val == class_idx
        
        X_class = np.concatenate([X_train[train_mask], X_val[val_mask]], axis=0)
        
        if max_images and len(X_class) > max_images:
            X_class = X_class[:max_images]
        
        print(f"[INFO] Estratte {len(X_class)} immagini della classe {class_target}")
        
        # Salva come PNG
        for i, img in enumerate(X_class):
            # Normalizza pixel scale se necessario (0-255)
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # Se in scala di grigi (H, W), convertire a RGB replicando canale
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            
            filename = output_path / f"img_{i:06d}.png"
            cv2.imwrite(str(filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            if (i + 1) % 100 == 0:
                print(f"[INFO] Salvate {i + 1}/{len(X_class)} immagini")
        
        print(f"[SUCCESS] Estratte {len(X_class)} immagini in {output_path}")

def create_zip(image_dir: str, zip_file: str):
    """Crea zip delle immagini per StyleGAN2."""
    print(f"[INFO] Creazione zip: {zip_file}")
    cmd = ["zip", "-r", "-q", zip_file, "."]
    subprocess.run(cmd, cwd=image_dir, check=True)
    print(f"[SUCCESS] Zip creato: {zip_file}")

def main():
    p = argparse.ArgumentParser(description="Estrai classe da H5 e crea zip per StyleGAN2")
    p.add_argument("--h5-file", required=True, help="Path del file H5")
    p.add_argument("--class-name", required=True, help="Nome della classe (es. SAD, FEAR)")
    p.add_argument("--output-dir", required=True, help="Directory di output per immagini PNG")
    p.add_argument("--max-images", type=int, default=None, help="Max immagini da estrarre")
    p.add_argument("--create-zip", type=str, default=None, help="Path zip finale (opzionale)")
    
    args = p.parse_args()
    
    extract_class(args.h5_file, args.class_name, args.output_dir, args.max_images)
    
    if args.create_zip:
        create_zip(args.output_dir, args.create_zip)

if __name__ == "__main__":
    main()
