#!/usr/bin/env python
"""Estrae una singola classe da dataset.h5 in PNG e opzionalmente crea uno zip.

Assume il formato h5 usato negli altri script:
- X_train, y_train, X_val, y_val
- class_names (array di byte string)

Esempio:
  python scripts/export_h5_class.py \
    --h5 ~/data/dataset.h5 \
    --class-name DISGUST \
    --out-dir ~/data/DISGUST \
    --zip
"""

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def _normalize_class_name(name: str) -> str:
    name = name.strip()
    if name.startswith('synthetic_'):
        name = name[len('synthetic_'):]
    normalized = {
        'anger': 'ANGRY',
        'disgust': 'DISGUST',
        'fear': 'FEAR',
        'ear': 'FEAR',
        'happiness': 'HAPPY',
        'neutrality': 'NEUTRAL',
        'sadness': 'SAD',
        'surprise': 'SURPRISE',
    }
    return normalized.get(name.lower(), name)


def parse_args():
    p = argparse.ArgumentParser(description="Estrai una classe da un dataset h5 in PNG")
    p.add_argument("--h5", type=Path, required=True, help="Percorso al file dataset.h5")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--class-name", type=str, help="Nome classe da estrarre (es. DISGUST)")
    group.add_argument("--class-idx", type=int, help="Indice classe da estrarre")
    p.add_argument("--out-dir", type=Path, required=True, help="Cartella di output per i PNG")
    p.add_argument("--zip", action="store_true", help="Crea anche lo zip della cartella estratta")
    return p.parse_args()


def load_dataset(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        X_train = np.array(f["X_train"])
        y_train = np.array(f["y_train"])
        X_val = np.array(f["X_val"])
        y_val = np.array(f["y_val"])
        class_names = [_normalize_class_name(c.decode("utf-8")) for c in f["class_names"]]
    X = np.concatenate([X_train, X_val])
    y = np.concatenate([y_train, y_val])
    return X, y, class_names


def main():
    args = parse_args()
    X, y, class_names = load_dataset(args.h5)

    if args.class_name:
        if args.class_name not in class_names:
            raise ValueError(f"Classe '{args.class_name}' non trovata. Disponibili: {class_names}")
        class_idx = class_names.index(args.class_name)
    else:
        class_idx = args.class_idx
        if class_idx < 0 or class_idx >= len(class_names):
            raise ValueError(f"Indice classe fuori range: {class_idx}, num classi={len(class_names)}")

    mask = y == class_idx
    imgs = X[mask]
    if len(imgs) == 0:
        raise ValueError(f"Nessuna immagine trovata per classe {class_idx} ({class_names[class_idx]})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Salvo {len(imgs)} immagini in {args.out_dir}")

    for i, arr in enumerate(imgs):
        img = Image.fromarray(arr.astype("uint8"))
        fname = args.out_dir / f"{i:06d}.png"
        img.save(fname)

    if args.zip:
        zip_path = shutil.make_archive(str(args.out_dir), "zip", root_dir=args.out_dir)
        print(f"Creato zip: {zip_path}")

    print(f"Completato per classe {class_idx} ({class_names[class_idx]})")


if __name__ == "__main__":
    main()
