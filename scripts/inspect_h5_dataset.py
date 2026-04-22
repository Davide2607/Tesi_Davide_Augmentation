import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def _decode_class_names(raw):
    if raw is None:
        return None
    names = []
    for v in raw:
        if isinstance(v, (bytes, np.bytes_)):
            names.append(v.decode("utf-8"))
        else:
            names.append(str(v))
    return names


def _normalize_class_name(name: str) -> str:
    name = name.strip()
    if name.startswith("synthetic_"):
        name = name[len("synthetic_"):]
    if name == "EAR":
        name = "FEAR"
    return name


def _sample_indices(n: int, sample: int, rng: np.random.Generator) -> np.ndarray:
    if sample is None or sample <= 0 or sample >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=sample, replace=False)


def _print_array_stats(name: str, arr: np.ndarray):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    if finite.any():
        vmin = float(arr[finite].min())
        vmax = float(arr[finite].max())
        mean = float(arr[finite].mean())
        std = float(arr[finite].std())
    else:
        vmin = vmax = mean = std = float("nan")

    print(
        f"[{name}] shape={arr.shape} dtype={arr.dtype} "
        f"min={vmin:.4f} max={vmax:.4f} mean={mean:.4f} std={std:.4f} "
        f"finite={finite.mean()*100:.2f}%"
    )


def _print_label_stats(split: str, y: np.ndarray, class_names):
    y = np.asarray(y).astype(np.int64, copy=False)
    print(f"[{split}] y shape={y.shape} dtype={y.dtype} min={y.min()} max={y.max()}")

    if class_names is not None:
        n_classes = len(class_names)
    else:
        n_classes = int(y.max()) + 1 if y.size else 0

    binc = np.bincount(y, minlength=n_classes) if y.size else np.zeros((n_classes,), dtype=np.int64)
    print(f"[{split}] class distribution:")
    if class_names is None:
        for i, c in enumerate(binc.tolist()):
            print(f"  class_{i}: {c}")
    else:
        for i, (name, c) in enumerate(zip(class_names, binc.tolist())):
            print(f"  {i:2d} {name:12s}: {c}")


def inspect_h5(path: Path, sample: int, seed: int):
    if not path.exists():
        raise FileNotFoundError(str(path))

    rng = np.random.default_rng(seed)

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print(f"[FILE] {path} ({path.stat().st_size/1024/1024:.1f} MB)")
        print("[KEYS]", keys)

        class_names_raw = f["class_names"][:] if "class_names" in f else None
        class_names = _decode_class_names(class_names_raw)
        if class_names is not None:
            normalized = [_normalize_class_name(n) for n in class_names]
            print(f"[class_names] n={len(class_names)} raw={class_names}")
            if normalized != class_names:
                print(f"[class_names] normalized={normalized}")

        # Train/val
        for split in ("train", "val"):
            x_key = f"X_{split}"
            y_key = f"y_{split}"
            if x_key not in f or y_key not in f:
                print(f"[WARN] missing {x_key} or {y_key}")
                continue

            x_ds = f[x_key]
            y = np.array(f[y_key], dtype=np.int64)

            idx = _sample_indices(x_ds.shape[0], sample, rng)
            x_sample = np.array(x_ds[idx], dtype=np.float32)
            _print_array_stats(x_key + "_sample", x_sample)
            _print_label_stats(split, y, class_names)

            # quick sanity
            if class_names is not None:
                bad = (y < 0) | (y >= len(class_names))
                if bad.any():
                    print(f"[ERROR] {split}: found {bad.sum()} labels outside [0, {len(class_names)-1}]")

        # Test (optional)
        if "X_test" in f and "y_test" in f:
            x_ds = f["X_test"]
            y = np.array(f["y_test"], dtype=np.int64)
            idx = _sample_indices(x_ds.shape[0], sample, rng)
            x_sample = np.array(x_ds[idx], dtype=np.float32)
            _print_array_stats("X_test_sample", x_sample)
            _print_label_stats("test", y, class_names)

        # Pixel scale heuristic
        if "X_train" in f:
            x_ds = f["X_train"]
            idx = _sample_indices(x_ds.shape[0], min(sample or 2000, 2000), rng)
            x_sample = np.array(x_ds[idx], dtype=np.float32)
            vmax = float(np.nanmax(x_sample)) if x_sample.size else 0.0
            if vmax <= 1.5:
                print("[WARN] Pixel max <= 1.5 (looks like 0..1). Ensure preprocessing is consistent.")
            elif vmax <= 300:
                print("[OK] Pixel scale looks like 0..255")
            else:
                print(f"[WARN] Pixel max unusually high: {vmax}")


def parse_args():
    p = argparse.ArgumentParser(description="Inspect an FER HDF5 dataset for training compatibility.")
    p.add_argument(
        "--path",
        type=Path,
        default=Path(os.path.expanduser("~/data/dataset_unito.h5")),
        help="Path to HDF5 dataset",
    )
    p.add_argument("--sample", type=int, default=2000, help="Sample size for X_* stats (0=all)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    return p.parse_args()


def main():
    args = parse_args()
    inspect_h5(args.path, args.sample, args.seed)


if __name__ == "__main__":
    main()
