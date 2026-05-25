import argparse
import os
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError as e:
    raise SystemExit(
        "[ERROR] Missing dependency 'h5py'.\n\n"
        "If you're on the HPC, you likely need to use the conda env used for training:\n"
        "  module load miniconda3/3.13.25\n"
        "  eval \"$(conda shell.bash hook)\"\n"
        "  conda activate fer_augmentation\n\n"
        "If the env exists but is missing the package:\n"
        "  python -m pip install h5py\n"
    ) from e


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
    idx = rng.choice(n, size=sample, replace=False)
    # h5py advanced indexing requires increasing order
    return np.sort(idx.astype(np.int64, copy=False))


def _stream_h5_stats(x_ds, indices: np.ndarray, chunk_size: int = 64):
    """Compute min/max/mean/std over selected indices without loading everything in RAM."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return {
            "shape": (0,) + tuple(x_ds.shape[1:]),
            "dtype": x_ds.dtype,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "finite_ratio": float("nan"),
        }

    # Welford's online algorithm for mean/std
    count = 0
    mean = 0.0
    m2 = 0.0
    vmin = float("inf")
    vmax = float("-inf")
    finite_count = 0
    total_count = 0

    # h5py requires indices to be strictly increasing for advanced indexing.
    if indices.size > 1 and np.any(indices[1:] < indices[:-1]):
        indices = np.sort(indices)

    for start in range(0, indices.size, chunk_size):
        chunk_idx = indices[start : start + chunk_size]
        x = np.asarray(x_ds[chunk_idx], dtype=np.float32)
        flat = x.reshape(-1)
        finite = np.isfinite(flat)
        total_count += flat.size
        if finite.any():
            finite_vals = flat[finite]
            finite_count += finite_vals.size

            cmin = float(finite_vals.min())
            cmax = float(finite_vals.max())
            vmin = min(vmin, cmin)
            vmax = max(vmax, cmax)

            # Update mean/std with chunk
            chunk_n = finite_vals.size
            chunk_mean = float(finite_vals.mean())
            chunk_var = float(finite_vals.var())  # population var
            if count == 0:
                mean = chunk_mean
                m2 = chunk_var * chunk_n
                count = chunk_n
            else:
                delta = chunk_mean - mean
                new_count = count + chunk_n
                mean = mean + delta * (chunk_n / new_count)
                m2 = m2 + chunk_var * chunk_n + (delta * delta) * (count * chunk_n / new_count)
                count = new_count

    if finite_count == 0:
        vmin = vmax = mean = float("nan")
        std = float("nan")
        finite_ratio = 0.0
    else:
        std = float(np.sqrt(m2 / count)) if count > 0 else float("nan")
        finite_ratio = finite_count / total_count if total_count else float("nan")

    return {
        "shape": (int(indices.size),) + tuple(x_ds.shape[1:]),
        "dtype": x_ds.dtype,
        "min": vmin,
        "max": vmax,
        "mean": float(mean),
        "std": std,
        "finite_ratio": finite_ratio,
    }


def _print_h5_stats(name: str, x_ds, indices: np.ndarray):
    s = _stream_h5_stats(x_ds, indices)
    finite_pct = s["finite_ratio"] * 100.0 if np.isfinite(s["finite_ratio"]) else float("nan")
    print(
        f"[{name}] shape={s['shape']} dtype={s['dtype']} "
        f"min={s['min']:.4f} max={s['max']:.4f} mean={s['mean']:.4f} std={s['std']:.4f} "
        f"finite={finite_pct:.2f}%"
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
            _print_h5_stats(x_key + "_sample", x_ds, idx)
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
            _print_h5_stats("X_test_sample", x_ds, idx)
            _print_label_stats("test", y, class_names)

        # Pixel scale heuristic
        if "X_train" in f:
            x_ds = f["X_train"]
            idx = _sample_indices(x_ds.shape[0], min(sample or 2000, 2000), rng)
            s = _stream_h5_stats(x_ds, idx)
            vmax = float(s["max"]) if np.isfinite(s["max"]) else 0.0
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
