import argparse
import collections
import os
from pathlib import Path

import h5py
import numpy as np


def _load_original_dataset(orig_path: Path):
    with h5py.File(orig_path, "r") as f:
        x_train = np.array(f["X_train"])
        y_train = np.array(f["y_train"], dtype=np.int32)
        x_val = np.array(f["X_val"])
        y_val = np.array(f["y_val"], dtype=np.int32)
        class_names = [c.decode("utf-8") for c in f["class_names"]]
    return x_train, y_train, x_val, y_val, class_names


def _align_synthetic_scale(synth_images: np.ndarray, target_max: float, target_dtype) -> np.ndarray:
    """Align synthetic image scale to match the original dataset scale."""
    synth_images = synth_images.astype("float32", copy=False)
    synth_max = float(np.nanmax(synth_images)) if synth_images.size else 0.0

    # If original data is in [0,255] but synthetic is [0,1], rescale synthetic.
    if target_max > 1.5 and synth_max <= 1.5:
        synth_images = synth_images * 255.0

    # If original data is in [0,1] but synthetic is [0,255], downscale synthetic.
    if target_max <= 1.5 and synth_max > 1.5:
        synth_images = synth_images / 255.0

    return synth_images.astype(target_dtype, copy=False)


def _load_one_synthetic_dir(syn_dir: Path, target_max: float, target_dtype, max_per_dir=None, rng=None):
    images_path = syn_dir / "synthetic_images.npy"
    labels_path = syn_dir / "synthetic_labels.npy"
    rare_idx_path = syn_dir / "rare_class_indices.npy"

    if not images_path.exists() or not labels_path.exists() or not rare_idx_path.exists():
        raise FileNotFoundError(
            f"Missing files in {syn_dir}. Expected: synthetic_images.npy, synthetic_labels.npy, rare_class_indices.npy"
        )

    synth_images = np.load(images_path)
    synth_labels = np.load(labels_path)
    rare_idx = np.load(rare_idx_path)

    # Two supported conventions for synthetic_labels.npy:
    # 1) LOCAL labels: values in [0, len(rare_idx)-1] that must be mapped using rare_class_indices.npy
    # 2) GLOBAL labels: values already in dataset label space; must be subset of rare_idx
    synth_labels = np.asarray(synth_labels).astype(np.int64, copy=False)
    rare_idx = np.asarray(rare_idx).astype(np.int64, copy=False)
    rare_set = set(rare_idx.tolist())

    is_local = False
    if rare_idx.size > 0:
        local_set = set(range(int(rare_idx.size)))
        # Treat as local only if *all* labels are within the local index range.
        is_local = set(np.unique(synth_labels).tolist()).issubset(local_set)

    if is_local:
        label_map_rev = {local: int(global_id) for local, global_id in enumerate(rare_idx)}
        synth_labels_global = np.array([label_map_rev[int(v)] for v in synth_labels], dtype=np.int32)
    else:
        # If labels are already global, they must be within the provided rare_idx.
        uniq = set(np.unique(synth_labels).tolist())
        if uniq.issubset(rare_set):
            synth_labels_global = synth_labels.astype(np.int32, copy=False)
        else:
            raise ValueError(
                "Synthetic labels are neither local (0..k-1) nor valid global labels for the provided rare_class_indices. "
                f"uniq_labels={sorted(list(uniq))[:20]} (showing up to 20), rare_idx={rare_idx.tolist()}, dir={syn_dir}"
            )

    if max_per_dir is not None and max_per_dir > 0 and len(synth_labels_global) > max_per_dir:
        if rng is None:
            rng = np.random.default_rng(42)
        subset_idx = rng.choice(len(synth_labels_global), size=max_per_dir, replace=False)
        synth_images = synth_images[subset_idx]
        synth_labels_global = synth_labels_global[subset_idx]

    synth_images = _align_synthetic_scale(synth_images, target_max, target_dtype)

    print(
        f"[{syn_dir.name}] images={synth_images.shape} labels={synth_labels.shape} "
        f"mapped_labels={synth_labels_global.shape} rare_idx={rare_idx.tolist()} (labels_format={'local' if is_local else 'global'})"
    )

    return synth_images, synth_labels_global


def _print_class_distribution(title: str, labels: np.ndarray, class_names):
    counter = collections.Counter(labels.tolist())
    print(title)
    for idx, name in enumerate(class_names):
        print(f"  {name:12s}: {counter.get(idx, 0)}")


def merge_augmented_data(orig_path: Path, synthetic_dirs, out_path: Path, seed: int, max_per_dir=None):
    x_train, y_train, x_val, y_val, class_names = _load_original_dataset(orig_path)
    target_max = float(np.nanmax(x_train)) if x_train.size else 0.0
    target_dtype = x_train.dtype

    all_synth_images = []
    all_synth_labels = []

    dir_rng = np.random.default_rng(seed)
    for syn_dir in synthetic_dirs:
        images, labels = _load_one_synthetic_dir(
            Path(syn_dir),
            target_max,
            target_dtype,
            max_per_dir=max_per_dir,
            rng=dir_rng,
        )
        all_synth_images.append(images)
        all_synth_labels.append(labels)

    if not all_synth_images:
        raise ValueError("No synthetic directories provided.")

    synth_images = np.concatenate(all_synth_images, axis=0)
    synth_labels = np.concatenate(all_synth_labels, axis=0)

    x_train_new = np.concatenate([x_train, synth_images], axis=0)
    y_train_new = np.concatenate([y_train, synth_labels], axis=0)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_train_new))
    x_train_new = x_train_new[idx]
    y_train_new = y_train_new[idx]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("X_train", data=x_train_new, compression="gzip")
        f.create_dataset("y_train", data=y_train_new, compression="gzip")
        f.create_dataset("X_val", data=x_val, compression="gzip")
        f.create_dataset("y_val", data=y_val, compression="gzip")
        f.create_dataset("class_names", data=np.array(class_names, dtype="S"))

    print("\nMerge completed")
    print(f"Original train shape: {x_train.shape}, labels: {y_train.shape}")
    print(f"Synthetic shape     : {synth_images.shape}, labels: {synth_labels.shape}")
    print(f"Merged train shape  : {x_train_new.shape}, labels: {y_train_new.shape}")
    print(f"Validation kept     : {x_val.shape}, labels: {y_val.shape}")
    _print_class_distribution("\nClass distribution after merge:", y_train_new, class_names)
    print(f"\nSaved augmented dataset to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge original FER dataset with synthetic samples into a new HDF5 dataset."
    )
    parser.add_argument(
        "--orig-path",
        type=Path,
        default=Path(os.path.expanduser("~/data/dataset.h5")),
        help="Path to original dataset.h5",
    )
    parser.add_argument(
        "--synthetic-dir",
        action="append",
        required=True,
        help=(
            "Synthetic folder containing synthetic_images.npy, synthetic_labels.npy, rare_class_indices.npy. "
            "Pass this argument multiple times for multiple folders."
        ),
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path(os.path.expanduser("~/data/dataset_augmented.h5")),
        help="Output path for merged dataset",
    )
    parser.add_argument(
        "--max-per-dir",
        type=int,
        default=None,
        help="Max synthetic samples to use from each --synthetic-dir (e.g. 5000 means 5000 per dir)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    return parser.parse_args()


def main():
    args = parse_args()
    merge_augmented_data(args.orig_path, args.synthetic_dir, args.out_path, args.seed, args.max_per_dir)


if __name__ == "__main__":
    main()
