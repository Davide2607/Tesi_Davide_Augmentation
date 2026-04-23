import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def _decode_class_names(raw):
    if raw is None:
        return None
    out = []
    for v in raw:
        if isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


def _normalize_class_name(name: str) -> str:
    name = name.strip()
    if name.startswith("synthetic_"):
        name = name[len("synthetic_"):]
    if name == "EAR":
        name = "FEAR"
    return name


def _load_labels_and_names(path: Path):
    with h5py.File(path, "r") as f:
        names = _decode_class_names(f["class_names"][:] if "class_names" in f else None)
        if names is not None:
            names = [_normalize_class_name(n) for n in names]

        y_train = np.array(f["y_train"], dtype=np.int64) if "y_train" in f else None
        y_val = np.array(f["y_val"], dtype=np.int64) if "y_val" in f else None

    return names, y_train, y_val


def _counts(y: np.ndarray, n: int) -> np.ndarray:
    if y is None:
        return np.zeros((n,), dtype=np.int64)
    if y.size == 0:
        return np.zeros((n,), dtype=np.int64)
    return np.bincount(y, minlength=n).astype(np.int64)


def _align_counts_to_union(names, y_train, y_val, union_names):
    if names is None:
        # Best effort: assume labels already match union order if lengths align; otherwise just use bincount.
        n = max(int(y_train.max()) if y_train is not None and y_train.size else -1,
                int(y_val.max()) if y_val is not None and y_val.size else -1) + 1
        n = max(n, len(union_names))
        return _counts(y_train, n)[: len(union_names)], _counts(y_val, n)[: len(union_names)]

    union_index_by_name = {n: i for i, n in enumerate(union_names)}
    src_to_union = np.array([union_index_by_name[n] for n in names], dtype=np.int64)

    # Remap labels into union index space (vectorized)
    def remap(y):
        if y is None:
            return None
        return src_to_union[y]

    y_train_u = remap(y_train)
    y_val_u = remap(y_val)

    return _counts(y_train_u, len(union_names)), _counts(y_val_u, len(union_names))


def compare(path_a: Path, path_b: Path):
    names_a, ytr_a, yva_a = _load_labels_and_names(path_a)
    names_b, ytr_b, yva_b = _load_labels_and_names(path_b)

    if names_a is None and names_b is None:
        raise ValueError("Neither dataset contains class_names; cannot align classes reliably.")

    # Union in a stable order: prefer the first file's order, then append missing from the second.
    union = []
    for src in (names_a or []):
        if src not in union:
            union.append(src)
    for src in (names_b or []):
        if src not in union:
            union.append(src)

    tr_a, va_a = _align_counts_to_union(names_a, ytr_a, yva_a, union)
    tr_b, va_b = _align_counts_to_union(names_b, ytr_b, yva_b, union)

    print(f"[A] {path_a}")
    print(f"[B] {path_b}")
    print(f"[classes] {union}")
    print()

    header = f"{'class':12s} | {'A_train':>7s} {'A_val':>7s} {'A_total':>7s} || {'B_train':>7s} {'B_val':>7s} {'B_total':>7s} || {'B-A total':>9s}"
    print(header)
    print("-" * len(header))

    for i, name in enumerate(union):
        a_tr = int(tr_a[i])
        a_va = int(va_a[i])
        a_tot = a_tr + a_va
        b_tr = int(tr_b[i])
        b_va = int(va_b[i])
        b_tot = b_tr + b_va
        diff = b_tot - a_tot
        print(f"{name:12s} | {a_tr:7d} {a_va:7d} {a_tot:7d} || {b_tr:7d} {b_va:7d} {b_tot:7d} || {diff:9d}")

    print("\n[totals]")
    print(f"A train={int(tr_a.sum())} val={int(va_a.sum())} total={int(tr_a.sum()+va_a.sum())}")
    print(f"B train={int(tr_b.sum())} val={int(va_b.sum())} total={int(tr_b.sum()+va_b.sum())}")


def parse_args():
    p = argparse.ArgumentParser(description="Compare class counts between two FER HDF5 datasets.")
    p.add_argument(
        "--a",
        type=Path,
        default=Path(os.path.expanduser("~/data/dataset.h5")),
        help="First H5 path (e.g., dataset.h5)",
    )
    p.add_argument(
        "--b",
        type=Path,
        default=Path(os.path.expanduser("~/data/dataset_augmented.h5")),
        help="Second H5 path (e.g., dataset_augmented.h5)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    compare(args.a, args.b)


if __name__ == "__main__":
    main()
