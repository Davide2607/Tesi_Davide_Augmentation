"""Compare per-class train/val counts between two FER HDF5 datasets.

Prints a per-class table with (A vs B) and the delta (B - A).

This script is intentionally self-contained so it can be executed directly via:
  python -u scripts/compare_h5_class_counts.py --a ... --b ...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def _decode_class_names(raw: Iterable[object] | None) -> list[str] | None:
    if raw is None:
        return None
    names: list[str] = []
    for v in raw:
        if isinstance(v, (bytes, np.bytes_)):
            names.append(v.decode("utf-8"))
        else:
            names.append(str(v))
    return names


def _normalize_class_name(name: str) -> str:
    name = name.strip()
    if name.startswith("synthetic_"):
        name = name[len("synthetic_") :]
    # Common typo observed in some merged/generated datasets
    if name == "EAR":
        name = "FEAR"
    return name


def _load_split_counts(path: Path) -> tuple[list[str] | None, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    with h5py.File(path, "r") as f:
        class_names_raw = f["class_names"][:] if "class_names" in f else None
        class_names = _decode_class_names(class_names_raw)
        if class_names is not None:
            class_names = [_normalize_class_name(n) for n in class_names]

        if "y_train" not in f or "y_val" not in f:
            raise KeyError(f"Missing y_train/y_val in H5 file: {path}")

        y_train = np.asarray(f["y_train"], dtype=np.int64)
        y_val = np.asarray(f["y_val"], dtype=np.int64)

    return class_names, y_train, y_val


def _index_to_union_map(class_names: list[str] | None, y_all: np.ndarray, union: list[str]) -> np.ndarray:
    if class_names is None:
        n = int(y_all.max()) + 1 if y_all.size else 0
        # class_0, class_1 ... fallback
        local = [f"class_{i}" for i in range(n)]
    else:
        local = class_names

    if not local:
        return np.zeros((0,), dtype=np.int64)

    union_index = {name: idx for idx, name in enumerate(union)}
    out = np.empty((len(local),), dtype=np.int64)
    for i, name in enumerate(local):
        if name not in union_index:
            raise RuntimeError(f"Internal error: union missing class '{name}'")
        out[i] = union_index[name]
    return out


def _counts(y: np.ndarray, index_to_union: np.ndarray, n_union: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    if y.size == 0:
        return np.zeros((n_union,), dtype=np.int64)

    if y.min() < 0:
        raise ValueError("Found negative labels in y")
    if y.max() >= index_to_union.size:
        raise ValueError(
            f"Found label id {int(y.max())} but class_names has only {int(index_to_union.size)} entries"
        )

    mapped = index_to_union[y]
    return np.bincount(mapped, minlength=n_union).astype(np.int64, copy=False)


def _build_union(a_names: list[str] | None, b_names: list[str] | None, a_y: np.ndarray, b_y: np.ndarray) -> list[str]:
    if a_names is None:
        a_n = int(a_y.max()) + 1 if a_y.size else 0
        a_list = [f"class_{i}" for i in range(a_n)]
    else:
        a_list = a_names

    if b_names is None:
        b_n = int(b_y.max()) + 1 if b_y.size else 0
        b_list = [f"class_{i}" for i in range(b_n)]
    else:
        b_list = b_names

    union: list[str] = []
    for n in a_list:
        if n not in union:
            union.append(n)
    for n in b_list:
        if n not in union:
            union.append(n)
    return union


def _print_table(union: list[str], a_tr: np.ndarray, a_va: np.ndarray, b_tr: np.ndarray, b_va: np.ndarray) -> None:
    print(
        f"{'class':12s} | {'A_train':>7s} {'A_val':>7s} {'A_total':>7s} || "
        f"{'B_train':>7s} {'B_val':>7s} {'B_total':>7s} || {'B-A total':>9s}"
    )
    print("-" * 78)
    for i, name in enumerate(union):
        a_total = int(a_tr[i] + a_va[i])
        b_total = int(b_tr[i] + b_va[i])
        print(
            f"{name:12s} | {int(a_tr[i]):7d} {int(a_va[i]):7d} {a_total:7d} || "
            f"{int(b_tr[i]):7d} {int(b_va[i]):7d} {b_total:7d} || {b_total - a_total:9d}"
        )
    print("\n[totals]")
    print("A", int(a_tr.sum()), int(a_va.sum()), int(a_tr.sum() + a_va.sum()))
    print("B", int(b_tr.sum()), int(b_va.sum()), int(b_tr.sum() + b_va.sum()))


def _maybe_check_test(path: Path, reference_names: list[str] | None, reference_label: str) -> None:
    if not path:
        return
    if not path.exists():
        print(f"[TEST WARN] file not found: {path}")
        return

    with h5py.File(path, "r") as f:
        if "class_names" not in f:
            print(f"[TEST] {path} has no class_names")
            return
        raw = f["class_names"][:]

    test_names = _decode_class_names(raw) or []
    test_names = [_normalize_class_name(n) for n in test_names]
    print(f"[TEST] class_names normalized={test_names}")

    if reference_names is None:
        return
    if test_names == reference_names:
        print(f"[TEST OK] order matches {reference_label}")
        return

    ref_set = set(reference_names)
    test_set = set(test_names)
    missing = sorted(ref_set - test_set)
    extra = sorted(test_set - ref_set)
    print(f"[TEST WARN] class_names order differs vs {reference_label}")
    if missing:
        print(f"[TEST WARN] missing in TEST: {missing}")
    if extra:
        print(f"[TEST WARN] extra in TEST: {extra}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare class counts between two FER HDF5 datasets")
    p.add_argument(
        "--a",
        type=Path,
        default=Path(os.path.expanduser("~/data/dataset_unito.h5")),
        help="Path to dataset A (baseline)",
    )
    p.add_argument(
        "--b",
        type=Path,
        default=Path(os.path.expanduser("~/data/dataset_unito_con_GAN.h5")),
        help="Path to dataset B (comparison)",
    )
    p.add_argument("--test", type=Path, default=None, help="Optional Adele test H5 to sanity-check class_names")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    a_names, a_ytr, a_yva = _load_split_counts(args.a)
    b_names, b_ytr, b_yva = _load_split_counts(args.b)

    a_y_all = np.concatenate([a_ytr, a_yva]) if a_ytr.size or a_yva.size else np.array([], dtype=np.int64)
    b_y_all = np.concatenate([b_ytr, b_yva]) if b_ytr.size or b_yva.size else np.array([], dtype=np.int64)

    union = _build_union(a_names, b_names, a_y_all, b_y_all)
    n_union = len(union)

    a_map = _index_to_union_map(a_names, a_y_all, union)
    b_map = _index_to_union_map(b_names, b_y_all, union)

    a_tr = _counts(a_ytr, a_map, n_union)
    a_va = _counts(a_yva, a_map, n_union)
    b_tr = _counts(b_ytr, b_map, n_union)
    b_va = _counts(b_yva, b_map, n_union)

    print(f"[A] {args.a}")
    if a_names is not None:
        print(f"[A] class_names n={len(a_names)} {a_names}")
    print(f"[B] {args.b}")
    if b_names is not None:
        print(f"[B] class_names n={len(b_names)} {b_names}")
    print(f"[UNION] n={len(union)} {union}")
    print()

    _print_table(union, a_tr, a_va, b_tr, b_va)

    if args.test is not None:
        _maybe_check_test(args.test, a_names, "A")


if __name__ == "__main__":
    main()
