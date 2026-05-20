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
        name = name[len("synthetic_") :]
    if name == "EAR":
        name = "FEAR"
    return name


def _estimate_max_value(x_ds, sample_n: int = 256) -> float:
    dt = x_ds.dtype
    if dt.kind in ("u", "i"):
        return float(np.iinfo(dt).max)

    n = int(x_ds.shape[0])
    if n == 0:
        return 0.0

    take = min(sample_n, n)
    sample = np.array(x_ds[:take], dtype="float32", copy=False)
    if sample.size == 0:
        return 0.0
    return float(np.nanmax(sample))


def _align_scale_and_dtype(images: np.ndarray, target_max: float, target_dtype) -> np.ndarray:
    images_f = images.astype("float32", copy=False)
    synth_max = float(np.nanmax(images_f)) if images_f.size else 0.0

    if target_max > 1.5 and synth_max <= 1.5:
        images_f = images_f * 255.0
    if target_max <= 1.5 and synth_max > 1.5:
        images_f = images_f / 255.0

    return images_f.astype(target_dtype, copy=False)


def _build_label_map(a_names, b_names):
    if a_names is None or b_names is None:
        return None

    a_index = {n: i for i, n in enumerate(a_names)}
    mapping = np.empty((len(b_names),), dtype=np.int64)
    missing = []
    for b_i, name in enumerate(b_names):
        if name not in a_index:
            missing.append(name)
            continue
        mapping[b_i] = a_index[name]

    if missing:
        raise ValueError(
            "class_names mismatch: these classes are in B but not in A: " + ", ".join(missing)
        )

    return mapping


def _read_by_indices(ds, indices: np.ndarray):
    """Read ds[indices] handling h5py requirement for increasing indices."""
    if indices.size == 0:
        return None

    order = np.argsort(indices)
    sorted_idx = indices[order]
    data = ds[sorted_idx]

    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return data[inv]


def _select_indices_per_class(
    y: np.ndarray,
    max_per_class: int,
    max_total: int,
    seed: int,
) -> np.ndarray:
    """Select indices from y with optional per-class and/or global caps.

    - max_per_class <= 0 means no per-class cap.
    - max_total <= 0 means no global cap.
    """
    y = np.asarray(y, dtype=np.int64)
    n = int(y.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64)

    rng = np.random.default_rng(seed)

    if max_per_class and max_per_class > 0:
        selected_parts = []
        for cls in np.unique(y):
            cls_idx = np.flatnonzero(y == cls).astype(np.int64, copy=False)
            if cls_idx.size > max_per_class:
                take = rng.choice(cls_idx, size=max_per_class, replace=False)
                selected_parts.append(take.astype(np.int64, copy=False))
            else:
                selected_parts.append(cls_idx)
        selected = np.concatenate(selected_parts, axis=0) if selected_parts else np.zeros((0,), dtype=np.int64)
    else:
        selected = np.arange(n, dtype=np.int64)

    if max_total and max_total > 0 and selected.size > max_total:
        selected = rng.choice(selected, size=max_total, replace=False).astype(np.int64, copy=False)

    # h5py advanced indexing prefers increasing order
    return np.sort(selected)


def merge_h5(
    a_path: Path,
    b_path: Path,
    out_path: Path,
    seed: int = 42,
    shuffle: bool = True,
    chunk_size: int = 256,
    include_b_val: bool = False,
    assume_same_labels_if_missing_class_names: bool = False,
    b_max_per_class: int = 0,
    b_max_total: int = 0,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(a_path, "r") as fa, h5py.File(b_path, "r") as fb, h5py.File(out_path, "w") as fo:
        for required in ("X_train", "y_train", "X_val", "y_val"):
            if required not in fa:
                raise KeyError(f"Missing key {required} in A: {a_path}. Keys={list(fa.keys())}")
        for required in ("X_train", "y_train"):
            if required not in fb:
                raise KeyError(f"Missing key {required} in B: {b_path}. Keys={list(fb.keys())}")

        a_names_raw = fa["class_names"][:] if "class_names" in fa else None
        b_names_raw = fb["class_names"][:] if "class_names" in fb else None
        a_names = _decode_class_names(a_names_raw)
        b_names = _decode_class_names(b_names_raw)
        if a_names is not None:
            a_names = [_normalize_class_name(n) for n in a_names]
        if b_names is not None:
            b_names = [_normalize_class_name(n) for n in b_names]

        if a_names is None:
            raise ValueError(f"A dataset has no class_names: {a_path} (required)")

        if b_names is None and not assume_same_labels_if_missing_class_names:
            raise ValueError(
                f"B dataset has no class_names: {b_path}. "
                "Pass --assume-same-labels-if-missing-class-names to assume label IDs already match A."
            )

        label_map = _build_label_map(a_names, b_names) if b_names is not None else None

        xa_tr = fa["X_train"]
        ya_tr = fa["y_train"]
        xa_va = fa["X_val"]
        ya_va = fa["y_val"]
        xb_tr = fb["X_train"]
        yb_tr = fb["y_train"]

        n_a = int(xa_tr.shape[0])
        n_b = int(xb_tr.shape[0])

        # Load B labels in memory once (cheap), apply label mapping and optionally downsample.
        yb_all = np.asarray(yb_tr, dtype=np.int64)
        if label_map is not None:
            yb_all = label_map[yb_all]

        b_selected = _select_indices_per_class(
            y=yb_all,
            max_per_class=int(b_max_per_class),
            max_total=int(b_max_total),
            seed=int(seed),
        )
        n_b_sel = int(b_selected.size)
        n_total = n_a + n_b_sel

        # Log B selection summary
        if b_max_per_class or b_max_total:
            print(f"[select B] original={n_b} selected={n_b_sel} b_max_per_class={b_max_per_class} b_max_total={b_max_total}")
            if a_names is not None and n_b_sel:
                binc = np.bincount(yb_all[b_selected], minlength=len(a_names))
                txt = ", ".join([f"{a_names[i]}={int(binc[i])}" for i in range(len(a_names)) if int(binc[i]) > 0])
                print(f"[select B] per-class selected: {txt}")

        x_shape = xa_tr.shape[1:]
        if tuple(xb_tr.shape[1:]) != tuple(x_shape):
            raise ValueError(
                f"Image shape mismatch: A {xa_tr.shape} vs B {xb_tr.shape}. "
                "Both must have the same HxWxC."
            )

        target_dtype = xa_tr.dtype
        target_max = _estimate_max_value(xa_tr)

        x_train_out = fo.create_dataset(
            "X_train",
            shape=(n_total,) + x_shape,
            dtype=target_dtype,
            compression="gzip",
            chunks=True,
        )
        y_train_out = fo.create_dataset(
            "y_train",
            shape=(n_total,),
            dtype=np.int32,
            compression="gzip",
            chunks=True,
        )

        # Validation: keep A validation by default.
        x_val_out = fo.create_dataset(
            "X_val",
            data=xa_va,
            compression="gzip",
            chunks=True,
        )
        y_val_out = fo.create_dataset(
            "y_val",
            data=np.array(ya_va, dtype=np.int32),
            compression="gzip",
            chunks=True,
        )

        if include_b_val:
            # Optional: append B val to A val. Requires B val keys.
            if "X_val" not in fb or "y_val" not in fb:
                raise KeyError("include_b_val=True but B is missing X_val/y_val")
            xb_va = fb["X_val"]
            yb_va = fb["y_val"]
            if tuple(xb_va.shape[1:]) != tuple(x_shape):
                raise ValueError(f"Val image shape mismatch: B val {xb_va.shape} vs A train {x_shape}")
            yb_va_arr = np.array(yb_va, dtype=np.int64)
            if label_map is not None:
                yb_va_arr = label_map[yb_va_arr]
            # Build new concatenated val datasets (copy-on-write into new datasets)
            x_val_concat = np.concatenate([np.array(xa_va), np.array(xb_va)], axis=0)
            y_val_concat = np.concatenate([np.array(ya_va, dtype=np.int32), yb_va_arr.astype(np.int32)], axis=0)
            del fo["X_val"]
            del fo["y_val"]
            x_val_out = fo.create_dataset("X_val", data=x_val_concat, compression="gzip", chunks=True)
            y_val_out = fo.create_dataset("y_val", data=y_val_concat, compression="gzip", chunks=True)

        fo.create_dataset("class_names", data=np.array(a_names, dtype="S"))

        if not shuffle:
            # Fast path: sequential copy (A then B) to avoid expensive random HDF5 indexing.
            for start in range(0, n_a, chunk_size):
                end = min(start + chunk_size, n_a)
                x_train_out[start:end] = xa_tr[start:end].astype(target_dtype, copy=False)
                y_train_out[start:end] = np.array(ya_tr[start:end], dtype=np.int32)
                if start == 0 or end == n_a or (start // chunk_size) % 50 == 0:
                    print(f"[write A] {end}/{n_a}")

            out_offset = n_a
            for start in range(0, n_b_sel, chunk_size):
                end = min(start + chunk_size, n_b_sel)
                idx_chunk = b_selected[start:end]
                x_b = _read_by_indices(xb_tr, idx_chunk)
                x_b = _align_scale_and_dtype(np.array(x_b), target_max, target_dtype)
                y_b = yb_all[idx_chunk]
                x_train_out[out_offset + start : out_offset + end] = x_b
                y_train_out[out_offset + start : out_offset + end] = y_b.astype(np.int32, copy=False)
                if start == 0 or end == n_b_sel or (start // chunk_size) % 50 == 0:
                    print(f"[write B] {end}/{n_b_sel}")

            print("\n[done]")
            print(f"A: {a_path}")
            print(f"B: {b_path}")
            print(f"OUT: {out_path}")
            print(f"train: {n_a} + {n_b_sel} = {n_total}")
            print(f"val (A only): {int(x_val_out.shape[0])}")
            print(f"class_names: {a_names}")
            return

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_total)

        def write_block(out_start: int, out_end: int):
            out_idx = perm[out_start:out_end]
            a_mask = out_idx < n_a
            b_mask = ~a_mask

            a_idx = out_idx[a_mask]
            b_idx = out_idx[b_mask] - n_a

            x_block = np.empty((out_end - out_start,) + x_shape, dtype=target_dtype)
            y_block = np.empty((out_end - out_start,), dtype=np.int32)

            if a_idx.size:
                x_a = _read_by_indices(xa_tr, a_idx)
                y_a = _read_by_indices(ya_tr, a_idx).astype(np.int32)
                x_block[a_mask] = x_a.astype(target_dtype, copy=False)
                y_block[a_mask] = y_a

            if b_idx.size:
                b_src = b_selected[b_idx]
                x_b = _read_by_indices(xb_tr, b_src)
                y_b = yb_all[b_src]
                x_b = _align_scale_and_dtype(np.array(x_b), target_max, target_dtype)
                x_block[b_mask] = x_b
                y_block[b_mask] = y_b.astype(np.int32, copy=False)

            x_train_out[out_start:out_end] = x_block
            y_train_out[out_start:out_end] = y_block

        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            write_block(start, end)
            if start == 0 or end == n_total or (start // chunk_size) % 20 == 0:
                print(f"[write] {end}/{n_total}")

        print("\n[done]")
        print(f"A: {a_path}")
        print(f"B: {b_path}")
        print(f"OUT: {out_path}")
        print(f"train: {n_a} + {n_b_sel} = {n_total}")
        print(f"val (A only): {int(x_val_out.shape[0])}")
        print(f"class_names: {a_names}")


def parse_args():
    p = argparse.ArgumentParser(description="Merge two FER HDF5 datasets (A + B) into a single output H5.")
    p.add_argument(
        "--a",
        type=Path,
        default=Path(os.path.expanduser("~/data/dataset.h5")),
        help="Base/original H5 (provides class_names and validation set)",
    )
    p.add_argument(
        "--b",
        type=Path,
        required=True,
        help="Second H5 to append to training set (e.g. database_zero with GAN images)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(os.path.expanduser("~/data/dataset_unito_con_GAN.h5")),
        help="Output merged H5 path",
    )
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    p.add_argument("--no-shuffle", action="store_true", help="Do not shuffle merged training set")
    p.add_argument("--chunk-size", type=int, default=256, help="Write chunk size (samples per block)")
    p.add_argument(
        "--include-b-val",
        action="store_true",
        help="Also append B validation to A validation (default: keep A val only)",
    )
    p.add_argument(
        "--assume-same-labels-if-missing-class-names",
        action="store_true",
        help="If B has no class_names, assume its label IDs already match A",
    )
    p.add_argument(
        "--b-max-per-class",
        type=int,
        default=0,
        help="Optional cap on how many B train samples to include per class (0 = no cap)",
    )
    p.add_argument(
        "--b-max-total",
        type=int,
        default=0,
        help="Optional cap on total B train samples to include (0 = no cap)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    merge_h5(
        a_path=args.a,
        b_path=args.b,
        out_path=args.out,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        chunk_size=args.chunk_size,
        include_b_val=args.include_b_val,
        assume_same_labels_if_missing_class_names=args.assume_same_labels_if_missing_class_names,
        b_max_per_class=args.b_max_per_class,
        b_max_total=args.b_max_total,
    )


if __name__ == "__main__":
    main()
