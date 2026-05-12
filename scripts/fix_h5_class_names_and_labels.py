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


def _infer_output_path(input_path: Path) -> Path:
    if input_path.suffix.lower() in {".h5", ".hdf5"}:
        return input_path.with_name(input_path.stem + "_fixed" + input_path.suffix)
    return input_path.with_name(input_path.name + "_fixed")


def _copy_dataset_in_chunks(src_ds, dst_ds, chunk_size: int) -> None:
    n = src_ds.shape[0]
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        dst_ds[start:end] = src_ds[start:end]


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Fix class_names and labels in an HDF5 FER dataset by normalizing names "
            "(e.g., EAR->FEAR) and collapsing duplicate classes."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input H5 path (expects X_train, y_train, X_val, y_val, class_names)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output H5 path (default: <input>_fixed.h5)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Copy chunk size along axis 0 for X_* datasets.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the inferred mapping without writing the output file.",
    )
    args = p.parse_args()

    in_path = args.input
    out_path = args.output or _infer_output_path(in_path)

    with h5py.File(in_path, "r") as f:
        for k in ("X_train", "y_train", "X_val", "y_val", "class_names"):
            if k not in f:
                raise SystemExit(f"[ERROR] Missing key '{k}' in input H5")

        raw_names = _decode_class_names(f["class_names"][:])
        if raw_names is None:
            raise SystemExit("[ERROR] class_names missing or empty")

        norm_names = [_normalize_class_name(n) for n in raw_names]

        # Build old_idx -> new_idx mapping by first occurrence in order.
        unique_names: list[str] = []
        name_to_new: dict[str, int] = {}
        old_to_new = np.zeros((len(norm_names),), dtype=np.int64)

        for old_i, n in enumerate(norm_names):
            if n not in name_to_new:
                name_to_new[n] = len(unique_names)
                unique_names.append(n)
            old_to_new[old_i] = name_to_new[n]

        duplicates = {
            n: [i for i, nn in enumerate(norm_names) if nn == n]
            for n in unique_names
            if sum(1 for nn in norm_names if nn == n) > 1
        }

        print(f"[IN]  {in_path}")
        print(f"[OUT] {out_path}")
        print(f"[class_names] raw n={len(raw_names)} {raw_names}")
        print(f"[class_names] normalized n={len(norm_names)} {norm_names}")
        print(f"[class_names] unique n={len(unique_names)} {unique_names}")
        if duplicates:
            print("[WARN] Duplicate classes after normalization:")
            for n, idxs in duplicates.items():
                print(f"  - {n}: old indices {idxs}")

        print("[mapping] old_idx -> new_idx")
        for old_i, n in enumerate(norm_names):
            print(f"  {old_i} ({raw_names[old_i]} -> {n}) -> {int(old_to_new[old_i])} ({unique_names[int(old_to_new[old_i])]})")

        y_train = np.array(f["y_train"], dtype=np.int64)
        y_val = np.array(f["y_val"], dtype=np.int64)

        if y_train.min() < 0 or y_train.max() >= len(raw_names):
            raise SystemExit(
                f"[ERROR] y_train has values outside [0, {len(raw_names)-1}] "
                f"(min={int(y_train.min())}, max={int(y_train.max())})"
            )
        if y_val.min() < 0 or y_val.max() >= len(raw_names):
            raise SystemExit(
                f"[ERROR] y_val has values outside [0, {len(raw_names)-1}] "
                f"(min={int(y_val.min())}, max={int(y_val.max())})"
            )

        y_train_new = old_to_new[y_train]
        y_val_new = old_to_new[y_val]

        print(
            "[labels] y_train old min/max="
            f"{int(y_train.min())}/{int(y_train.max())} -> new min/max={int(y_train_new.min())}/{int(y_train_new.max())}"
        )
        print(
            "[labels] y_val   old min/max="
            f"{int(y_val.min())}/{int(y_val.max())} -> new min/max={int(y_val_new.min())}/{int(y_val_new.max())}"
        )

        if args.dry_run:
            print("[DRY RUN] Not writing output")
            return

        # Write output
        if out_path.exists():
            raise SystemExit(f"[ERROR] Output file already exists: {out_path}")

        with h5py.File(out_path, "w") as out:
            # Copy X datasets (chunked)
            for x_key in ("X_train", "X_val"):
                src = f[x_key]
                kwargs = {}
                # Preserve chunking/compression settings where possible
                if src.chunks is not None:
                    kwargs["chunks"] = src.chunks
                if src.compression is not None:
                    kwargs["compression"] = src.compression
                    if src.compression_opts is not None:
                        kwargs["compression_opts"] = src.compression_opts
                if src.shuffle:
                    kwargs["shuffle"] = True
                if src.fletcher32:
                    kwargs["fletcher32"] = True

                dst = out.create_dataset(x_key, shape=src.shape, dtype=src.dtype, **kwargs)
                _copy_dataset_in_chunks(src, dst, args.chunk_size)

            out.create_dataset("y_train", data=y_train_new.astype(np.int64), dtype=np.int64)
            out.create_dataset("y_val", data=y_val_new.astype(np.int64), dtype=np.int64)

            # Store class_names as bytes to match existing loaders that decode utf-8
            dt = h5py.string_dtype(encoding="utf-8")
            out.create_dataset("class_names", data=np.array(unique_names, dtype=object), dtype=dt)

            out.attrs["source_path"] = str(in_path)
            out.attrs["normalized_class_names"] = True
            out.attrs["collapsed_duplicates"] = True

    print("[OK] Wrote fixed dataset")


if __name__ == "__main__":
    main()
