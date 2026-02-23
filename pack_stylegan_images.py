#!/usr/bin/env python
from pathlib import Path
import argparse
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Pack all StyleGAN images into .npy files (no filtering)")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing generated PNG/JPG images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where synthetic_*.npy files are saved")
    parser.add_argument("--target-class", type=int, required=True, help="Global class id (e.g. 1=DISGUST)")
    parser.add_argument("--max-images", type=int, default=0, help="Max number of images to pack (0 = use all)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed used when sampling max-images")
    parser.add_argument("--file-pattern", type=str, default="seed*.png", help="Filename glob to select generated images")
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    image_files = sorted(list(input_dir.glob(args.file_pattern)))
    if not image_files:
        image_files = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")))
    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    total_available = len(image_files)
    if args.max_images > 0 and args.max_images < total_available:
        rng = np.random.default_rng(args.random_seed)
        selected_idx = np.sort(rng.choice(total_available, size=args.max_images, replace=False))
        image_files = [image_files[index] for index in selected_idx]

    images = []
    for image_path in image_files:
        image = Image.open(image_path).convert("RGB")
        images.append(np.array(image, dtype=np.uint8))

    synthetic_images = np.stack(images, axis=0)

    # Local labels for merge notebook mapping: only one rare class => all zeros.
    synthetic_labels = np.zeros((len(synthetic_images),), dtype=np.int32)
    rare_class_indices = np.array([args.target_class], dtype=np.int32)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "synthetic_images.npy", synthetic_images)
    np.save(output_dir / "synthetic_labels.npy", synthetic_labels)
    np.save(output_dir / "rare_class_indices.npy", rare_class_indices)

    with open(output_dir / "filter_stats.txt", "w", encoding="utf-8") as file_handle:
        file_handle.write("No-filter packing mode\n")
        file_handle.write("======================\n")
        file_handle.write(f"total: {len(synthetic_images)}\n")
        file_handle.write(f"accepted: {len(synthetic_images)}\n")
        file_handle.write("rejected: 0\n")
        file_handle.write("acceptance_rate: 100.0\n")

    print("=== Pack complete (no filter) ===")
    print(f"Total available images: {total_available}")
    print(f"Input images: {len(synthetic_images)}")
    print(f"synthetic_images.npy: {synthetic_images.shape} {synthetic_images.dtype}")
    print(f"synthetic_labels.npy: {synthetic_labels.shape} (all local label 0)")
    print(f"rare_class_indices.npy: {rare_class_indices.tolist()}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
