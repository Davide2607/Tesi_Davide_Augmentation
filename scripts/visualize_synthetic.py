import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image

# Load synthetic data
data_dir = os.path.expanduser('~/models/STYLE_GAN_augmentation')
images_path = os.path.join(data_dir, 'synthetic_images.npy')
labels_path = os.path.join(data_dir, 'synthetic_labels.npy')

images = None
labels = None
if os.path.exists(images_path) and os.path.exists(labels_path):
    images = np.load(images_path)
    labels = np.load(labels_path)
else:
    print(f"[WARN] File .npy non trovati in {data_dir}. Salto la sezione dataset packed.")

if images is not None and labels is not None:
    # Try to load rare_idx mapping to original class ids (optional)
    rare_idx_path = os.path.join(data_dir, 'rare_class_indices.npy')
    rare_idx = np.load(rare_idx_path) if os.path.exists(rare_idx_path) else None

    print(f"Synthetic images shape: {images.shape} (dtype={images.dtype})")
    print(f"Synthetic labels shape: {labels.shape} (dtype={labels.dtype})")
    print(f"Value range: [{images.min()}, {images.max()}]")

    # Determine label indices
    if labels.ndim == 1:
        label_idx = labels.astype(int)
    else:
        label_idx = np.argmax(labels, axis=1)

    # Map to original class ids if rare_idx is present
    if rare_idx is not None:
        orig_label_idx = np.array([rare_idx[i] for i in label_idx])
    else:
        orig_label_idx = label_idx

    # Class names (fallback)
    class_names = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE']

    # Count per class
    print("\nSynthetic images per class (original idx):")
    unique, counts = np.unique(orig_label_idx, return_counts=True)
    for cid, cnt in zip(unique, counts):
        name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        print(f"{cid:2d} {name:10s}: {cnt}")

    # Save sample grid
    rows, cols = 4, 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle('Sample Synthetic Images', fontsize=12)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i]
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            ax.imshow(img)
            cid = orig_label_idx[i]
            name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
            ax.set_title(name, fontsize=7)
        ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(data_dir, 'synthetic_samples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSample grid saved to: {output_path}")

# Directory delle immagini generate da StyleGAN2
stylegan2_dir = os.path.expanduser('~/models/stylegan2_generated_surprise')
image_paths = sorted(glob.glob(os.path.join(stylegan2_dir, '*.png')))

if not image_paths:
    print(f"Nessuna immagine trovata in {stylegan2_dir}")
    exit(1)

print(f"Trovate {len(image_paths)} immagini in {stylegan2_dir}")

# Carica un sottoinsieme per anteprima (evita griglie enormi)
max_preview = 64
preview_paths = image_paths[:max_preview]
images = [np.array(Image.open(p)) for p in preview_paths]

# Parametri griglia
total = len(images)
cols = min(8, total)
rows = (total + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
fig.suptitle(f'Anteprima StyleGAN2 ({total}/{len(image_paths)} immagini)', fontsize=12)

for i, ax in enumerate(axes.flat):
    if i < total:
        img = images[i]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        ax.imshow(img)
        ax.set_title(f"img {i}", fontsize=7)
    ax.axis('off')

plt.tight_layout()
output_path = os.path.join(stylegan2_dir, 'stylegan2_all_samples.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nGriglia di anteprima salvata in: {output_path}")
print("Done!")
