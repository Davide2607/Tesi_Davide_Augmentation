import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Load synthetic data
data_dir = os.path.expanduser('~/models/WGAN_GP_augmentation')
images = np.load(os.path.join(data_dir, 'synthetic_images.npy'))
labels = np.load(os.path.join(data_dir, 'synthetic_labels.npy'))

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
print("Done!")
