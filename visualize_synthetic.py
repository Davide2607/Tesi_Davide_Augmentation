import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load synthetic data
data_dir = os.path.expanduser('~/models/WGAN_GP_augmentation')
synthetic_images = np.load(os.path.join(data_dir, 'synthetic_images.npy'))
synthetic_labels = np.load(os.path.join(data_dir, 'synthetic_labels.npy'))

print(f"Synthetic images shape: {synthetic_images.shape}")
print(f"Synthetic labels shape: {synthetic_labels.shape}")
print(f"Value range: [{synthetic_images.min():.3f}, {synthetic_images.max():.3f}]")

# Count per class
class_names = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE']
labels_idx = np.argmax(synthetic_labels, axis=1)
print("\nSynthetic images per class:")
for i, name in enumerate(class_names):
    count = np.sum(labels_idx == i)
    print(f"{name:10s}: {count}")

# Save sample grid
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Sample Synthetic Images (DISGUST & SURPRISE)', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < len(synthetic_images):
        img = synthetic_images[i]
        # Denormalize from [-1, 1] to [0, 255]
        img = ((img + 1) * 127.5).astype(np.uint8)
        ax.imshow(img)
        class_idx = labels_idx[i]
        ax.set_title(f"{class_names[class_idx]}", fontsize=8)
    ax.axis('off')

plt.tight_layout()
output_path = os.path.join(data_dir, 'synthetic_samples.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSample grid saved to: {output_path}")
print("Done!")
