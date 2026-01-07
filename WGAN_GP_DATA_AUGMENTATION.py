#!/usr/bin/env python
# coding: utf-8

# # WGAN-GP Data Augmentation for Rare Emotion Classes
# 
# This script implements a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** 
# for generating synthetic samples of rare emotion classes to balance the dataset.

# ## Section 1: Configurazione Path per Cluster HPC

import os

# Path configurabili per cluster HPC
BASE_DIR = os.path.expanduser('~')  # /home/dravida
PROJECT_DIR = os.path.join(BASE_DIR, 'Tesi_Davide_Augmentation')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'models/WGAN_GP_augmentation')

# Crea cartelle se non esistono
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'Project dir: {PROJECT_DIR}')
print(f'Data dir: {DATA_DIR}')
print(f'Output dir: {OUTPUT_DIR}')

# ## Section 2: Import e check GPU

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend per cluster
import matplotlib.pyplot as plt
import h5py

print('TF', tf.__version__)
print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)

# ## Section 3: Caricamento dataset HDF5

DATASET_PATH = os.path.join(DATA_DIR, 'dataset.h5')
with h5py.File(DATASET_PATH, 'r') as f:
    X_train = np.array(f['X_train'])
    y_train = np.array(f['y_train'])
    X_val = np.array(f['X_val'])
    y_val = np.array(f['y_val'])
    class_names = [c.decode('utf-8') for c in f['class_names']]
X = np.concatenate([X_train, X_val])
y = np.concatenate([y_train, y_val])
print('Shapes', X.shape, y.shape)

# ## Section 4: Individuazione classi rare

counts = np.bincount(y)
max_count = counts.max()
rare_idx = np.where(counts < 0.15 * max_count)[0]
rare_names = [class_names[i] for i in rare_idx]
print('Rare classes:', dict(zip(rare_idx, rare_names)))
for i, n in enumerate(class_names):
    print(f'{n:10s}: {counts[i]}')

# ## Section 5: Prepara subset classi rare

mask = np.isin(y, rare_idx)
X_rare = X[mask].astype('float32') / 127.5 - 1.0
y_rare = y[mask]
label_map = {old:new for new, old in enumerate(rare_idx)}
y_remap = np.array([label_map[v] for v in y_rare])
num_classes = len(rare_idx)
y_onehot = tf.keras.utils.to_categorical(y_remap, num_classes)
print('Rare subset', X_rare.shape, y_onehot.shape)

# ## Section 6: Hyperparameter setup

IMG_SHAPE = (128,128,3)
NOISE_DIM = 100
BATCH_SIZE = 128
LR_C = 1e-4
LR_G = 1e-4
BETA1 = 0.0
LAMBDA_GP = 10
CRITIC_STEPS = 6

# ## Section 7: Definizione modelli WGAN-GP

def build_generator():
    noise = tf.keras.Input(shape=(NOISE_DIM,))
    label = tf.keras.Input(shape=(num_classes,))
    x = tf.keras.layers.Concatenate()([noise, label])
    x = tf.keras.layers.Dense(16*16*256, activation='relu')(x)
    x = tf.keras.layers.Reshape((16,16,256))(x)
    for f in [256,128,64]:
        x = tf.keras.layers.Conv2DTranspose(f, 4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Conv2D(3, 3, padding='same', activation='tanh')(x)
    return tf.keras.Model([noise, label], out)

def build_critic():
    img = tf.keras.Input(shape=IMG_SHAPE)
    label = tf.keras.Input(shape=(num_classes,))
    l = tf.keras.layers.Reshape((1,1,num_classes))(label)
    l = tf.keras.layers.UpSampling2D((IMG_SHAPE[0], IMG_SHAPE[1]))(l)
    x = tf.keras.layers.Concatenate()([img, l])
    for f in [64,128,256]:
        x = tf.keras.layers.Conv2D(f, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model([img, label], out)

G = build_generator()
C = build_critic()
print('Generator summary:')
G.summary()
print('\nCritic summary:')
C.summary()

# ## Section 8: Funzioni WGAN-GP (loss + gradient penalty)

crit_opt = tf.keras.optimizers.Adam(LR_C, beta_1=BETA1, beta_2=0.9, clipnorm=5.0)
gen_opt = tf.keras.optimizers.Adam(LR_G, beta_1=BETA1, beta_2=0.9)

@tf.function
def gradient_penalty(real_img, fake_img, labels):
    alpha = tf.random.uniform([tf.shape(real_img)[0], 1,1,1], 0.0, 1.0)
    inter = alpha * real_img + (1 - alpha) * fake_img
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(inter)
        pred = C([inter, labels], training=True)
    grads = gp_tape.gradient(pred, inter)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1,2,3]) + 1e-10)
    return tf.reduce_mean((norm - 1.0) ** 2)

# ## Section 9: Training loop WGAN-GP

CKPT_DIR = OUTPUT_DIR
os.makedirs(CKPT_DIR, exist_ok=True)
BEST_GEN_PATH = os.path.join(CKPT_DIR, 'best_generator_wgan_gp.h5')
BEST_CRIT_PATH = os.path.join(CKPT_DIR, 'best_critic_wgan_gp.h5')

@tf.function
def train_step(real_img, labels):
    bs = tf.shape(real_img)[0]
    # Update Critic n volte
    for _ in tf.range(CRITIC_STEPS):
        noise = tf.random.normal((bs, NOISE_DIM))
        with tf.GradientTape() as tape:
            fake_img = G([noise, labels], training=True)
            real_out = C([real_img, labels], training=True)
            fake_out = C([fake_img, labels], training=True)
            gp = gradient_penalty(real_img, fake_img, labels)
            c_loss = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out) + LAMBDA_GP * gp
        grads = tape.gradient(c_loss, C.trainable_variables)
        crit_opt.apply_gradients(zip(grads, C.trainable_variables))
    # Update Generator
    noise = tf.random.normal((bs, NOISE_DIM))
    with tf.GradientTape() as tape:
        fake_img = G([noise, labels], training=True)
        fake_out = C([fake_img, labels], training=True)
        g_loss = -tf.reduce_mean(fake_out)
    grads = tape.gradient(g_loss, G.trainable_variables)
    gen_opt.apply_gradients(zip(grads, G.trainable_variables))
    return c_loss, g_loss

def train(epochs=150):
    ds = tf.data.Dataset.from_tensor_slices((X_rare, y_onehot)).shuffle(len(X_rare)).batch(BATCH_SIZE)
    c_hist, g_hist = [], []
    best_g = np.inf
    for epoch in range(epochs):
        c_loss_epoch = g_loss_epoch = 0.0; steps = 0
        for real_img, labels in ds:
            c_loss, g_loss = train_step(real_img, labels)
            c_loss_epoch += c_loss; g_loss_epoch += g_loss; steps += 1
        c_hist.append(float(c_loss_epoch/steps)); g_hist.append(float(g_loss_epoch/steps))
        if g_hist[-1] < best_g:
            best_g = g_hist[-1]
            G.save(BEST_GEN_PATH)
            C.save(BEST_CRIT_PATH)
            print(f'Saved best at epoch {epoch+1} (Gen: {best_g:.3f})')
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Critic: {c_hist[-1]:.3f} - Gen: {g_hist[-1]:.3f}')
    return c_hist, g_hist

print('\nStarting training...')
c_losses, g_losses = train(epochs=100)

# Salva grafico loss
plt.figure(figsize=(10,5))
plt.plot(c_losses, label='Critic')
plt.plot(g_losses, label='Gen')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_losses.png'))
print(f'Loss plot saved to {OUTPUT_DIR}/training_losses.png')

# ## Section 10: Generazione sintetici

BEST_GEN_PATH = os.path.join(OUTPUT_DIR, 'best_generator_wgan_gp.h5')
if os.path.exists(BEST_GEN_PATH):
    G_best = tf.keras.models.load_model(BEST_GEN_PATH, compile=False)
    print('Loaded best generator')
else:
    G_best = G
    print('Using current generator')

target_per_class = int(max_count * 0.25)
synth_images = []
synth_labels = []
for cid in range(num_classes):
    noise = tf.random.normal((target_per_class, NOISE_DIM))
    labels = np.zeros((target_per_class, num_classes), dtype=np.float32)
    labels[:, cid] = 1
    imgs = G_best.predict([noise, labels], verbose=0)
    synth_images.append(imgs)
    synth_labels += [cid]*target_per_class
synth_images = np.concatenate(synth_images)
synth_images_uint8 = ((synth_images + 1)/2 * 255).astype(np.uint8)
synth_labels = np.array(synth_labels)
print('Synthetic set', synth_images_uint8.shape, synth_labels.shape)

# ## Section 11: Salvataggio

os.makedirs(OUTPUT_DIR, exist_ok=True)
G.save(os.path.join(OUTPUT_DIR, 'generator_wgan_gp.h5'))
C.save(os.path.join(OUTPUT_DIR, 'critic_wgan_gp.h5'))
np.save(os.path.join(OUTPUT_DIR, 'synthetic_images.npy'), synth_images_uint8)
np.save(os.path.join(OUTPUT_DIR, 'synthetic_labels.npy'), synth_labels)
np.save(os.path.join(OUTPUT_DIR, 'rare_class_indices.npy'), rare_idx)
print('Saved to', OUTPUT_DIR)

# ## Section 12: Classificatore e filtro immagini generate

import gc
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print('\n--- Training classifier for filtering ---')

# Libera memoria non piu' necessaria prima del training del classifier
for var_name in ['X', 'y', 'X_rare', 'y_rare', 'y_onehot']:
    if var_name in globals():
        del globals()[var_name]
gc.collect()

# Ricarica il dataset originale (train/val) per il classifier
with h5py.File(DATASET_PATH, 'r') as f:
    X_train_cls = np.array(f['X_train'])
    y_train_cls = np.array(f['y_train'])
    X_val_cls = np.array(f['X_val'])
    y_val_cls = np.array(f['y_val'])

print(f'Classifier data -> train: {X_train_cls.shape}, val: {X_val_cls.shape}, classes: {len(class_names)}')

CLASSIFIER_BATCH_SIZE = 32
BEST_CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, 'best_classifier.h5')

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
base_model.trainable = False

inputs = layers.Input(shape=IMG_SHAPE)
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
classifier = models.Model(inputs, outputs)

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow(X_train_cls, y_train_cls, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True)
val_gen = val_datagen.flow(X_val_cls, y_val_cls, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False)

# Calcola pesi di classe per gestire lo sbilanciamento
class_counts = np.bincount(y_train_cls, minlength=len(class_names))
total_samples = class_counts.sum()
class_weights = {i: float(total_samples / (len(class_names) * class_counts[i])) for i in range(len(class_names))}
print('Class weights:', class_weights)

callbacks = [
    ModelCheckpoint(BEST_CLASSIFIER_PATH, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
]

print('\n--- Phase 1: Training head (base frozen) ---')
classifier.fit(
    train_gen,
    steps_per_epoch=max(1, len(X_train_cls) // CLASSIFIER_BATCH_SIZE),
    validation_data=val_gen,
    validation_steps=max(1, len(X_val_cls) // CLASSIFIER_BATCH_SIZE),
    epochs=30,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

print('\n--- Phase 2: Fine-tuning ---')
base_model.trainable = True
for layer in base_model.layers[:-80]:
    layer.trainable = False

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(5e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

classifier.fit(
    train_gen,
    steps_per_epoch=max(1, len(X_train_cls) // CLASSIFIER_BATCH_SIZE),
    validation_data=val_gen,
    validation_steps=max(1, len(X_val_cls) // CLASSIFIER_BATCH_SIZE),
    epochs=40,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, 'filter_classifier.h5')
classifier.save(CLASSIFIER_PATH)
print(f'Classifier salvato: {CLASSIFIER_PATH}')

# Cleanup oggetti pesanti prima del filtro
del train_gen, val_gen, X_train_cls, X_val_cls, y_train_cls, y_val_cls
gc.collect()

print('\n--- Filtering generated images ---')

classifier = tf.keras.models.load_model(BEST_CLASSIFIER_PATH, compile=False)

synth_images_uint8 = np.load(os.path.join(OUTPUT_DIR, 'synthetic_images.npy'))
synth_labels = np.load(os.path.join(OUTPUT_DIR, 'synthetic_labels.npy'))
rare_idx = np.load(os.path.join(OUTPUT_DIR, 'rare_class_indices.npy'))

synth_images_norm = synth_images_uint8.astype('float32') / 255.0
predictions = classifier.predict(synth_images_norm, batch_size=64, verbose=1)
pred_classes = np.argmax(predictions, axis=1)
pred_confidences = np.max(predictions, axis=1)

# Etichette sintetiche sono mappate [0..num_classes-1]; converti all'id originale per il confronto
orig_labels = np.array([rare_idx[idx] for idx in synth_labels])

CONFIDENCE_THRESHOLD = 0.5
mask_confident = pred_confidences >= CONFIDENCE_THRESHOLD
mask_correct_class = pred_classes == orig_labels
mask_keep = mask_confident & mask_correct_class

filtered_images = synth_images_uint8[mask_keep]
filtered_labels = synth_labels[mask_keep]

print(f'Originali: {len(synth_images_uint8)} immagini')
print(f'Filtrate (confidence>={CONFIDENCE_THRESHOLD} e classe corretta): {len(filtered_images)} immagini')
print(f'Scartate: {len(synth_images_uint8) - len(filtered_images)} ({100*(1-len(filtered_images)/len(synth_images_uint8)):.1f}%)')

for cid in range(len(rare_idx)):
    count = np.sum(filtered_labels == cid)
    print(f'Classe {rare_idx[cid]} ({rare_names[cid]}): {count} immagini')

np.save(os.path.join(OUTPUT_DIR, 'synthetic_images_filtered.npy'), filtered_images)
np.save(os.path.join(OUTPUT_DIR, 'synthetic_labels_filtered.npy'), filtered_labels)
print('Salvate immagini filtrate')

# Anteprima filtrate salvata a disco
preview_path = os.path.join(OUTPUT_DIR, 'synthetic_images_filtered_preview.png')
fig, ax = plt.subplots(2, 5, figsize=(10, 4))
for i in range(min(10, len(filtered_images))):
    r, c = divmod(i, 5)
    ax[r, c].imshow(filtered_images[i])
    ax[r, c].set_title(f'Conf: {pred_confidences[mask_keep][i]:.2f}')
    ax[r, c].axis('off')
plt.suptitle('Immagini filtrate (alta confidence)')
plt.tight_layout()
plt.savefig(preview_path)
plt.close()
print(f'Preview filtrate salvato in {preview_path}')

print('\n=== WGAN-GP Training Complete ===')
