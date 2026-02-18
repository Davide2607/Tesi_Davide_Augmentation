#!/usr/bin/env python
# coding: utf-8

"""
Filter StyleGAN2-ADA generated images by FER confidence
Keeps only high-quality synthetic images with correct emotion prediction
"""

import os
import sys
import numpy as np
import h5py
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import EfficientNetB1, VGG19, MobileNet, ResNet50V2, InceptionV3, ConvNeXtBase
from tensorflow.keras.models import Model
from PIL import Image
import argparse

# Custom layers per compatibilità modello
class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

class SqueezeLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis
    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)


def build_model_final_layers(learning_rate, dropout_rate, l2_reg, initial_bias, model_name='EfficientNetB1'):
    """Build FER model - same architecture as training"""
    NUM_CLASSES = 7
    IMG_SHAPE = (128, 128, 3)
    input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')

    if model_name == 'EfficientNetB1':
        backbone = EfficientNetB1(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer('block5c_add').output, name='base_model')
    elif model_name == 'VGG19':
        backbone = VGG19(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.vgg19.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer('block4_pool').output, name='base_model')
    elif model_name == 'PattLite':
        backbone = MobileNet(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        base_model = Model(backbone.input, backbone.layers[-29].output, name='base_model')
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
    elif model_name == 'ResNet':
        backbone = ResNet50V2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer('conv4_block5_out').output, name='base_model')
    elif model_name == 'ConvNeXt':
        backbone = ConvNeXtBase(include_top=False, include_preprocessing=True, weights='imagenet',
                               input_shape=(128,128,3), classifier_activation='softmax')
        base_model = Model(backbone.input, backbone.get_layer('convnext_base_stage_2_block_24_identity').output, name='base_model')
    elif model_name == 'InceptionV3':
        backbone = InceptionV3(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer('mixed5').output, name='base_model')
    else:
        raise ValueError(f"Modello '{model_name}' non supportato.")

    base_model.trainable = False
    
    self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')
    patch_extraction = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'),
        tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(l2_reg))
    ], name='patch_extraction')
    
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
    dropout_layer = tf.keras.layers.Dropout(dropout_rate, name='top_dropout')
    prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, 
                                             kernel_regularizer=l2(l2_reg),
                                             bias_initializer=Constant(initial_bias),
                                             activation='softmax',
                                             name='predictions')
    
    x = input_layer
    x = base_model(x, training=False)
    x = patch_extraction(x)
    value = global_average_layer(x)
    value_exp = ExpandDimsLayer(axis=1)(value)
    query_key = tf.keras.layers.Reshape((-1, 256))(x)
    attention_output = self_attention([query_key, value_exp])
    attention_output = SqueezeLayer(axis=1)(attention_output)
    x = dropout_layer(attention_output)
    outputs = prediction_layer(x)
    
    model = Model(inputs=input_layer, outputs=outputs, name=f'{model_name}_FER')
    return model


def load_fer_model(model_path, model_name='EfficientNetB1'):
    """Load pre-trained FER model with custom objects"""
    print(f"Loading FER model: {model_path}")
    
    custom_objects = {
        'ExpandDimsLayer': ExpandDimsLayer,
        'SqueezeLayer': SqueezeLayer
    }
    
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Building model from scratch (requires initial_bias)")
        initial_bias = np.zeros(7)  # fallback
        model = build_model_final_layers(0.001, 0.1, 0.1, initial_bias, model_name)
        return model


def load_stylegan_images(input_dir):
    """Load StyleGAN2-ADA generated images from directory"""
    input_dir = Path(input_dir)
    
    # Cerca immagini PNG/JPG
    image_files = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))
    
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {input_dir}")
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    images = []
    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((128, 128))  # Resize to model input
        img_array = np.array(img, dtype=np.uint8)
        images.append(img_array)
    
    images = np.array(images)
    print(f"Loaded images shape: {images.shape}")
    return images, image_files


def filter_by_confidence(images, target_class, model, confidence_threshold=0.7):
    """
    Filter images by FER model confidence
    
    Args:
        images: np.array (N, 128, 128, 3) uint8 [0-255]
        target_class: int, expected emotion class
        model: keras model
        confidence_threshold: float, minimum confidence to keep
        
    Returns:
        filtered_images: np.array of accepted images
        accepted_indices: indices of accepted images
        stats: dict with filtering statistics
    """
    print(f"\n=== Filtering with confidence >= {confidence_threshold} ===")
    print(f"Target class: {target_class}")
    
    # Normalize to [0,1]
    images_norm = images.astype('float32') / 255.0
    
    # Predict
    predictions = model.predict(images_norm, batch_size=32, verbose=1)
    pred_classes = np.argmax(predictions, axis=1)
    pred_confidences = np.max(predictions, axis=1)
    
    # Filter: high confidence AND correct class
    mask_confident = pred_confidences >= confidence_threshold
    mask_correct = pred_classes == target_class
    mask_keep = mask_confident & mask_correct
    
    accepted_indices = np.where(mask_keep)[0]
    filtered_images = images[mask_keep]
    
    # Stats
    stats = {
        'total': len(images),
        'accepted': len(filtered_images),
        'rejected': len(images) - len(filtered_images),
        'acceptance_rate': len(filtered_images) / len(images) * 100,
        'mean_confidence_accepted': pred_confidences[mask_keep].mean() if len(filtered_images) > 0 else 0,
        'mean_confidence_rejected': pred_confidences[~mask_keep].mean() if np.sum(~mask_keep) > 0 else 0,
        'wrong_class_count': np.sum(~mask_correct),
        'low_confidence_count': np.sum(~mask_confident & mask_correct)
    }
    
    print(f"\n=== Filtering Results ===")
    print(f"Total images: {stats['total']}")
    print(f"Accepted: {stats['accepted']} ({stats['acceptance_rate']:.1f}%)")
    print(f"Rejected: {stats['rejected']}")
    print(f"  - Wrong class: {stats['wrong_class_count']}")
    print(f"  - Low confidence (but correct class): {stats['low_confidence_count']}")
    print(f"Mean confidence (accepted): {stats['mean_confidence_accepted']:.3f}")
    print(f"Mean confidence (rejected): {stats['mean_confidence_rejected']:.3f}")
    
    return filtered_images, accepted_indices, stats


def save_filtered_output(filtered_images, target_class, output_dir, stats):
    """Save filtered images as .npy files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create labels array (all same class)
    labels = np.full(len(filtered_images), target_class, dtype=np.int32)
    
    # Save
    np.save(output_dir / 'synthetic_images.npy', filtered_images)
    np.save(output_dir / 'synthetic_labels.npy', labels)
    np.save(output_dir / 'rare_class_indices.npy', np.array([target_class]))
    
    # Save stats as text
    with open(output_dir / 'filter_stats.txt', 'w') as f:
        f.write(f"Filtering Statistics\n")
        f.write(f"====================\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n=== Saved to {output_dir} ===")
    print(f"- synthetic_images.npy: {filtered_images.shape}")
    print(f"- synthetic_labels.npy: {labels.shape}")
    print(f"- rare_class_indices.npy: [target_class={target_class}]")
    print(f"- filter_stats.txt")


def main():
    parser = argparse.ArgumentParser(description='Filter StyleGAN2-ADA images by FER confidence')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory with generated PNG images (e.g., ~/models/stylegan2_generated_disgust)')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained FER model (.h5 or .keras)')
    parser.add_argument('--model-name', type=str, default='EfficientNetB1',
                       choices=['EfficientNetB1', 'VGG19', 'PattLite', 'ResNet', 'ConvNeXt', 'InceptionV3'],
                       help='Backbone model name')
    parser.add_argument('--target-class', type=int, required=True,
                       help='Expected emotion class (0=ANGRY, 1=DISGUST, 2=FEAR, etc.)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                       help='Minimum confidence threshold (default: 0.7)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for filtered images')
    
    args = parser.parse_args()
    
    print("=== StyleGAN2 Image Filter ===")
    print(f"Input dir: {args.input_dir}")
    print(f"Model: {args.model_path}")
    print(f"Target class: {args.target_class}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"Output dir: {args.output_dir}")
    
    # Load FER model
    model = load_fer_model(args.model_path, args.model_name)
    
    # Load generated images
    images, image_files = load_stylegan_images(args.input_dir)
    
    # Filter
    filtered_images, accepted_indices, stats = filter_by_confidence(
        images, args.target_class, model, args.confidence_threshold
    )
    
    # Save
    save_filtered_output(filtered_images, args.target_class, args.output_dir, stats)
    
    print("\n=== Complete ===")
    print(f"Acceptance rate: {stats['acceptance_rate']:.1f}%")
    print(f"Use these images in MERGE_AUGMENTED_DATA.ipynb")


if __name__ == "__main__":
    main()
