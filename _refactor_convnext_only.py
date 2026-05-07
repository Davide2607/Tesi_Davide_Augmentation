import json
import re
from pathlib import Path

nb_path = Path(r"c:\Users\david\Desktop\Polito\Tesi_Davide\CONFIDENCE_MODEL.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))


def to_lines(s: str) -> list[str]:
    # Notebook source expects list-of-lines; keep trailing \n
    return [line + "\n" for line in s.splitlines()]


convnext_defs = """import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dropout, Dense, SeparableConv2D, BatchNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

class_names = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE']

class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

class SqueezeLayer(Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

# (Opzionale) builder: il notebook lavora principalmente caricando il pretrained ConvNeXt
def build_convnext_model(learning_rate, dropout_rate, l2_reg, initial_bias):
    num_classes = 7
    img_shape = (128, 128, 3)
    input_layer = tf.keras.Input(shape=img_shape, name='universal_input')

    backbone = tf.keras.applications.ConvNeXtBase(
        include_top=False,
        include_preprocessing=True,
        weights='imagenet',
        input_shape=img_shape,
    )
    base_model = Model(
        backbone.input,
        backbone.get_layer('convnext_base_stage_2_block_24_identity').output,
        name='base_model',
    )
    base_model.trainable = False

    self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')
    patch_extraction = tf.keras.Sequential([
        SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'),
        SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(l2_reg)),
    ], name='patch_extraction')

    global_average_layer = GlobalAveragePooling2D(name='gap')
    pre_classification = tf.keras.Sequential([
        Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
    ], name='pre_classification')
    prediction_layer = Dense(num_classes, activation='softmax', name='classification_head', bias_initializer=Constant(initial_bias))

    x = base_model(input_layer, training=False)
    x = patch_extraction(x)
    x = global_average_layer(x)
    x = Dropout(dropout_rate)(x)
    x = pre_classification(x)
    x = ExpandDimsLayer(axis=-1)(x)
    x = self_attention([x, x])
    x = SqueezeLayer(axis=-1)(x)
    outputs = prediction_layer(x)

    model = Model(inputs=input_layer, outputs=outputs, name='convnext-train-head')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=3.0),
        loss=categorical_focal_loss(alpha=0.25, gamma=2.0),
        metrics=['categorical_accuracy'],
    )
    return model
"""

convnext_load = """import numpy as np
import tensorflow as tf

def categorical_focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        focal_loss = weight * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)
    return loss

# Caricamento SOLO del modello ConvNeXt
convnext_path = '/content/drive/MyDrive/Colab Notebooks/HPC/finale/model/pretrained_ConvNeXt_finetuning'

custom_objects = {
    'loss': categorical_focal_loss(alpha=0.25, gamma=2.0),
    'categorical_focal_loss': categorical_focal_loss,
    'ExpandDimsLayer': ExpandDimsLayer,
    'SqueezeLayer': SqueezeLayer,
}

with tf.keras.utils.custom_object_scope(custom_objects):
    convnext = tf.keras.models.load_model(convnext_path)

convnext.summary(show_trainable=True)
"""

keras_models_only = """keras_models = {
    "ConvNeXt": convnext
}
"""

eval_loop_only = """from sklearn.metrics import confusion_matrix
from collections import defaultdict

keras_models = {
    "ConvNeXt": convnext
}

errors_by_model = defaultdict(set)
confidences_by_model = defaultdict(set)
predictions_by_image = defaultdict(list)
model_results = []
conf_matrix_sum = np.zeros((7, 7))

for model_name, model in keras_models.items():
    print(f"\n📌 Valutazione per {model_name}...\n")
    probabilities, y_true, y_pred = evaluate_keras_model(model, test_generator_focal_smoot, model_name)
    model_results.append((model_name, probabilities, y_true, y_pred))

    for idx, pred in enumerate(y_pred):
        predictions_by_image[idx].append(pred)

    confidences = np.max(probabilities, axis=1)
    high_conf_wrong = (y_pred != y_true)

    errors_by_model[model_name] = set(np.where(high_conf_wrong)[0])
    confidences_by_model[model_name] = confidences
    conf_matrix_sum += confusion_matrix(y_true, y_pred, labels=range(7))
"""

csv_export_only = """import os
import numpy as np
import pandas as pd

def _decode_path(p):
    if isinstance(p, (bytes, np.bytes_)):
        return p.decode('utf-8')
    return str(p)

image_names = [os.path.basename(_decode_path(p)) for p in test_generator_focal_smoot.paths_data]

data = []
for model_name, probabilities, y_true, y_pred in model_results:
    for idx, (true_label, pred_label, prob) in enumerate(zip(y_true, y_pred, probabilities)):
        data.append([model_name, image_names[idx], idx, true_label, pred_label, prob.tolist()])

df = pd.DataFrame(data, columns=["Model", "Image_Path", "Image_Index", "True_Label", "Pred_Label", "Probabilities"])
csv_path = "/content/drive/MyDrive/model_results.csv"
df.to_csv(csv_path, index=False)
print(f"Risultati salvati in: {csv_path}")
"""

changed = 0
cells_out = []

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        cells_out.append(cell)
        continue

    src = "".join(cell.get("source", []))

    # Replace model definition cell
    if "def build_model_final_layers" in src or "MobileNet, ResNet50V2" in src:
        cell["source"] = to_lines(convnext_defs)
        changed += 1
        cells_out.append(cell)
        continue

    # Replace multi-model loading cell
    if (
        "pretrained_EfficientNetB1_finetuning_weights" in src
        or "pretrained_ResNet_finetuning" in src
        or "pretrained_PattLite_finetuning" in src
        or "pretrained_VGG19_finetuning" in src
        or "pretrained_InceptionV3_finetuning" in src
    ):
        cell["source"] = to_lines(convnext_load)
        changed += 1
        cells_out.append(cell)
        continue

    # Neutralize YOLO/ultralytics cells
    if (
        "ultralytics" in src
        or re.search(r"\bYOLO\b", src)
        or "evaluate_yolo_model" in src
        or "yolo_models" in src
        or "!pip install ultralytics" in src
    ):
        cell["source"] = to_lines("# YOLO/Ultralytics rimosso (notebook ConvNeXt-only)")
        changed += 1
        cells_out.append(cell)
        continue

    # Replace keras_models dict cell
    if "keras_models = {" in src and any(x in src for x in ["ResNet", "EfficientNetB1", "PattLite", "VGG19", "InceptionV3"]):
        cell["source"] = to_lines(keras_models_only)
        changed += 1
        cells_out.append(cell)
        continue

    # Replace evaluation loop cell
    if "errors_by_model" in src and "for model_name, model in keras_models.items()" in src and any(x in src for x in ["ResNet", "EfficientNetB1", "PattLite", "VGG19", "InceptionV3"]):
        cell["source"] = to_lines(eval_loop_only)
        changed += 1
        cells_out.append(cell)
        continue

    # Replace CSV export cell
    if "model_results.csv" in src and "image_paths" in src and "Probabilities" in src:
        cell["source"] = to_lines(csv_export_only)
        changed += 1
        cells_out.append(cell)
        continue

    # Fix hardcoded index
    if "print(model_results[6])" in src:
        cell["source"] = to_lines("print(model_results[0])")
        changed += 1
        cells_out.append(cell)
        continue

    cells_out.append(cell)

nb["cells"] = cells_out
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

print("Updated notebook:", nb_path)
print("Cells changed:", changed)
