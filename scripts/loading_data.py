# Funzione per caricare i dati
import os
import sys
import numpy as np
import h5py
from sklearn.utils import shuffle

# Allow running this module directly (or from notebooks) without having to
# manually set PYTHONPATH to include the scripts directory.
_SCRIPTS_DIR = os.path.dirname(__file__)
if _SCRIPTS_DIR and _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from data_generators import CustomBalancedDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2

def loading_data():
    def _normalize_class_name(name: str) -> str:
        name = name.strip()
        if name.startswith('synthetic_'):
            name = name[len('synthetic_'):]
        normalized = {
            'anger': 'ANGRY',
            'disgust': 'DISGUST',
            'fear': 'FEAR',
            'ear': 'FEAR',
            'happiness': 'HAPPY',
            'neutrality': 'NEUTRAL',
            'sadness': 'SAD',
            'surprise': 'SURPRISE',
        }
        return normalized.get(name.lower(), name)

    def load_data_and_labels(file_path, info):
        class_names = None
        with h5py.File(file_path, 'r') as f:
            if info == 'train':
                X_train = np.array(f['X_train'])
                y_train = np.array(f['y_train'])
                X_val = np.array(f['X_val'])
                y_val = np.array(f['y_val'])
                class_names = [_normalize_class_name(name.decode('utf-8')) for name in f['class_names']]
                return X_train, y_train, X_val, y_val, class_names
            else:
                if 'X_test' in f and 'y_test' in f:
                    x = np.array(f['X_test'])
                    y = np.array(f['y_test'])
                elif all(k in f for k in ('X_train', 'y_train', 'X_val', 'y_val')):
                    x = np.concatenate([np.array(f['X_train']), np.array(f['X_val'])], axis=0)
                    y = np.concatenate([np.array(f['y_train']), np.array(f['y_val'])], axis=0)
                    print(f"[WARN] {file_path} has no X_test/y_test. Using X_train+X_val as test set.")
                else:
                    raise KeyError(
                        f"{file_path} must contain X_test/y_test or X_train,y_train,X_val,y_val for test loading"
                    )
                if 'class_names' in f:
                    class_names = [_normalize_class_name(name.decode('utf-8')) for name in f['class_names']]
                return x, y, class_names

    file_path = os.path.expanduser('~/data') # path del dataset (uses HOME)
    train_path = os.environ.get('DATASET_H5', os.path.join(file_path, 'dataset.h5'))
    test_path = os.environ.get('TEST_DATASET_H5', os.path.join(file_path, 'test_data_adele.h5'))

    X_train, y_train, X_val, y_val, class_names = load_data_and_labels(train_path, 'train')
    X_test, y_test, test_class_names = load_data_and_labels(test_path, 'test')

    # If test label IDs follow a different class order than the training set,
    # remap y_test so that metrics are computed correctly.
    if test_class_names is not None and class_names is not None and test_class_names != class_names:
        train_index_by_name = {name: idx for idx, name in enumerate(class_names)}
        remap = {}
        missing = []
        for test_idx, name in enumerate(test_class_names):
            if name in train_index_by_name:
                remap[test_idx] = train_index_by_name[name]
            else:
                missing.append(name)

        if missing:
            raise ValueError(
                "class_names mismatch between train and test and some test classes are missing in train: "
                + ", ".join(missing)
            )

        print(
            "[WARN] class_names order differs between train and test. "
            "Remapping y_test to match training class order."
        )
        y_test = np.vectorize(remap.get)(y_test)

    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_probabilities = class_counts / total_samples
    initial_bias = np.log(class_probabilities / (1 - class_probabilities))

    # print("Bias iniziale per ciascuna classe:", initial_bias)

    X_train, y_train = shuffle(X_train, y_train)
    X_val, y_val = shuffle(X_val, y_val)
    return X_train, y_train, X_val, y_val, X_test, y_test, class_names, initial_bias

# Funzione per creare i generatori di dati
def carica_dati():
    X_train, y_train, X_val, y_val, X_test, y_test, _, initial_bias = loading_data()
    batch_size = int(os.environ.get('BATCH_SIZE', '16'))

    aug_rotation_range = float(os.environ.get('AUG_ROTATION_RANGE', '10'))
    aug_width_shift_range = float(os.environ.get('AUG_WIDTH_SHIFT_RANGE', '0.2'))
    aug_shear_range = float(os.environ.get('AUG_SHEAR_RANGE', '0.3'))
    aug_horizontal_flip = os.environ.get('AUG_HORIZONTAL_FLIP', '1') == '1'
    aug_fill_mode = os.environ.get('AUG_FILL_MODE', 'wrap')
    augmentations = {
        'rotation_range': aug_rotation_range,
        'width_shift_range': aug_width_shift_range,
        'shear_range': aug_shear_range,
        'horizontal_flip': aug_horizontal_flip,
        'fill_mode': aug_fill_mode,
    }
    valid_augmentations = {}
    test_augmentations = {}
    NUM_CLASSES = 7
# Conversione delle etichette in one-hot encoding
    y_train_one_hot = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_one_hot = to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test_one_hot = to_categorical(y_test, num_classes=NUM_CLASSES)
    train_generator_focal_smoot = CustomBalancedDataGenerator(X_train, y_train_one_hot, batch_size=batch_size, augmentations=augmentations, data_inf='train', label_smoothing=0.05)
    valid_generator_focal_smoot = CustomBalancedDataGenerator(X_val, y_val_one_hot, batch_size=batch_size, augmentations=valid_augmentations, data_inf='valid', label_smoothing=0)
    test_generator_focal_smoot = CustomBalancedDataGenerator(X_test, y_test_one_hot, batch_size=batch_size, augmentations=test_augmentations, data_inf='test', label_smoothing=0)
    
    return train_generator_focal_smoot, valid_generator_focal_smoot, test_generator_focal_smoot, initial_bias



