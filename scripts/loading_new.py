# Funzione per caricare i dati
import os
import numpy as np
import h5py
from sklearn.utils import shuffle
from data_generators import CustomBalancedDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2

def loading_data():
    def load_data_and_labels(file_path, info):
        class_names = None
        with h5py.File(file_path, 'r') as f:
            if info == 'train':
                X_train = np.array(f['X_train'])
                y_train = np.array(f['y_train'])
                X_val = np.array(f['X_val'])
                y_val = np.array(f['y_val'])
                class_names = [name.decode('utf-8') for name in f['class_names']]
                return X_train, y_train, X_val, y_val, class_names
            else:
                x = np.array(f['X_test'])
                y = np.array(f['y_test'])
                return x, y, class_names

    file_path = '/home/famato/final_scripts/dataset' #path del dataset 
    train_path = os.path.join( file_path, 'dataset.h5')
    test_path = os.path.join(file_path, 'test_data_adele.h5')

    X_train, y_train, X_val, y_val, class_names = load_data_and_labels(train_path, 'train')
    X_test, y_test, _ = load_data_and_labels(test_path, 'test')

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
    augmentations = {
        'rotation_range': 10,
        'width_shift_range': 0.2,
        'shear_range': 0.3,
        'horizontal_flip': True,
        'fill_mode': 'wrap',
    }
    test_augmentations = {}
    NUM_CLASSES = 7
# Conversione delle etichette in one-hot encoding
    y_train_one_hot = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_one_hot = to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test_one_hot = to_categorical(y_test, num_classes=NUM_CLASSES)
    train_generator_focal_smoot = CustomBalancedDataGenerator(X_train, y_train_one_hot, batch_size=64, augmentations=augmentations, data_inf='train', label_smoothing=0.05)
    valid_generator_focal_smoot = CustomBalancedDataGenerator(X_val, y_val_one_hot, batch_size=64, augmentations=augmentations, data_inf='valid', label_smoothing=0)
    test_generator_focal_smoot = CustomBalancedDataGenerator(X_test, y_test_one_hot, batch_size=64, augmentations=test_augmentations, data_inf='test', label_smoothing=0)
    
    return train_generator_focal_smoot, valid_generator_focal_smoot, test_generator_focal_smoot, initial_bias



