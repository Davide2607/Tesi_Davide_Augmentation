import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle

class CustomBalancedDataGenerator(Sequence):
    def __init__(self, x_data, y_data, batch_size, augmentations=None, data_inf=None, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.data_inf = data_inf
        self.label_smoothing = label_smoothing
        self.indices = np.arange(len(x_data))

        # Se siamo in 'train' o 'valid', impostiamo le augmentation e il bilanciamento
        if data_inf in ['train', 'valid']:
            #print(y_data)
            self.augmentations = ImageDataGenerator(**augmentations)
            self.classes = np.unique(np.argmax(y_data, axis=1))  # Ricaviamo le classi dai dati one-hot encoded
            self.class_indices = {cls: np.where(np.argmax(y_data, axis=1) == cls)[0] for cls in self.classes}
            self.num_classes = len(self.classes)
            self.samples_per_class = max(1, self.batch_size // self.num_classes)

            # Coda ciclica per le classi minoritarie
            self.class_pointers = {cls: 0 for cls in self.classes}

        # Se siamo in 'test', usiamo solo rescale e nessuna augmentation o bilanciamento
        elif data_inf == 'test':
            self.augmentations = ImageDataGenerator(**(augmentations or {}))

        self.on_epoch_end()
        print(f"Generator initialized: {data_inf} mode")

    def __len__(self):
        return int(np.ceil(len(self.x_data) / self.batch_size))

    def __getitem__(self, index):
        if self.data_inf == 'test':
            # Per il test set, usiamo semplicemente gli indici
            start_idx = index * self.batch_size
            end_idx = min((index + 1) * self.batch_size, len(self.x_data))
            batch_x = self.x_data[start_idx:end_idx]
            batch_y = self.y_data[start_idx:end_idx]
        else:
            # Per train/valid, selezioniamo batch bilanciati
            batch_x, batch_y = [], []
            for cls in self.classes:
                cls_indices = self.class_indices[cls]
                cls_pointer = self.class_pointers[cls]

                # Seleziona i dati dalla coda ciclica
                selected_indices = cls_indices[cls_pointer:cls_pointer + self.samples_per_class]
                batch_x.extend(self.x_data[selected_indices])
                batch_y.extend(self.y_data[selected_indices])

                # Aggiorna il puntatore per la classe
                self.class_pointers[cls] += len(selected_indices)

                # Se abbiamo esaurito i dati per la classe, fai uno shuffle e riparti
                if self.class_pointers[cls] >= len(cls_indices):
                    self.class_pointers[cls] = 0
                    np.random.shuffle(cls_indices)  # Shuffle della classe
                    self.class_indices[cls] = cls_indices

            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            batch_x, batch_y = shuffle(batch_x, batch_y)

            # Applica il label smoothing
            if self.label_smoothing > 0:
                batch_y = self.apply_label_smoothing(batch_y)



        # Applica il rescale o le trasformazioni per augmentation
        augmented_batch_x = np.zeros_like(batch_x)
        for i in range(len(batch_x)):
            augmented_batch_x[i] = self.augmentations.random_transform(batch_x[i])

        return augmented_batch_x, batch_y


    def on_epoch_end(self):
        if self.data_inf != 'test':
            print("Epoch ended. Shuffling data.")
            for cls in self.classes:
                np.random.shuffle(self.class_indices[cls])  # Shuffle degli indici per ogni classe

    def apply_label_smoothing(self, labels):
        """Applica il label smoothing alle etichette one-hot"""
        if self.label_smoothing > 0:
            labels = labels.astype(np.float32)  # Assicurati che sia in formato float
            num_classes = labels.shape[1]  # Ottieni il numero di classi (assumendo one-hot encoding)
            smooth_value = self.label_smoothing / (num_classes - 1)  # Calcolo del valore per le classi non corrette
            smoothed_labels = np.ones_like(labels, dtype=np.float32) * smooth_value  # Etichette smussate per tutte le classi
            for i in range(len(labels)):
                true_class = np.argmax(labels[i])  # Ottieni la classe corretta (indice della classe 1)
                smoothed_labels[i, true_class] = 1.0 - self.label_smoothing  # Imposta la probabilità della classe corretta
            return smoothed_labels
        else:
            return labels


