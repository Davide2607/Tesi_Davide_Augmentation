import numpy as np
import tensorflow as tf
from neptune_init import init_neptune
import argparse
from pattlite_new import build_model_finetuning,build_model_final_layers
import neptune
from loading_new import carica_dati
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.utils import get_custom_objects
from tensorflow.keras.layers import Dropout, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate
def categorical_focal_loss(alpha=0.25, gamma=2.0):
    """
    Implementazione della categorical focal loss per etichette one-hot.

    Args:
        alpha (float): Ponderazione degli esempi positivi.
        gamma (float): Esponente che controlla il peso degli esempi ben classificati.

    Returns:
        Callable: Funzione di perdita focal loss.
    """
    def loss(y_true, y_pred):
        # Garantisce che le predizioni siano comprese tra 0 e 1
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

        # Calcolo della cross-entropy
        ce = -y_true * tf.math.log(y_pred)

        # Calcolo del fattore modulatorio focal
        modulating_factor = tf.pow(1.0 - y_pred, gamma)

        # Calcolo della focal loss
        focal_loss = alpha * modulating_factor * ce

        # Ritorno della perdita media per batch
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    return loss
# Funzione per valutare il modello
def valuta_modello(model, test_generator, run, model_name):
    class_names = ['ANGER', 'DISGUST', 'FEAR', 'HAPPINESS', 'NEUTRALITY', 'SADNESS', 'SURPRISE']
    test_loss, test_acc = model.evaluate(test_generator)
    run[f"{model_name}/final_layers/test/loss"].append(test_loss)
    run[f"{model_name}/final_layers/test/accuracy"].append(test_acc)

    # Predizioni del modello
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.concatenate([np.argmax(y, axis=1) for _, y in test_generator], axis=0)

    # Calcola la confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print("Confusion Matrix:")
    print(cm)
    print("Normalized Confusion Matrix:")
    print(cm_normalized)

    # Visualizza la confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{model_name}_cm.png')
    plt.show()

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Salva il grafico
    plt.savefig(f'{model_name}_cm_normalized.png')
    plt.show()

    return test_loss, test_acc


# Aggiungi la funzione di perdita personalizzata al dizionario degli oggetti personalizzati
custom_objects = {'loss': categorical_focal_loss()}

# Definisci la tua classe personalizzata per il layer (se non è definita già)
class ExpandDimsLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

# Registra il layer personalizzato
get_custom_objects().update({'ExpandDimsLayer': ExpandDimsLayer})

class SqueezeLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)
    
# Registra il layer personalizzato
get_custom_objects().update({'SqueezeLayer': SqueezeLayer})

from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_tsne(model, test_generator, class_names,model_name):
    """
    Funzione per visualizzare le features ridotte con t-SNE e visualizzare le classi
    anziché i numeri nella legenda, con colori più distintivi.

    Args:
    - model: Il modello addestrato.
    - test_generator: Il generatore di test.
    - class_names: Lista dei nomi delle classi, dove l'indice corrisponde alla classe numerica.
    """
    features = []
    labels = []

    # Ottieni le features e le etichette dai dati di test
    for x_batch, y_batch in test_generator:
        features.append(model.predict(x_batch))
        labels.append(np.argmax(y_batch, axis=1))  # Usa np.argmax per ottenere l'etichetta numerica

    # Unisci tutte le features e le etichette
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Riduzione dimensionale con t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    reduced_features = tsne.fit_transform(features)

    # Crea un elenco di etichette di classe corrispondenti
    class_labels = [class_names[label] for label in labels]

    # Visualizza il grafico t-SNE con colori distintivi
    plt.figure(figsize=(10, 8))

    # Usa una palette diversa (Set1 è più vivida e distintiva)
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=class_labels, palette='Set1', s=50, edgecolor='k')

    # Aggiungi il titolo e la legenda
    plt.title('t-SNE Visualization of Features')
    plt.legend(title='Class', loc='best')

    # Salva la figura
    plt.savefig(f'tsne_{model_name}_distinct_colors.png')
    plt.show()


def main():
    
    # Inizializza Neptune
    run = init_neptune()

    # Carica i dati
    _,_, test_generator, initial_bias = carica_dati()

    models = ['PattLite', 'EfficientNetB1', 'VGG19', 'ResNet50', 'InceptionV3', 'ConvNeXt']
    
    with tf.keras.utils.custom_object_scope(custom_objects):
        # Genera un bias iniziale casuale per ciascuna classe
        num_classes = 7
        initial_bias = np.random.randn(num_classes)
        model = build_model_final_layers(0.001, 0.1, 0.1, initial_bias, 'EfficientNetB1')
        backbone = model.get_layer('base_model')
        backbone.trainable = True
        unfreeze = 114
        fine_tune_from = len(backbone.layers) - unfreeze
        for layer in backbone.layers[:fine_tune_from]:
            layer.trainable = False
        for layer in backbone.layers[fine_tune_from:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        self_attention = model.get_layer('attention')
        patch_extraction = model.get_layer('patch_extraction')
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        global_average_layer = model.get_layer('gap')
        prediction_layer = model.get_layer('classification_head')
        IMG_SHAPE = (128, 128, 3)
        input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
        #sample_resizing = tf.keras.layers.Resizing(128, 128, name="resize")
        l2_reg = 0.07099871122599184
        learning_rate = 0.0005486860365638318
        dropout_rate = 0.4603464152900125
        x = input_layer
        pre_classification = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='relu', kernel_regularizer = l2(l2_reg)),
                                              tf.keras.layers.BatchNormalization()], name='pre_classification')


        x = preprocess_input(x)
        x = backbone(x, training=False)
        x = patch_extraction(x)
        x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)
        x = global_average_layer(x)
        x = Dropout(dropout_rate)(x)
        x = pre_classification(x)
        x = ExpandDimsLayer(axis=-1)(x)
        x = self_attention([x, x])
        x = SqueezeLayer(axis=-1)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = prediction_layer(x)

        model = Model(inputs=input_layer, outputs=outputs, name='train-head')
        model.summary(show_trainable=True)
        model.load_weights(f'/home/famato/final_scripts/scripts/model/finetuning/pretrained_EfficientNetB1_finetuning_weights.h5')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=3.0),
                  loss= categorical_focal_loss(alpha=0.25, gamma=2.0),
                  metrics=['categorical_accuracy'])
        efficientnet = model
    with tf.keras.utils.custom_object_scope(custom_objects):
        
        convnext = tf.keras.models.load_model(f'/home/famato/final_scripts/scripts/model/finetuning/pretrained_ConvNeXt_finetuning')
        pattlite = tf.keras.models.load_model(f'/home/famato/final_scripts/scripts/model/finetuning/pretrained_PattLite_finetuning')
        vgg19 = tf.keras.models.load_model(f'/home/famato/final_scripts/scripts/model/finetuning/pretrained_VGG19_finetuning')
        #efficientnet = tf.keras.models.load_model(f'/home/famato/final_scripts/scripts/model/finetuning/pretrained_EfficientNetB1_finetuning')
        resnet50 = tf.keras.models.load_model(f'/home/famato/final_scripts/scripts/model/finetuning/pretrained_ResNet_finetuning')
        inception = tf.keras.models.load_model(f'/home/famato/final_scripts/scripts/model/finetuning/pretrained_InceptionV3_finetuning')

    models_run = [pattlite, efficientnet, vgg19, resnet50, inception, convnext]

    # Valuta il modello
    for i,m in enumerate(models):
        # _,_=valuta_modello(models_run[i], test_generator, run, m)
        # ESEMPIO DI UTILIZZO:
        # 1) Definisci i nomi delle classi, dove l'indice corrisponde all'etichetta numerica.
        class_names = ['ANGER', 'DISGUST', 'FEAR', 'HAPPINESS', 'NEUTRALITY', 'SADNESS', 'SURPRISE']

        # 2) Calcola t-SNE per il tuo modello e il generatore di test.
        plot_tsne(models_run[i], test_generator, class_names,m)


if __name__ == "__main__":
    main()