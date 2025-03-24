import numpy as np
from neptune_init import init_neptune
import argparse
from pattlite_new import build_model_final_layers
from carica_dati_prova import carica_dati
from tensorflow.keras.models import Model
from losses import categorical_focal_loss
from sklearn.model_selection import ParameterGrid
import gc
import neptune
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNet, ResNet50V2, VGG19, EfficientNetB1, InceptionV3, ConvNeXtBase
import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dropout, Dense, SeparableConv2D, BatchNormalization
from losses import categorical_focal_loss


# Aggiungi la funzione di perdita personalizzata al dizionario degli oggetti personalizzati
custom_objects = {'loss': categorical_focal_loss()}


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

def build_model_final_layers(learning_rate, dropout_rate, l2_reg, initial_bias, layer_name, model_name='PattLite'):
    NUM_CLASSES = 7
    IMG_SHAPE = (120, 120, 3)
    input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
    sample_resizing = tf.keras.layers.Resizing(128, 128, name="resize")

    # Seleziona il modello backbone
    if model_name == 'EfficientNetB1':
        backbone = EfficientNetB1(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer(layer_name).output, name='base_model')
    elif model_name == 'VGG19':
        backbone = VGG19(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.vgg19.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer(layer_name).output, name='base_model')
    elif model_name == 'PattLite':
        backbone = MobileNet(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        base_model = Model(backbone.input, backbone.get_layer(layer_name).output, name='base_model')
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
    elif model_name == 'ResNet':
        backbone = ResNet50V2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer(layer_name).output, name='base_model')
    elif model_name == 'ConvNeXt':
        backbone = tf.keras.applications.ConvNeXtBase(
        include_top=False,
        include_preprocessing=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=(128,128,3),
        classifier_activation='softmax'
        )
        base_model = Model(backbone.input, backbone.get_layer(layer_name).output, name='base_model')

    elif model_name == 'InceptionV3':
        backbone = InceptionV3(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer(layer_name).output, name='base_model')

    else:
        raise ValueError(f"Modello '{model_name}' non supportato.")
    
   
    base_model.trainable = False

    self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')
    patch_extraction = tf.keras.Sequential([
        SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'),
        SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(l2_reg))
    ], name='patch_extraction')

    global_average_layer = GlobalAveragePooling2D(name='gap')
    pre_classification = tf.keras.Sequential([Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
                                              BatchNormalization()], name='pre_classification')
    prediction_layer = Dense(NUM_CLASSES, activation="softmax", name='classification_head', bias_initializer=Constant(initial_bias))

    x = sample_resizing(input_layer)
    if model_name != 'ConvNeXt':
        x = preprocess_input(x)
    x = base_model(x, training=False)
    x = patch_extraction(x)
    x = global_average_layer(x)
    x = Dropout(dropout_rate)(x)
    x = pre_classification(x)
    x = ExpandDimsLayer(axis=-1)(x)
    x = self_attention([x, x])
    x = SqueezeLayer(axis=-1)(x)
    outputs = prediction_layer(x)

    model = Model(inputs=input_layer, outputs=outputs, name='train-head')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=3.0),
                loss= categorical_focal_loss(alpha=0.25, gamma=2.0),
                metrics=['categorical_accuracy'])

    return model

# Funzione per addestrare il modello
def addestra_modello(model, train_generator, valid_generator, TRAIN_EPOCH, TRAIN_ES_PATIENCE, TRAIN_LR_PATIENCE, ES_LR_MIN_DELTA, TRAIN_MIN_LR, run, model_name, layer_name, learning_rate, dropout, l2_reg):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=TRAIN_ES_PATIENCE, min_delta=ES_LR_MIN_DELTA, restore_best_weights=True)
    learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=TRAIN_LR_PATIENCE, verbose=0, min_delta=ES_LR_MIN_DELTA, min_lr=TRAIN_MIN_LR)

    history = model.fit(train_generator, epochs=TRAIN_EPOCH, validation_data=valid_generator, verbose=1,
                        callbacks=[early_stopping_callback, learning_rate_callback])
    
    acc_val = max(history.history['val_categorical_accuracy'])
    # Durante il training
    for _, (train_acc, val_acc, train_loss, val_loss) in enumerate(zip(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'], history.history['loss'], history.history['val_loss'])):
        run[f"{model_name}/training/accuracy"].append(train_acc)
        run[f"{model_name}/validation/accuracy"].append(val_acc)
        run[f"{model_name}/training/loss"].append(train_loss)
        run[f"{model_name}/validation/loss"].append(val_loss)

    
    run[f"{model_name}/final_layers"].append(f"accuracy= {acc_val} at layer {layer_name} with learning rate {learning_rate}, dopout {dropout}, l2_reg {l2_reg}")

    
    
    return history

# Funzione per valutare il modello
def valuta_modello(model, test_generator, run, model_name, layer_name):
    test_loss, test_acc = model.evaluate(test_generator)
    run[f"{model_name}/final_layers/test"].append(f"loss= {test_loss} and accuracy= {test_acc} ad layer {layer_name}")
    return test_loss, test_acc

# Funzione principale
def main():
    # Inizializza Neptune
    run = init_neptune()
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU trovata e configurata correttamente.")
            run[f"config"].append("GPU trovata e configurata correttamente.")
        except RuntimeError as e:
            print(f"Errore durante la configurazione della GPU: {e}")
            run["config"].append(f"Errore durante la configurazione della GPU: {e}")
    else:
        print("Nessuna GPU trovata, utilizzo della CPU.")
        run['config'].append("Nessuna GPU trovata, utilizzo della CPU.")

    # Definisci gli argomenti della linea di comando
    parser = argparse.ArgumentParser(description='Testing different layers accuracy for Final Layers')
    parser.add_argument('--model_name', type=str, required=True, help='Model name. Default is PattLite', default='PattLite')
    args = parser.parse_args()


    TRAIN_EPOCH = 10
    TRAIN_ES_PATIENCE = 3
    TRAIN_LR_PATIENCE = 2
    ES_LR_MIN_DELTA = 0.0001
    TRAIN_MIN_LR = 1e-6
    model_name = args.model_name

    # Carica i dati
    train_generator, valid_generator, test_generator, initial_bias = carica_dati()
    if model_name ==    'PattLite': 
        backbone = MobileNet(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        base_model = Model(backbone.input, backbone.layers[-29].output, name='base_model')
    elif model_name == 'VGG19':
        base_model = VGG19(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    elif model_name == 'ResNet':
        base_model = ResNet50V2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        
    elif model_name == 'EfficientNetB1':
        base_model = EfficientNetB1(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    elif model_name == 'ConvNeXt':
        base_model = tf.keras.applications.ConvNeXtBase(
        include_top=False,
        include_preprocessing=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=(128,128,3),
        classifier_activation='softmax'
        )
        selected_layers = [layer for layer in base_model.layers if 'input' not in layer.name and ('conv' in layer.name.split('_')[-1] or  'conv' in layer.name.split('_')[-2]) and 'pointwise_conv_2' in layer.name  and layer and tuple(layer.output.shape[1:4]) == (8, 8, 512)]
        index = 4
        shape = (8, 8, 512)
    if model_name == 'EfficientNetB1':
        # Identifica i blocchi convoluzionali principali (MBConv blocks)
        conv_blocks = [layer for layer in base_model.layers if 'block' in layer.name and 'conv' in layer.name]

        # Stampa i blocchi convoluzionali principali
        for layer in conv_blocks:
            print(f"Layer: {layer.name}, Output shape: {layer.output.shape}")

        # Seleziona un sottoinsieme di layer basato su criteri specifici
        selected_layers = [layer for layer in conv_blocks if layer.output.shape[1:4] in [(8, 8,112)]]
        count = 0
        # Stampa i layer selezionati
        for layer in selected_layers:
            print(f"Selected Layer: {layer.name}, Output shape: {layer.output.shape}")
            count +=1
            layers = selected_layers
        index = 3
        shape = (8, 8)
    elif model_name == 'InceptionV3':
        
        selected_layers = [layer for layer in base_model.layers if 'input' not in layer.name and 'mixed' in layer.name and tuple(layer.output.shape[1:3]) > (2, 2)]
        layers = selected_layers
    elif model_name == 'ResNet':
        conv_blocks = [layer for layer in base_model.layers if 'block' in layer.name and 'conv' in layer.name]
        selected_layers = [layer for layer in conv_blocks if layer.output.shape[1:4] in [(8, 8,512)]]   
        layers = selected_layers
        index = 4
        shape = (8, 8, 512)
    else:
        layers = base_model.layers
        index = 4
        shape = (8, 8, 512)
    
    # Libera la memoria inutilizzata
    del base_model
    # Definisci i parametri da cercare
    param_grid = {
        'learning_rate': [1e-4, 1e-3],
        'dropout': [0.3, 0.5],
        'l2_reg': [1e-3, 1e-1]
    }

    # Itera su tutti i layer del modello backbone
    for layer in layers:
        ##### se il modello è EfficientNetB1:
        # if layer.output_shape[1:3] == (8, 8)
        # altrimenti: 
        if model_name == 'InceptionV3':
            # Itera attraverso i parametri della griglia
            for params in ParameterGrid(param_grid):

                learning_rate = params['learning_rate']
                dropout = params['dropout']
                l2_reg = params['l2_reg']
                model = build_model_final_layers(learning_rate, dropout, l2_reg, initial_bias, layer.name, model_name)
                
                print(f"Testing model with {layer.name} as final layer")
                
                # Riaddestra i layer finali
                history = addestra_modello(model, train_generator, valid_generator, TRAIN_EPOCH, TRAIN_ES_PATIENCE, TRAIN_LR_PATIENCE, ES_LR_MIN_DELTA, TRAIN_MIN_LR, run, model_name, layer.name, learning_rate, dropout, l2_reg)

                # Libera la memoria inutilizzata
                del model
                gc.collect()
                tf.keras.backend.clear_session()

        else:
            if layer.output_shape[1:index] == shape:
                
                # Itera attraverso i parametri della griglia
                for params in ParameterGrid(param_grid):

                    learning_rate = params['learning_rate']
                    dropout = params['dropout']
                    l2_reg = params['l2_reg']
                    model = build_model_final_layers(learning_rate, dropout, l2_reg, initial_bias, layer.name, model_name)
                    
                    print(f"Testing model with {layer.name} as final layer")
                    
                    # Riaddestra i layer finali
                    history = addestra_modello(model, train_generator, valid_generator, TRAIN_EPOCH, TRAIN_ES_PATIENCE, TRAIN_LR_PATIENCE, ES_LR_MIN_DELTA, TRAIN_MIN_LR, run, model_name, layer.name, learning_rate, dropout, l2_reg)

                    # Libera la memoria inutilizzata
                    del model
                    gc.collect()
                    tf.keras.backend.clear_session()
                
                
    # Termina la sessione di Neptune
    run.stop()

if __name__ == "__main__":
    main()