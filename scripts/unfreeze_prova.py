import numpy as np
import tensorflow as tf
from neptune_init import init_neptune
import argparse
import neptune
from carica_dati_prova import carica_dati
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dropout, Dense, SeparableConv2D, BatchNormalization
from losses import categorical_focal_loss
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
import gc

import neptune
import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dropout, Dense, SeparableConv2D, BatchNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNet, ResNet50V2, VGG19, EfficientNetB1, InceptionV3, ConvNeXtBase
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dropout, Dense, SeparableConv2D, BatchNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNet, ResNet50V2, VGG19, EfficientNetB1, InceptionV3, ConvNeXtBase
from losses import categorical_focal_loss
from tensorflow.keras.losses import Loss
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
    
def build_model_finetuning(learning_rate, dropout_rate, l2_reg, initial_bias, model_name='PattLite', run=None, unfreeze = 0, layer_name=None):


 # Seleziona il modello backbone
    if model_name == 'EfficientNetB1':
        with tf.keras.utils.custom_object_scope(custom_objects):
            model,_ = build_model_final_layers(learning_rate, dropout_rate, l2_reg, initial_bias,layer_name, model_name)
            model.load_weights(f'model_new/pretrained_{model_name}_final_layers_weights.h5')

        backbone = model.get_layer('base_model')
        total_layers = len(backbone.layers)
        backbone.trainable = True

        preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    elif model_name == 'VGG19':
         # Scarica il modello dal server locale
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(f'model_new/pretrained_{model_name}_final_layers')
        backbone = model.get_layer('base_model')
        backbone.trainable = True
        total_layers = len(backbone.layers)
        preprocess_input = tf.keras.applications.vgg19.preprocess_input
    elif model_name == 'PattLite':

         # Scarica il modello dal server locale
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(f'model_new_prova/pretrained_{model_name}_final_layers')
        backbone = model.get_layer('base_model')
        backbone.trainable = True
        total_layers = len(backbone.layers)
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
    elif model_name == 'ResNet':
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(f'model_new/pretrained_{model_name}_final_layers')
        backbone = model.get_layer('base_model')
        backbone.trainable = True
        total_layers = len(backbone.layers)
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    elif model_name == 'ConvNeXt':
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(f'model_new/pretrained_{model_name}_final_layers')
        backbone = model.get_layer('base_model')
        backbone.trainable = True
        total_layers = len(backbone.layers)
    elif model_name == 'InceptionV3':
         # Scarica il modello dal server locale
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(f'model_new/pretrained_{model_name}_final_layers')
        backbone = model.get_layer('base_model')
        backbone.trainable = True

        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        total_layers = len(backbone.layers)

    else:
        raise ValueError(f"Modello '{model_name}' non supportato.")
    

    pre_classification = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='relu', kernel_regularizer = l2(l2_reg)),
                                              tf.keras.layers.BatchNormalization()], name='pre_classification')

    fine_tune_from = total_layers - unfreeze
    for layer in backbone.layers[:fine_tune_from]:
        layer.trainable = False
    for layer in backbone.layers[fine_tune_from:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    self_attention = model.get_layer('attention')
    patch_extraction = model.get_layer('patch_extraction')

    global_average_layer = model.get_layer('gap')
    prediction_layer = model.get_layer('classification_head')
    IMG_SHAPE = (120, 120, 3)
    input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
    sample_resizing = tf.keras.layers.Resizing(128, 128, name="resize")
    x = sample_resizing(input_layer)
    if model_name != 'ConvNeXt':
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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=3.0),
                  loss= categorical_focal_loss(alpha=0.25, gamma=2.0),
                  metrics=['categorical_accuracy'])
    
    return model, total_layers

def build_model_final_layers(learning_rate, dropout_rate, l2_reg, initial_bias, layer_name, model_name='PattLite'):
    NUM_CLASSES = 7
    IMG_SHAPE = (120, 120, 3)
    input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
    sample_resizing = tf.keras.layers.Resizing(128, 128, name="resize")
    print(model_name)
    # Seleziona il modello backbone
    if model_name == 'EfficientNetB1':
        backbone = EfficientNetB1(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer(layer_name).output, name='base_model')
        total_layers = len(base_model.layers)
    elif model_name == 'VGG19':
        backbone = VGG19(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.vgg19.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer(layer_name).output, name='base_model')
        total_layers = len(base_model.layers)
    elif model_name == 'PattLite':
        backbone = MobileNet(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        base_model = Model(backbone.input, backbone.get_layer(layer_name).output, name='base_model')
        total_layers = len(base_model.layers)
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
    elif model_name == 'ResNet':
        backbone = ResNet50V2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer(layer_name).output, name='base_model')
        total_layers = len(base_model.layers)
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
        total_layers = len(base_model.layers)
    elif model_name == 'InceptionV3':
        backbone = InceptionV3(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer(layer_name).output, name='base_model')
        total_layers = len(base_model.layers)
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

    return model, total_layers

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

def trova_num_layers(initial_bias, model_name, layer_name):
    _,total_layers = build_model_final_layers(1e-4, 0.3, 1e-3, initial_bias, layer_name, model_name)
    return total_layers

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
    parser.add_argument('--layer_name', type=str, required=True, help='Layer name. Default is attention', default='attention')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate. Default is 1e-4', default=1e-4)
    parser.add_argument('--l2_reg', type=float, required=True, help='L2 regularization. Default is 1e-3', default=1e-3)
    args = parser.parse_args()


    TRAIN_EPOCH = 10
    TRAIN_ES_PATIENCE = 3
    TRAIN_LR_PATIENCE = 2
    ES_LR_MIN_DELTA = 0.0001
    TRAIN_MIN_LR = 1e-6
    model_name = args.model_name

    # Carica i dati
    train_generator, valid_generator, _, initial_bias = carica_dati()

    
    total_layers = trova_num_layers(initial_bias,model_name, args.layer_name)
    
    num_layers_to_unfreeze = list(range(1, total_layers + 1))
    # Definisci una griglia di parametri per il numero di layer da scongelare
    param_dist = {
                  'learning_rate': [1e-5, args.learning_rate],
                  'dropout_rate': [0.3, 0.5],
                  }

    for params in ParameterGrid(param_dist):
        # Esegui la ricerca randomica
        best_accuracy = 0
        best_params = None
        n_iter_search = 10  # Numero di iterazioni di ricerca randomica
        chosen_numbers = set()
        for _ in range(n_iter_search):
            # Filtra la lista dei valori possibili per escludere 0
            valid_choices = [x for x in num_layers_to_unfreeze if x != 0]

            # Scegli un numero di layer da scongelare casualmente dalla lista filtrata, assicurandoti che non sia già stato scelto
            unfreeze = np.random.choice([x for x in valid_choices if x not in chosen_numbers])
            chosen_numbers.add(unfreeze)
            
            # Ricostruisci il modello con il numero specificato di layer da scongelare
            model, _ = build_model_finetuning(params['learning_rate'], params['dropout_rate'], args.l2_reg, initial_bias, model_name, unfreeze=unfreeze, layer_name=args.layer_name)
            
            # Addestra il modello
            history = addestra_modello(model, train_generator, valid_generator, TRAIN_EPOCH, TRAIN_ES_PATIENCE, TRAIN_LR_PATIENCE, ES_LR_MIN_DELTA, TRAIN_MIN_LR, run, model_name, args.layer_name, 1e-4, 0.3, 1e-3)
            
            val_acc = max(history.history['val_categorical_accuracy'])
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_params = {'num_layers_to_unfreeze': unfreeze, 'learning_rate': params['learning_rate'], 'dropout_rate': params['dropout_rate']}
                run[f"{model_name}/finetuning"].append(f"accuracy= {best_accuracy} with {best_params}")

        print(f"Best parameters: {best_params} with accuracy: {best_accuracy}")
        run[f"{model_name}/finetuning"].append(f"FINISHED...")
        run[f"{model_name}/finetuning"].append(f"Best parameters: {best_params} with accuracy: {best_accuracy}")



                
    # Termina la sessione di Neptune
    run.stop()

if __name__ == "__main__":
    main()