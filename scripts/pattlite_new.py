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

def build_model_final_layers(learning_rate, dropout_rate, l2_reg, initial_bias, model_name='PattLite'):
    NUM_CLASSES = 7
    IMG_SHAPE = (128, 128, 3)
    input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')

    # Seleziona il modello backbone
    if model_name == 'EfficientNetB1':
        backbone = EfficientNetB1(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        #### prima era block5c_add poi block5a_project_conv
        base_model = Model(backbone.input, backbone.get_layer('block5c_add').output, name='base_model')
    elif model_name == 'VGG19':
        backbone = VGG19(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.vgg19.preprocess_input
        #### prima era block4_pool
        base_model = Model(backbone.input, backbone.get_layer('block4_pool').output, name='base_model')
    elif model_name == 'PattLite':
        backbone = MobileNet(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        base_model = Model(backbone.input, backbone.get_layer('conv_dw_9_bn').output, name='base_model')
        #### prima era backbone.layers[-29].output poi conv_dw_9_bn
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
    elif model_name == 'ResNet':
        backbone = ResNet50V2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        #### prima era conv4_block5_out poi conv4_block1_preact_relu
        base_model = Model(backbone.input, backbone.get_layer('conv4_block5_out').output, name='base_model')
    elif model_name == 'ConvNeXt':
        backbone = tf.keras.applications.ConvNeXtBase(
        include_top=False,
        include_preprocessing=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=(128,128,3),
        classifier_activation='softmax'
        )
        ### prima era convnext_base_stage_2_block_24_identity
        base_model = Model(backbone.input, backbone.get_layer('convnext_base_stage_2_block_24_identity').output, name='base_model')

    elif model_name == 'InceptionV3':
        backbone = InceptionV3(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        #### prima era mixed5 poi mixed4
        base_model = Model(backbone.input, backbone.get_layer('mixed4').output, name='base_model')

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

    x = input_layer
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



def build_model_finetuning(learning_rate, dropout_rate, l2_reg, initial_bias, model_name='PattLite', run=None):


 # Seleziona il modello backbone
    if model_name == 'EfficientNetB1':
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = build_model_final_layers(learning_rate, dropout_rate, l2_reg, initial_bias, model_name)
            model.load_weights(f'model/pretrained_{model_name}_final_layers_weights.h5')
        backbone = model.get_layer('base_model')
        backbone.trainable = True

        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        unfreeze = 114
    elif model_name == 'VGG19':
         # Scarica il modello dal server locale
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(f'model/pretrained_{model_name}_final_layers')
        backbone = model.get_layer('base_model')
        backbone.trainable = True

        preprocess_input = tf.keras.applications.vgg19.preprocess_input
        unfreeze = 3
    elif model_name == 'PattLite':

         # Scarica il modello dal server locale
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(f'model/pretrained_{model_name}_final_layers')
        backbone = model.get_layer('base_model')
        backbone.trainable = True
        ### prima era unfreeze = 10
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
        unfreeze = 54
    elif model_name == 'ResNet':
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(f'model/prima_prova/pretrained_{model_name}_final_layers')
        backbone = model.get_layer('base_model')
        backbone.trainable = True

        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        ## prima era unfreeze = 20
        unfreeze = 70
    elif model_name == 'ConvNeXt':
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(f'model/pretrained_{model_name}_final_layers')
        backbone = model.get_layer('base_model')
        backbone.trainable = True
        ## prima era unfreeze = 10
        unfreeze = 241
    elif model_name == 'InceptionV3':
         # Scarica il modello dal server locale
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(f'model/pretrained_{model_name}_final_layers')
        backbone = model.get_layer('base_model')
        backbone.trainable = True
        
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        ### prima era unfreeze = 20
        unfreeze = 81

    else:
        raise ValueError(f"Modello '{model_name}' non supportato.")
    

    pre_classification = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='relu', kernel_regularizer = l2(l2_reg)),
                                              tf.keras.layers.BatchNormalization()], name='pre_classification')

    fine_tune_from = len(backbone.layers) - unfreeze
    for layer in backbone.layers[:fine_tune_from]:
        layer.trainable = False
    for layer in backbone.layers[fine_tune_from:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    self_attention = model.get_layer('attention')
    patch_extraction = model.get_layer('patch_extraction')

    global_average_layer = model.get_layer('gap')
    prediction_layer = model.get_layer('classification_head')
    IMG_SHAPE = (128, 128, 3)
    input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
    #sample_resizing = tf.keras.layers.Resizing(128, 128, name="resize")
    x = input_layer
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

    ##### prima prova: alpa =0.25 e gamma = 2.0
    #### seconda prova: alpha = 0.75 e gamma = 3.0
    #### terza prova: alpha = 0.8 e gamma = 5.0
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=3.0),
                  loss= categorical_focal_loss(alpha=0.25, gamma=2.0),
                  metrics=['categorical_accuracy'])
    
    return model

