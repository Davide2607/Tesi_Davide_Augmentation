import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

import tensorflow as tf

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

