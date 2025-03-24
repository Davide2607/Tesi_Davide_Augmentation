import numpy as np
import tensorflow as tf
from neptune_init import init_neptune
import argparse
from pattlite_new import build_model_final_layers
import neptune
from loading_new import carica_dati

# Funzione per addestrare il modello
def addestra_modello(model, train_generator, valid_generator,test_generator, TRAIN_EPOCH, TRAIN_ES_PATIENCE, TRAIN_LR_PATIENCE, ES_LR_MIN_DELTA, TRAIN_MIN_LR, run, model_name):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=TRAIN_ES_PATIENCE, min_delta=ES_LR_MIN_DELTA, restore_best_weights=True)
    learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=TRAIN_LR_PATIENCE, verbose=0, min_delta=ES_LR_MIN_DELTA, min_lr=TRAIN_MIN_LR)

    history = model.fit(train_generator, epochs=TRAIN_EPOCH, validation_data=valid_generator, verbose=1,
                        callbacks=[early_stopping_callback, learning_rate_callback])
    
    test_loss, test_acc = model.evaluate(test_generator)
     # Loggare l'accuratezza del training e della validazione su Neptune
    for epoch in range(len(history.history['categorical_accuracy'])):
        run[f"{model_name}/final_layers/training/accuracy"].log(history.history['categorical_accuracy'][epoch])
        run[f"{model_name}/final_layers/validation/accuracy"].log(history.history['val_categorical_accuracy'][epoch])
        run[f"{model_name}/final_layers/training/loss"].log(history.history['loss'][epoch])
        run[f"{model_name}/final_layers/validation/loss"].log(history.history['val_loss'][epoch])
        
    run[f"{model_name}/final_layers/test/loss"].log(test_loss)
    run[f"{model_name}/final_layers/test/accuracy"].log(test_acc)
    
    return history

# Funzione per valutare il modello
def valuta_modello(model, test_generator, run, model_name):
    test_loss, test_acc = model.evaluate(test_generator)
    run[f"{model_name}/final_layers/test/loss"].append(test_loss)
    run[f"{model_name}/final_layers/test/accuracy"].append(test_acc)
    return test_loss, test_acc

# Funzione per salvare il modello e la storia dell'addestramento
def salva_modello(model, run, model_name):
    try:
        # Salva il modello in formato TensorFlow
        model.save(f'model/pretrained_{model_name}_final_layers', save_format='tf')
        print(f"Model saved as pretrained_{model_name}_final_layers")
    except Exception as e:
        print(f"An error occurred while saving the model in TensorFlow format: {e}")

    try:
        # Salva il modello in formato HDF5
        model.save(f'model/pretrained_{model_name}_final_layers.h5', save_format='h5')
        print(f"Model saved as pretrained_{model_name}_final_layers.h5")
    except Exception as e:
        print(f"An error occurred while saving the model in HDF5 format: {e}")

    try:
        # Salva il modello in formato Keras
        model.save(f'model/pretrained_{model_name}_final_layers.keras', save_format='keras')
        print(f"Model saved as pretrained_{model_name}_final_layers.keras")
    except Exception as e:
        print(f"An error occurred while saving the model in Keras format: {e}")

    try:
        # Salva i pesi del modello
        model.save_weights(f'model/pretrained_{model_name}_final_layers_weights.h5')
        print(f"Model weights saved as pretrained_{model_name}_final_layers_weights.h5")
    except Exception as e:
        print(f"An error occurred while saving the model weights: {e}")

    try:
        # Carica i file su Neptune
        run[f"{model_name}/saved_model"].upload(f'pretrained_{model_name}_final_layers')
        run[f"{model_name}/saved_weights"].upload(f'pretrained_{model_name}_final_layers_weights.h5')
    except Exception as e:
        print(f"An error occurred while uploading the model to Neptune: {e}")

# Funzione principale
def main():
    # Inizializza Neptune
    run = init_neptune()

    # Definisci gli argomenti della linea di comando
    parser = argparse.ArgumentParser(description='Training parameters for Final Layers')
    parser.add_argument('--l2_reg', type=float, required=True, help='L2 regularization parameter')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, required=True, help='Dropout rate')
    parser.add_argument('--TRAIN_EPOCH', type=int, required=True, help='Training epochs')
    parser.add_argument('--model_name', type=str, required=True, help='Model name. Default is PattLite', default='PattLite')
    args = parser.parse_args()

    # Recupera i parametri dalla linea di comando
    l2_reg = args.l2_reg
    TRAIN_LR = args.learning_rate
    TRAIN_DROPOUT = args.dropout_rate
    TRAIN_EPOCH = args.TRAIN_EPOCH
    model_name = args.model_name

    # Carica i dati
    train_generator, valid_generator, test_generator, initial_bias = carica_dati()


    model = build_model_final_layers(TRAIN_LR, TRAIN_DROPOUT, l2_reg, initial_bias, model_name)

    print(model.summary())
    # Logga i parametri di addestramento su Neptune
    run[f"{model_name}/parameters"] = {
        "learning_rate": TRAIN_LR,
        "dropout_rate": TRAIN_DROPOUT,
        "l2_reg": l2_reg,
        "epochs": TRAIN_EPOCH,
        "batch_size": 64
    }

    # Addestra il modello
    history = addestra_modello(model, train_generator, valid_generator, test_generator, TRAIN_EPOCH, 50, 15, 0.003, 1e-6, run, model_name)

    # Valuta il modello
    _, _ = valuta_modello(model, test_generator, run, model_name)

    # Salva il modello e la storia dell'addestramento
    salva_modello(model, run, model_name)

    # Termina la sessione di Neptune
    run.stop()

if __name__ == "__main__":
    main()