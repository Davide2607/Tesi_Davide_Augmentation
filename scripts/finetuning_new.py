import numpy as np
import tensorflow as tf
from neptune_init import init_neptune
import argparse
from pattlite_new import build_model_finetuning
import neptune
from loading_new import carica_dati
import os

# Funzione per addestrare il modello
def addestra_modello(model, train_generator, valid_generator,test_generator, TRAIN_EPOCH, TRAIN_ES_PATIENCE, TRAIN_LR_PATIENCE, ES_LR_MIN_DELTA, TRAIN_MIN_LR, run, model_name):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=TRAIN_ES_PATIENCE, min_delta=ES_LR_MIN_DELTA, restore_best_weights=True, mode = 'max')
    learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=TRAIN_LR_PATIENCE, verbose=0, min_delta=ES_LR_MIN_DELTA, min_lr=TRAIN_MIN_LR)

    history = model.fit(train_generator, epochs=TRAIN_EPOCH, validation_data=valid_generator, verbose=1,
                        callbacks=[early_stopping_callback, learning_rate_callback])
    
     # Loggare l'accuratezza del training e della validazione su Neptune
    for epoch in range(len(history.history['categorical_accuracy'])):
        run[f"{model_name}/finetuning/training/accuracy"].log(history.history['categorical_accuracy'][epoch])
        run[f"{model_name}/finetuning/validation/accuracy"].log(history.history['val_categorical_accuracy'][epoch])
        run[f"{model_name}/finetuning/training/loss"].log(history.history['loss'][epoch])
        run[f"{model_name}/finetuning/validation/loss"].log(history.history['val_loss'][epoch])
        

    
    return history

# Funzione per valutare il modello
def valuta_modello(model, test_generator, run, model_name):
    test_loss, test_acc = model.evaluate(test_generator)
    run[f"{model_name}/finetuning/test/loss"].log(test_loss)
    run[f"{model_name}/finetuning/test/accuracy"].log(test_acc)
    return test_loss, test_acc

# Funzione per ottenere un nome di file unico
def get_unique_filename(base_path, base_name, extension):
    counter = 1
    file_path = f"{base_path}/{base_name}"
    while os.path.exists(file_path):
        file_path = f"{base_path}/{base_name}_{counter}"
        counter += 1
    return file_path

# Funzione per salvare il modello e la storia dell'addestramento
def salva_modello(model, run, model_name):
    base_path = 'model/finetuning'
    
    try:
        # Salva il modello in formato TensorFlow
        tf_model_path = os.path.join(base_path,f'pretrained_{model_name}_finetuning')
        model.save(tf_model_path, save_format='tf')
        print(f"Model saved as {tf_model_path}")
    except Exception as e:
        print(f"An error occurred while saving the model in TensorFlow format: {e}")

    try:
        # Salva il modello in formato HDF5
        h5_model_path = os.path.join(base_path,f'pretrained_{model_name}_finetuning')
        model.save(h5_model_path, save_format='h5')
        print(f"Model saved as {h5_model_path}")
    except Exception as e:
        print(f"An error occurred while saving the model in HDF5 format: {e}")

    try:
        # Salva i pesi del modello
        weights_path = os.path.join(base_path,f'pretrained_{model_name}_finetuning_weights.h5')
        model.save_weights(weights_path)
        weights_path = os.path.join(base_path,f'pretrained_{model_name}_finetuning.weights.h5')
        model.save_weights(weights_path)
        print(f"Model weights saved as {weights_path}")
    except Exception as e:
        print(f"An error occurred while saving the model weights: {e}")

    try:
        keras_model_path = os.path.join(base_path,f'pretrained_{model_name}_finetuning')
        # Salva il modello in formato Keras
        model.save(keras_model_path, save_format='keras')
        print(f"Model saved as pretrained_{model_name}_finetuning.keras")
    except Exception as e:
        print(f"An error occurred while saving the model in Keras format: {e}")

    try:
        # Carica i file su Neptune
        run[f"{model_name}/saved_model"].upload(tf_model_path)
        run[f"{model_name}/saved_weights"].upload(weights_path)
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
    parser.add_argument('--FT_EPOCH', type=int, required=True, help='Training epochs')
    parser.add_argument('--model_name', type=str, required=True, help='Model name. Default is PattLite', default='PattLite')
    args = parser.parse_args()

    # Recupera i parametri dalla linea di comando
    l2_reg = args.l2_reg
    FT_LR = args.learning_rate
    FT_DROPOUT = args.dropout_rate
    FT_EPOCH = args.FT_EPOCH
    model_name = args.model_name

    # Carica i dati
    train_generator, valid_generator, test_generator, initial_bias = carica_dati()


    model = build_model_finetuning(FT_LR, FT_DROPOUT, l2_reg, initial_bias, model_name)


    # Logga i parametri di addestramento su Neptune
    run[f"{model_name}finetuning/parameters"] = {
        "learning_rate": FT_LR,
        "dropout_rate": FT_DROPOUT,
        "l2_reg": l2_reg,
        "epochs": FT_EPOCH,
        "batch_size": 64
    }

    # Addestra il modello
    history = addestra_modello(model, train_generator, valid_generator, test_generator, FT_EPOCH, 50, 15, 0.003, 1e-6, run, model_name)

    # Valuta il modello
    _, _ = valuta_modello(model, test_generator, run, model_name)

    # Salva il modello e la storia dell'addestramento
    salva_modello(model, run, model_name)

    # Termina la sessione di Neptune
    run.stop()

if __name__ == "__main__":
    main()