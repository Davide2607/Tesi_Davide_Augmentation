import argparse
# Definisci la funzione da ottimizzare
from bayes_opt import BayesianOptimization
from pattlite_new import build_model_finetuning
from neptune_init import init_neptune
from loading_new import carica_dati
import tensorflow as tf 
from losses import categorical_focal_loss
import tensorflow as tf
from tensorflow.keras.losses import Loss


# Definisci la funzione per creare e addestrare il modello
def train_model(learning_rate, dropout_rate, l2_reg,run, train_generator_focal_smoot, valid_generator_focal_smoot, test_generator_focal_smoot, initial_bias, model_name='PattLite'):
    

    model = build_model_finetuning(learning_rate, dropout_rate, l2_reg, initial_bias, model_name,run)
    

    history = model.fit(train_generator_focal_smoot, epochs=15 ,verbose=1, validation_data=valid_generator_focal_smoot)

    # Durante il training
    for _, (train_acc, val_acc, train_loss, val_loss) in enumerate(zip(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'], history.history['loss'], history.history['val_loss'])):
        run[f"{model_name}/finetuning/training/accuracy"].append(train_acc)
        run[f"{model_name}/finetuning/validation/accuracy"].append(val_acc)
        run[f"{model_name}/finetuning/training/loss"].append(train_loss)
        run[f"{model_name}/finetuning/validation/loss"].append(val_loss)
      
    # test_loss, test_acc = model.evaluate(test_generator_focal_smoot)
    # run[f"{model_name}/test/loss"].append(test_loss)
    # run[f"{model_name}/test/accuracy"].append(test_acc)
    accuracy = max(history.history['val_categorical_accuracy'])
    return accuracy


def optimize_model( train_generator_focal_smoot, valid_generator_focal_smoot, test_generator_focal_smoot, initial_bias,learning_rate, dropout_rate, l2_reg, model_name, run):
    # Logga gli iperparametri della prova corrente
    params_final_layers = f"learning rate = {learning_rate}, dropout_rate = {dropout_rate}, l2_reg = {l2_reg}"
    run[f"{model_name}/hyperparameters"].append(params_final_layers)


    accuracy = train_model(learning_rate, dropout_rate, l2_reg, run, train_generator_focal_smoot, valid_generator_focal_smoot, test_generator_focal_smoot, initial_bias, model_name)
    
    # Logga la metrica di interesse
    run["accuracy"] = accuracy

    return accuracy




# Funzione per gestire gli argomenti da linea di comando
def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization and Model Training")
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='PattLite', choices=['PattLite', 'MobileNet', 'ResNet', 'EfficientNetB1', 'VGG19', 'InceptionV3','Yolo', 'ConvNeXt'], 
                        help='The model name to train (default: PattLite)')
    return parser.parse_args()
    
def main():

    run = init_neptune()

    # Assicurati che TensorFlow utilizzi la GPU

    # Verifica la disponibilità della GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU trovata e configurata correttamente.")
            run[f"config"].append("GPU trovata e configurata correttamente.")
            run[f"config"].append(f"New Distribution")
        except RuntimeError as e:
            print(f"Errore durante la configurazione della GPU: {e}")
            run["config"].append(f"Errore durante la configurazione della GPU: {e}")
    else:
        print("Nessuna GPU trovata, utilizzo della CPU.")
        run['config'].append("Nessuna GPU trovata, utilizzo della CPU.")

    # Aggiungi il parsing degli argomenti da linea di comando
    args = parse_args()
    model_name = args.model_name
    lr_max = args.learning_rate
    # Imposta il range degli iperparametri
    pbounds = {
        'learning_rate': (1e-6, lr_max),
        'dropout_rate': (0.1, 0.5),
        'l2_reg': (1e-3, 1e-1)
    }
    # Funzione per caricare i dati e inizializzare il modello
    # Carica i dati
    train_generator_focal_smoot, valid_generator_focal_smoot, test_generator_focal_smoot, initial_bias = carica_dati()
    


    optimizer = BayesianOptimization(
            f=lambda learning_rate, dropout_rate, l2_reg: optimize_model( train_generator_focal_smoot, valid_generator_focal_smoot, test_generator_focal_smoot, initial_bias,learning_rate, dropout_rate, l2_reg, model_name, run),
            pbounds=pbounds,
            random_state=42,
        )

    # Avvia l'ottimizzazione
    optimizer.maximize(
        init_points=5,
        n_iter=150,
    )

    best_params = optimizer.max['params']
    for param, value in best_params.items():
        run[f"{args.model_name}/best_params_finetuning/{param}"] = value

if __name__ == "__main__":
    main()
