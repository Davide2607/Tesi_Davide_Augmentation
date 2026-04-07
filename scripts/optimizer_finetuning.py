import argparse
import gc
import numpy as np

# Definisci la funzione da ottimizzare
from bayes_opt import BayesianOptimization
from scripts.backbone import build_model_finetuning
from neptune_init import init_neptune
from scripts.loading_data import carica_dati
import tensorflow as tf
from tensorflow.keras import backend as K


def _compute_balanced_and_f1_macro(model, eval_generator):
    try:
        from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
    except Exception as e:
        raise RuntimeError(f"sklearn non disponibile per calcolare balanced_accuracy/f1_macro: {e}")

    y_pred_prob = model.predict(eval_generator, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.concatenate([np.argmax(y_batch, axis=1) for _, y_batch in eval_generator], axis=0)

    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    f1_macro = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='macro',
        zero_division=0,
    )[2]
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    return float(balanced_acc), float(f1_macro)


# Definisci la funzione per creare e addestrare il modello
def train_model(
    learning_rate,
    dropout_rate,
    l2_reg,
    run,
    train_generator_focal_smoot,
    valid_generator_focal_smoot,
    eval_generator,
    initial_bias,
    trial_epochs,
    model_name='PattLite',
):
    model = None
    history = None
    try:
        model = build_model_finetuning(learning_rate, dropout_rate, l2_reg, initial_bias, model_name, run)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_categorical_accuracy',
                mode='max',
                patience=3,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        history = model.fit(
            train_generator_focal_smoot,
            epochs=trial_epochs,
            verbose=1,
            validation_data=valid_generator_focal_smoot,
            callbacks=callbacks,
        )

        # Durante il training
        for train_acc, val_acc, train_loss, val_loss in zip(
            history.history['categorical_accuracy'],
            history.history['val_categorical_accuracy'],
            history.history['loss'],
            history.history['val_loss'],
        ):
            run[f"{model_name}/finetuning/training/accuracy"].append(train_acc)
            run[f"{model_name}/finetuning/validation/accuracy"].append(val_acc)
            run[f"{model_name}/finetuning/training/loss"].append(train_loss)
            run[f"{model_name}/finetuning/validation/loss"].append(val_loss)

        balanced_acc, f1_macro = _compute_balanced_and_f1_macro(model, eval_generator)
        # Objective: maximize both macro-F1 and balanced accuracy.
        # Using the mean keeps the score in [0, 1].
        score = (balanced_acc + f1_macro) / 2.0

        run[f"{model_name}/eval/balanced_accuracy"].append(float(balanced_acc))
        run[f"{model_name}/eval/f1_macro"].append(float(f1_macro))
        run[f"{model_name}/eval/score"].append(float(score))
        print(
            f"[EVAL][{model_name}] balanced_accuracy={balanced_acc:.6f} f1_macro={f1_macro:.6f} score={score:.6f}"
        )

        return float(score)
    finally:
        # Evita accumulo di grafi/pesi tra i trial BO
        del history
        del model
        K.clear_session()
        gc.collect()


def optimize_model(
    train_generator_focal_smoot,
    valid_generator_focal_smoot,
    eval_generator,
    initial_bias,
    learning_rate,
    dropout_rate,
    l2_reg,
    model_name,
    run,
    trial_epochs,
):
    # Logga gli iperparametri della prova corrente
    params_final_layers = f"learning rate = {learning_rate}, dropout_rate = {dropout_rate}, l2_reg = {l2_reg}"
    run[f"{model_name}/hyperparameters"].append(params_final_layers)

    score = train_model(
        learning_rate,
        dropout_rate,
        l2_reg,
        run,
        train_generator_focal_smoot,
        valid_generator_focal_smoot,
        eval_generator,
        initial_bias,
        trial_epochs,
        model_name,
    )
    
    # Logga la metrica di interesse
    run["score"] = score

    return score




# Funzione per gestire gli argomenti da linea di comando
def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization and Model Training")
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='PattLite', choices=['PattLite', 'MobileNet', 'ResNet', 'EfficientNetB1', 'VGG19', 'InceptionV3','Yolo', 'ConvNeXt'], 
                        help='The model name to train (default: PattLite)')
    parser.add_argument('--trial_epochs', type=int, default=8, help='Max epochs per BO trial')
    parser.add_argument('--init_points', type=int, default=4, help='Initial random BO points')
    parser.add_argument('--n_iter', type=int, default=20, help='BO optimization iterations')
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

    # Riduce warning del layout optimizer e puo abbassare overhead memoria
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

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
            f=lambda learning_rate, dropout_rate, l2_reg: optimize_model(
                train_generator_focal_smoot,
                valid_generator_focal_smoot,
                test_generator_focal_smoot,
                initial_bias,
                learning_rate,
                dropout_rate,
                l2_reg,
                model_name,
                run,
                args.trial_epochs,
            ),
            pbounds=pbounds,
            random_state=42,
        )

    # Avvia l'ottimizzazione
    optimizer.maximize(
        init_points=args.init_points,
        n_iter=args.n_iter,
    )

    best_params = optimizer.max['params']
    for param, value in best_params.items():
        run[f"{args.model_name}/best_params_finetuning/{param}"] = value
    print(f"[BEST][{args.model_name}][finetuning] {best_params}")

if __name__ == "__main__":
    main()
