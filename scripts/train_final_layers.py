import numpy as np
import tensorflow as tf
from neptune_init import init_neptune
import argparse
import random
import os
from scripts.backbone import build_model_final_layers
from scripts.loading_data import carica_dati


def _parse_class_weights(num_classes: int):
    raw = os.environ.get('CLASS_WEIGHTS', '').strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    if len(parts) != num_classes:
        raise ValueError(
            f"CLASS_WEIGHTS deve avere {num_classes} valori separati da virgola (es. '1,1,2,1,1,1,1'). "
            f"Trovati {len(parts)}."
        )
    weights = {i: float(parts[i]) for i in range(num_classes)}
    return weights


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def log_extra_test_metrics(model, test_generator, run, model_name, stage_name):
    try:
        from sklearn.metrics import (
            balanced_accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
        )
    except Exception as e:
        print(f"sklearn non disponibile, salto metriche aggiuntive: {e}")
        run[f"{model_name}/{stage_name}/test/extra_metrics_error"].log(str(e))
        return

    def _normalize_class_name(name: str) -> str:
        name = name.strip()
        if name.startswith('synthetic_'):
            name = name[len('synthetic_'):]
        normalized = {
            'anger': 'ANGRY',
            'disgust': 'DISGUST',
            'fear': 'FEAR',
            'ear': 'FEAR',
            'happiness': 'HAPPY',
            'neutrality': 'NEUTRAL',
            'sadness': 'SAD',
            'surprise': 'SURPRISE',
        }
        return normalized.get(name.lower(), name)

    def _try_get_class_names() -> list[str] | None:
        dataset_path = os.environ.get('DATASET_H5', '').strip()
        if not dataset_path:
            return None
        try:
            import h5py
            import numpy as _np

            with h5py.File(os.path.expanduser(dataset_path), 'r') as f:
                if 'class_names' not in f:
                    return None
                raw = f['class_names'][:]

            names: list[str] = []
            for v in raw:
                if isinstance(v, (bytes, _np.bytes_)):
                    names.append(v.decode('utf-8'))
                else:
                    names.append(str(v))
            return [_normalize_class_name(n) for n in names]
        except Exception as e:
            print(f"[METRICS][{stage_name}][WARN] Impossibile leggere class_names da DATASET_H5='{dataset_path}': {e}")
            return None

    y_pred_prob = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.concatenate([np.argmax(y_batch, axis=1) for _, y_batch in test_generator], axis=0)

    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    n_classes = int(y_pred_prob.shape[1])
    labels = np.arange(n_classes)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    worst_class_idx = int(np.argmin(recall_per_class))

    class_names = _try_get_class_names()
    if class_names is None or len(class_names) != n_classes:
        class_names = [f"class_{i}" for i in range(n_classes)]

    run[f"{model_name}/{stage_name}/test/precision_macro"].log(float(precision_macro))
    run[f"{model_name}/{stage_name}/test/recall_macro"].log(float(recall_macro))
    run[f"{model_name}/{stage_name}/test/f1_macro"].log(float(f1_macro))
    run[f"{model_name}/{stage_name}/test/precision_weighted"].log(float(precision_weighted))
    run[f"{model_name}/{stage_name}/test/recall_weighted"].log(float(recall_weighted))
    run[f"{model_name}/{stage_name}/test/f1_weighted"].log(float(f1_weighted))
    run[f"{model_name}/{stage_name}/test/balanced_accuracy"].log(float(balanced_acc))

    print(
        f"[METRICS][{stage_name}] "
        f"precision_macro={precision_macro:.6f} "
        f"recall_macro={recall_macro:.6f} "
        f"f1_macro={f1_macro:.6f} "
        f"precision_weighted={precision_weighted:.6f} "
        f"recall_weighted={recall_weighted:.6f} "
        f"f1_weighted={f1_weighted:.6f} "
        f"balanced_accuracy={balanced_acc:.6f}"
    )
    true_counts = np.bincount(y_true, minlength=n_classes)
    pred_counts = np.bincount(y_pred, minlength=n_classes)
    print(
        f"[METRICS][{stage_name}][class_dist] "
        + " ".join([f"{class_names[i]}:true={int(true_counts[i])},pred={int(pred_counts[i])}" for i in range(n_classes)])
    )

    per_class_recall_str = " ".join(
        [f"{class_names[i]}={float(v):.6f}" for i, v in enumerate(recall_per_class)]
    )
    print(
        f"[METRICS][{stage_name}][per_class_recall] "
        f"{per_class_recall_str} "
        f"worst_class={class_names[worst_class_idx]} "
        f"worst_recall={recall_per_class[worst_class_idx]:.6f}"
    )

    per_class_prf_str = " ".join(
        [
            f"{class_names[i]}:p={float(precision_per_class[i]):.3f},r={float(recall_per_class[i]):.3f},f1={float(f1_per_class[i]):.3f},n={int(support_per_class[i])}"
            for i in range(n_classes)
        ]
    )
    print(f"[METRICS][{stage_name}][per_class_prf] {per_class_prf_str}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"[METRICS][{stage_name}][confusion_matrix] rows=true cols=pred")
    header = "pred-> " + " ".join([f"{name[:4]:>4s}" for name in class_names])
    print(header)
    for i, name in enumerate(class_names):
        row = " ".join([f"{int(v):4d}" for v in cm[i].tolist()])
        print(f"{name[:4]:>4s} | {row}")

# Funzione per addestrare il modello
def addestra_modello(model, train_generator, valid_generator,test_generator, TRAIN_EPOCH, TRAIN_ES_PATIENCE, TRAIN_LR_PATIENCE, ES_LR_MIN_DELTA, TRAIN_MIN_LR, run, model_name):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=TRAIN_ES_PATIENCE, min_delta=ES_LR_MIN_DELTA, restore_best_weights=True)
    learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=TRAIN_LR_PATIENCE, verbose=0, min_delta=ES_LR_MIN_DELTA, min_lr=TRAIN_MIN_LR)

    fit_workers = int(os.environ.get('FIT_WORKERS', '4'))
    fit_multiprocessing = os.environ.get('FIT_MULTIPROCESSING', '1') == '1'
    fit_max_queue_size = int(os.environ.get('FIT_MAX_QUEUE_SIZE', '16'))

    class_weight = _parse_class_weights(num_classes=7)

    history = model.fit(
        train_generator,
        epochs=TRAIN_EPOCH,
        validation_data=valid_generator,
        verbose=1,
        callbacks=[early_stopping_callback, learning_rate_callback],
        workers=fit_workers,
        use_multiprocessing=fit_multiprocessing,
        max_queue_size=fit_max_queue_size,
        class_weight=class_weight,
    )
    
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
    log_extra_test_metrics(model, test_generator, run, model_name, 'final_layers')
    return test_loss, test_acc

# Funzione per salvare il modello e la storia dell'addestramento
def salva_modello(model, run, model_name):
    model_dir = os.environ.get('MODEL_DIR', 'model')
    os.makedirs(model_dir, exist_ok=True)
    try:
        # Salva il modello in formato TensorFlow
        model.save(f'{model_dir}/pretrained_{model_name}_final_layers', save_format='tf')
        print(f"Model saved as pretrained_{model_name}_final_layers")
    except Exception as e:
        print(f"An error occurred while saving the model in TensorFlow format: {e}")

    try:
        # Salva il modello in formato HDF5
        model.save(f'{model_dir}/pretrained_{model_name}_final_layers.h5', save_format='h5')
        print(f"Model saved as pretrained_{model_name}_final_layers.h5")
    except Exception as e:
        print(f"An error occurred while saving the model in HDF5 format: {e}")

    try:
        # Salva il modello in formato Keras
        model.save(f'{model_dir}/pretrained_{model_name}_final_layers.keras', save_format='keras')
        print(f"Model saved as pretrained_{model_name}_final_layers.keras")
    except Exception as e:
        print(f"An error occurred while saving the model in Keras format: {e}")

    try:
        # Salva i pesi del modello
        model.save_weights(f'{model_dir}/pretrained_{model_name}_final_layers_weights.h5')
        print(f"Model weights saved as pretrained_{model_name}_final_layers_weights.h5")
    except Exception as e:
        print(f"An error occurred while saving the model weights: {e}")

    try:
        # Carica i file su Neptune
        run[f"{model_name}/saved_model"].upload(f'{model_dir}/pretrained_{model_name}_final_layers')
        run[f"{model_name}/saved_weights"].upload(f'{model_dir}/pretrained_{model_name}_final_layers_weights.h5')
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
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Recupera i parametri dalla linea di comando
    l2_reg = args.l2_reg
    TRAIN_LR = args.learning_rate
    TRAIN_DROPOUT = args.dropout_rate
    TRAIN_EPOCH = args.TRAIN_EPOCH
    model_name = args.model_name
    seed = args.seed

    set_global_seed(seed)

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
        "batch_size": getattr(train_generator, 'batch_size', None),
        "seed": seed,
    }

    # Addestra il modello
    history = addestra_modello(model, train_generator, valid_generator, test_generator, TRAIN_EPOCH, 10, 5, 0.003, 1e-6, run, model_name)

    # Valuta il modello
    _, _ = valuta_modello(model, test_generator, run, model_name)

    # Salva il modello e la storia dell'addestramento
    salva_modello(model, run, model_name)

    # Termina la sessione di Neptune
    run.stop()

if __name__ == "__main__":
    main()