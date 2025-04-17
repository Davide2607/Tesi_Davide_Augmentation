# Final Scripts - Facial Expression Recognition

Questa cartella contiene gli script finali utilizzati per il training, il fine-tuning e la generazione delle mappe di attenzione nel progetto di tesi.

## 📁 Struttura della cartella

---

## 📂 `scripts/`: Script Python principali

Questa cartella contiene tutti gli script organizzati per funzione:

### 📊 Analisi e visualizzazione
- `cm_images.py`: genera le confusion matrix a partire dai risultati dei modelli.
- `neptune_init.py`:  tracciamento degli esperimenti con Neptune.ai.

### 🧠 Training e fine-tuning dei modelli
- `train_final_layers.py`: script principale per il training finale dei modelli.
- `finetuning.py`: script per il fine-tuning di modelli pre-addestrati.
- `optimizer_final_layers.py`: definisce l'ottimizzazione standard per la ricerca dei migliori parametri per l'addestramento
dei final layers.
- `optimizer_finetuning.py`: definisce ottimizzatori specifici per il fine-tuning.
- `find_unfreeze.py`: sblocca layer specifici di modelli pre-addestrati per il fine-tuning e testa qual è il numero migliore di layer da scongelare.

### 🛠️ Generazione e gestione dei dati
- `data_generators.py`: contiene le classi per la generazione dinamica dei dati durante il training.
- `loading_data.py`: gestisce il caricamento delle immagini.

### 🧪 Loss e metriche
- `losses.py`: definizione delle funzioni di loss personalizzate.

### 🔁 Test e sperimentazioni varie
- `find_truncated_layers.py`: script per la ricerca del miglior layer di troncamento.
- `backbone.py`: definisce l'architettura dei modelli.

---

## 📂 `sbatch/`
Contiene gli script per il lancio dei job su cluster HPC tramite SLURM.
---

## ℹ️ Note
- I modelli preaddestrati e i risultati ottenuti è possibile trovarli al link [Drive](https://drive.google.com/drive/folders/190lg4UPtDAOTYNwVwmCeQHpThr2FS5AX?usp=share_link)

