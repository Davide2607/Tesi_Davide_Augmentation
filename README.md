# Davide_Augmentation_Thesis

PoliTO thesis project focused on improving facial expression recognition accuracy through data augmentation techniques.

Author: Davide Ravida

## Struttura del repository

Il progetto è organizzato in cartelle che separano la documentazione, la generazione dei dati sintetici, l'addestramento dei modelli e gli script per l'esecuzione su cluster HPC:

- `README.md`: descrizione generale del progetto e istruzioni di base.
- `docs/`: file di supporto e dipendenze del progetto; `requiriments.txt` elenca i pacchetti Python necessari.
- `scripts/`: moduli Python riutilizzabili per la gestione dell'intera pipeline. In particolare:
  - `loading_data.py` carica e prepara i dataset, anche in formato HDF5;
  - `data_generators.py` contiene generatori e procedure di data augmentation;
  - `backbone.py` definisce i backbone CNN utilizzabili, tra cui ResNet, VGG, EfficientNet, Inception, MobileNet e ConvNeXt;
  - `finetuning.py`, `train_final_layers.py` e gli script `optimizer_*.py` gestiscono fine-tuning, addestramento e ottimizzazione dei classificatori;
  - `losses.py` raccoglie le funzioni di loss, mentre gli script `inspect_*`, `compare_*`, `summarize_*` e `visualize_*` servono per controlli, analisi e visualizzazione dei risultati;
  - `merge_*`, `export_*`, `extract_*`, `filter_*` e `pack_*` gestiscono conversione, filtraggio e fusione dei dataset e delle immagini generate.
- `neaural_networks/`: codice e notebook dedicati alle reti generative usate per l'augmentation. `WGAN_GP_DATA_AUGMENTATION.py` implementa una WGAN-GP condizionata per le classi rare, mentre `StyleGAN2_ADA_FER.py` automatizza preparazione del dataset, training e generazione con StyleGAN2-ADA. Sono presenti anche notebook per CGAN, WGAN-GP e StyleGAN2.
- `sbatch/`: job SLURM per lanciare training, fine-tuning, preparazione dei dataset e generazione delle immagini sul cluster.
- `sbatch_meta/`: job di configurazione dell'ambiente e template per esecuzioni CPU/GPU.
- `Immagini/`: esempi e griglie di immagini sintetiche prodotte dalle GAN, utili per un controllo qualitativo dei risultati.
- `CONFIDENCE_MODEL.ipynb`: notebook per l'analisi della confidence dei modelli.

In generale, i notebook e gli script in `neaural_networks/` producono immagini sintetiche; gli strumenti in `scripts/` le integrano nei dataset e addestrano i classificatori; infine i file in `sbatch/` permettono di automatizzare gli stessi passaggi sul cluster.
