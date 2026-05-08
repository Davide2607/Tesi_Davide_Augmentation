import json
import textwrap
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer


NB_PATH = Path(r"c:\Users\david\Desktop\Polito\Tesi_Davide\CONFIDENCE_MODEL.ipynb")
OUT_PDF = Path(r"c:\Users\david\Desktop\Polito\Tesi_Davide\CONFIDENCE_MODEL_explained.pdf")


def _first_nonempty_lines(lines: list[str], max_lines: int = 8) -> list[str]:
    out: list[str] = []
    for line in lines:
        if line.strip() == "":
            continue
        out.append(line.rstrip("\n"))
        if len(out) >= max_lines:
            break
    return out


def _classify_cell(src_text: str) -> str:
    s = src_text
    if "google.colab" in s or "drive.mount" in s or "%cd" in s:
        return "Setup Colab / Drive"
    if "!pip install" in s or "pip install" in s:
        return "Installazione dipendenze"
    if "categorical_focal_loss" in s:
        return "Loss (Focal) + caricamento modello"
    if "load_model" in s and "ConvNeXt" in s:
        return "Caricamento modello ConvNeXt"
    if "class CustomBalancedDataGenerator" in s:
        return "Data generator (bilanciato + label smoothing + paths)"
    if "h5py" in s and "load_data_and_labels" in s:
        return "Caricamento dataset da H5"
    if "def evaluate_keras_model" in s:
        return "Valutazione modello (predict + analisi errori)"
    if "to_csv" in s and "model_results" in s:
        return "Export CSV risultati"
    if "pd.read_csv" in s and "model_results.csv" in s:
        return "Analisi downstream da CSV"
    if "agreement" in s.lower() or "IOA" in s:
        return "Analisi agreement / IOA"
    if "matplotlib" in s or "plt." in s:
        return "Plot / visualizzazione"
    if "keras_models" in s and "ConvNeXt" in s:
        return "Selezione modelli (ConvNeXt-only)"
    return "Altro"


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    styles = getSampleStyleSheet()
    title = styles["Title"]
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]

    mono = ParagraphStyle(
        name="Mono",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.5,
        leading=10,
        spaceBefore=6,
        spaceAfter=6,
    )

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title="CONFIDENCE_MODEL.ipynb – Spiegazione",
    )

    story: list = []
    story.append(Paragraph("CONFIDENCE_MODEL.ipynb – Spiegazione dettagliata", title))
    story.append(Paragraph(f"Generato: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body))
    story.append(Paragraph(f"Notebook: {NB_PATH}", body))
    story.append(Spacer(1, 12))

    # High-level summary based on detected paths
    all_text = "\n".join("".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code")

    def _extract_first(pattern: str) -> str | None:
        import re

        m = re.search(pattern, all_text)
        return m.group(1) if m else None

    convnext_path = _extract_first(r"convnext_path\s*=\s*['\"]([^'\"]+)['\"]")
    trainval_path = _extract_first(r"file_path\s*=\s*['\"]([^'\"]+)['\"]")
    test_h5_path = _extract_first(r"test_path\s*=\s*['\"]([^'\"]+)['\"]")

    story.append(Paragraph("Obiettivo del notebook", h1))
    story.append(
        Paragraph(
            "Il notebook carica un modello Keras (ConvNeXt) e un dataset (H5), esegue predizioni sul set di test, "
            "calcola confidence/errore ad alta confidenza, esporta un CSV con le probabilità per immagine e svolge analisi/plot downstream (uncertainty, agreement/IOA).",
            body,
        )
    )

    story.append(Paragraph("Input principali (path)", h1))
    if convnext_path:
        story.append(Paragraph(f"- Modello ConvNeXt: <b>{convnext_path}</b>", body))
    else:
        story.append(Paragraph("- Modello ConvNeXt: variabile <b>convnext_path</b> (non rilevata automaticamente)", body))

    if trainval_path:
        story.append(Paragraph(f"- Dataset train/val (cartella): <b>{trainval_path}</b> (usa dataset.h5)", body))
    if test_h5_path:
        story.append(Paragraph(f"- Dataset test (H5): <b>{test_h5_path}</b>", body))

    story.append(Spacer(1, 12))

    story.append(Paragraph("Flusso logico (in breve)", h1))
    bullets = [
        "Setup ambiente (Colab/Drive + install dipendenze) se presente.",
        "Definizione di layer custom e loss (focal loss) per compatibilità nel load del modello.",
        "Caricamento ConvNeXt (tf.keras.models.load_model) con custom_object_scope.",
        "Caricamento dataset da file H5: X_train/X_val/X_test + label e (opzionalmente) paths delle immagini nel test.",
        "Creazione data generator bilanciato (con label smoothing) e generatore test con paths_data.",
        "Valutazione: predict → y_pred/y_true → errori ad alta confidenza + salvataggio grid immagini.",
        "Salvataggio risultati in CSV: per ogni immagine salva prob. per classe.",
        "Analisi downstream da CSV (uncertainty, agreement/IOA, plot).",
    ]
    story.append(Paragraph("<br/>".join(f"• {b}" for b in bullets), body))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Dettaglio per cella", h1))

    for idx, cell in enumerate(cells, start=1):
        ctype = cell.get("cell_type")
        lang = cell.get("metadata", {}).get("language")
        if ctype == "markdown":
            src_lines = cell.get("source", [])
            title_line = src_lines[0] if src_lines else "(markdown)"
            story.append(Paragraph(f"Cella {idx} – Markdown", h2))
            story.append(Paragraph(title_line, body))
            story.append(Spacer(1, 6))
            continue

        if ctype != "code":
            continue

        src_lines = cell.get("source", [])
        src_text = "".join(src_lines)
        category = _classify_cell(src_text)
        story.append(Paragraph(f"Cella {idx} – {category}", h2))

        # Plain-language explanation based on category
        expl = {
            "Setup Colab / Drive": "Monta Google Drive e cambia directory per accedere ai file su /content/drive. In esecuzione locale queste righe non sono utilizzabili.",
            "Installazione dipendenze": "Installa TensorFlow (tipicamente solo su Colab). In locale va fatto nel tuo ambiente/venv una volta sola.",
            "Loss (Focal) + caricamento modello": "Definisce la focal loss e carica il modello ConvNeXt da convnext_path usando custom_object_scope (necessario perché il modello usa oggetti custom).",
            "Caricamento modello ConvNeXt": "Carica il SavedModel/.keras/.h5 del ConvNeXt. Se il path è su Drive, richiede mount; in locale serve un path Windows valido.",
            "Data generator (bilanciato + label smoothing + paths)": "Definisce un generator Keras Sequence che bilancia le classi per batch e applica label smoothing; per il test può mantenere anche i path originali delle immagini (paths_data).",
            "Caricamento dataset da H5": "Legge dataset da file H5 con h5py: per train legge X_train/y_train e X_val/y_val; per test legge X_test/y_test e (se esiste) il dataset 'paths'.",
            "Valutazione modello (predict + analisi errori)": "Esegue model.predict sul generator test, calcola y_pred/y_true, identifica errori ad alta confidenza e salva figure riassuntive.",
            "Export CSV risultati": "Costruisce un DataFrame con (modello, immagine, true/pred, probabilità) e salva un CSV su Drive.",
            "Analisi downstream da CSV": "Ricarica i CSV prodotti e fa analisi/plot su subset di immagini, incertezze, agreement.",
            "Analisi agreement / IOA": "Costruisce metriche/plot di agreement tra predizioni e annotazioni (e salva immagini/CSV di supporto).",
            "Plot / visualizzazione": "Celle dedicate a plot e salvataggio di immagini/figure.",
            "Selezione modelli (ConvNeXt-only)": "Definisce la mappa dei modelli da valutare. Nel tuo caso è ridotta a ConvNeXt.",
            "Altro": "Cella di supporto (utility, variabili, trasformazioni o analisi specifica).",
        }.get(category, "")
        if expl:
            story.append(Paragraph(expl, body))

        # Include a short code excerpt
        excerpt = _first_nonempty_lines(src_lines, max_lines=10)
        if excerpt:
            story.append(Paragraph("Estratto (prime righe non vuote):", body))
            story.append(Preformatted("\n".join(excerpt), mono))

        story.append(Spacer(1, 8))

    story.append(Paragraph("Note pratiche (locale vs Colab)", h1))
    story.append(
        Paragraph(
            "Il notebook è scritto in stile Colab (drive.mount, path /content/drive, !pip). "
            "Per eseguirlo in locale su VS Code devi: (1) installare TensorFlow nel tuo ambiente, "
            "(2) cambiare i path a path locali, (3) evitare drive.mount/!pip. "
            "Le celle di predict possono risultare lente o difficili da interrompere perché TensorFlow esegue molto lavoro in runtime nativo.",
            body,
        )
    )

    doc.build(story)
    print("Wrote:", OUT_PDF)


if __name__ == "__main__":
    main()
