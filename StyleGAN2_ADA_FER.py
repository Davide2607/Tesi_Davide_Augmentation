#!/usr/bin/env python
"""Script orchestratore StyleGAN2-ADA (FER) senza magic Colab.

Funzioni chiave:
- Clona la repo ufficiale NVLabs se manca.
- Scarica il modello pre-addestrato FFHQ.
- (Opzionale) prepara il dataset in formato .zip con dataset_tool.py.
- Lancia train.py e generate.py con parametri configurabili.

Esempi:
  python StyleGAN2_ADA_FER.py \
  --data-zip /path/DISGUST.zip \
  --outdir training-runs \
  --gpus 1 \
  --kimg 300 \
  --seeds 0-999 \
  --generate-out generated
"""

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


REPO_URL = "https://github.com/NVlabs/stylegan2-ada-pytorch.git"
# List of mirrors in order; first valid one wins
PRETRAINED_URLS = [
  "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
  "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/ffhq-512-avg-tpurun1.pkl",
]
REPO_DIR = Path("stylegan2-ada-pytorch")


def run(cmd, cwd=None):
  print(f"\n[cmd] {' '.join(cmd)} (cwd={cwd or os.getcwd()})")
  subprocess.run(cmd, cwd=cwd, check=True)


def ensure_repo():
  if REPO_DIR.exists():
    print("Repo gia' presente, skip clone")
    return
  run(["git", "clone", REPO_URL])


def ensure_requirements():
  req = REPO_DIR / "requirements.txt"
  if req.exists():
    run([sys.executable, "-m", "pip", "install", "-r", str(req)])
  else:
    print("requirements.txt non trovato, continuo senza")


def download_pretrained(dest: Path):
  if dest.exists():
    print(f"Pretrained gia' presente: {dest}")
    return
  dest.parent.mkdir(parents=True, exist_ok=True)

  last_err = None
  for url in PRETRAINED_URLS:
    try:
      print(f"Scarico pretrained da {url} -> {dest}")
      urllib.request.urlretrieve(url, dest)
      print("Download OK")
      return
    except Exception as e:  # pragma: no cover - interactive download
      print(f"Download fallito da {url}: {e}")
      last_err = e
  raise last_err if last_err else RuntimeError("Download pretrained fallito")


def maybe_prepare_dataset(source_dir: Path, dest_zip: Path):
  if dest_zip.exists():
    print(f"Dataset zip gia' presente: {dest_zip}")
    return
  if not source_dir.exists():
    raise FileNotFoundError(f"Directory sorgente mancante: {source_dir}")
  run([sys.executable, "dataset_tool.py", "--source", str(source_dir), "--dest", str(dest_zip)], cwd=REPO_DIR)


def train(data_zip: Path, outdir: Path, gpus: int, kimg: int, resume: Path | None, aug: str, cfg: str):
  cmd = [
    sys.executable,
    "train.py",
    f"--data={data_zip}",
    f"--outdir={outdir}",
    f"--gpus={gpus}",
    f"--kimg={kimg}",
    f"--aug={aug}",
    f"--cfg={cfg}",
  ]
  if resume:
    cmd.append(f"--resume={resume}")
  run(cmd, cwd=REPO_DIR)


def generate(network_pkl: Path, seeds: str, outdir: Path):
  cmd = [
    sys.executable,
    "generate.py",
    f"--network={network_pkl}",
    f"--seeds={seeds}",
    f"--outdir={outdir}",
  ]
  run(cmd, cwd=REPO_DIR)


def parse_args():
  p = argparse.ArgumentParser(description="Orchestratore StyleGAN2-ADA FER")
  p.add_argument("--data-zip", type=Path, required=True, help="Zip del dataset (una classe) prodotto da dataset_tool.py")
  p.add_argument("--source-dir", type=Path, help="Directory immagini per creare lo zip se manca")
  p.add_argument("--outdir", type=Path, default=Path("training-runs"), help="Cartella output training")
  p.add_argument("--gpus", type=int, default=1, help="Numero GPU")
  p.add_argument("--kimg", type=int, default=300, help="Kimg di training")
  p.add_argument("--aug", type=str, default="ada", help="Augmenter (es. ada)")
  p.add_argument("--cfg", type=str, default="auto", help="Config (auto, stylegan2, etc.)")
  p.add_argument("--resume", type=Path, default=None, help="Checkpoint da cui riprendere (default: ffhq)")
  p.add_argument("--seeds", type=str, default="0-999", help="Intervallo seeds per generate.py")
  p.add_argument("--generate-out", type=Path, default=Path("generated"), help="Cartella output immagini generate")
  return p.parse_args()


def main():
  args = parse_args()

  ensure_repo()
  ensure_requirements()

  pretrained_path = REPO_DIR / "ffhq.pkl"
  download_pretrained(pretrained_path)

  if not args.data_zip.exists():
    if not args.source_dir:
      raise FileNotFoundError("--data-zip mancante e --source-dir non fornito")
    maybe_prepare_dataset(args.source_dir, args.data_zip)

  resume_path = args.resume if args.resume else pretrained_path
  train(args.data_zip, args.outdir, args.gpus, args.kimg, resume_path, args.aug, args.cfg)

  # Trova l'ultimo snapshot se non specificato
  snapshots = sorted((args.outdir).glob("**/network-snapshot-*.pkl"))
  if not snapshots:
    raise FileNotFoundError("Nessun snapshot trovato dopo il training")
  latest = snapshots[-1]
  print(f"Uso snapshot: {latest}")

  args.generate_out.mkdir(parents=True, exist_ok=True)
  generate(latest, args.seeds, args.generate_out)

  print("\nCompletato. Filtra le immagini generate con il classifier FER ad alta confidenza e uniscile al dataset.")


if __name__ == "__main__":
  main()
