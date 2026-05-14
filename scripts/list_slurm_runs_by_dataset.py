import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DATASET_RE = re.compile(r"\[INFO\]\s+DATASET_H5=(?P<path>\S+)")
MODEL_DIR_RE = re.compile(r"\[INFO\]\s+MODEL_DIR=(?P<path>\S+)")
BEST_RE = re.compile(r"\[BEST\]\[(?P<model>[^\]]+)\]\[(?P<stage>[^\]]+)\]\s+(?P<body>.+)")
METRICS_RE = re.compile(r"\[METRICS\]\[(?P<stage>[^\]]+)\]\s+(?P<body>.+)")


@dataclass
class RunInfo:
    log_path: Path
    dataset_h5: Optional[str] = None
    model_dir: Optional[str] = None
    best_line: Optional[str] = None
    metrics_line: Optional[str] = None


def is_gan_dataset(dataset_path: str) -> bool:
    """Heuristic classifier based on dataset filename/path.

    Adjust keywords to match your naming conventions.
    """
    p = dataset_path.lower()
    gan_keywords = [
        "gan",
        "stylegan",
        "wgan",
        "cgan",
        "synthetic",
        "augmented",
        "dataset_aug",
        "unito_con_gan",
    ]
    return any(k in p for k in gan_keywords)


def detect_stage_from_filename(name: str) -> str:
    n = name.lower()
    if "opt_final" in n:
        return "optimizer_final"
    if "opt_ft" in n:
        return "optimizer_finetuning"
    if "_final_" in n:
        return "final_layers"
    if "_ft_" in n:
        return "finetuning"
    return "unknown"


def parse_log(log_path: Path) -> RunInfo:
    info = RunInfo(log_path=log_path)
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return info

    for line in text:
        if info.dataset_h5 is None:
            m = DATASET_RE.search(line)
            if m:
                info.dataset_h5 = m.group("path")
        if info.model_dir is None:
            m = MODEL_DIR_RE.search(line)
            if m:
                info.model_dir = m.group("path")
        m = BEST_RE.search(line)
        if m:
            info.best_line = line.strip()
        m = METRICS_RE.search(line)
        if m:
            # Keep the last metrics line (usually final metrics at end)
            info.metrics_line = line.strip()

    return info


def iter_logs(logs_dir: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(logs_dir.glob(pattern))


def fmt(value: Optional[str]) -> str:
    return value if value else "-"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "List Slurm .out logs and show which DATASET_H5 each run used. "
            "Helps distinguish GAN/augmented vs non-augmented experiments."
        )
    )
    ap.add_argument(
        "--logs-dir",
        type=str,
        default="/home/dravida/models/FER-Augmentation/out_err",
        help="Directory containing Slurm .out logs",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="convnext_*.out",
        help="Glob pattern for log files (relative to logs-dir)",
    )
    ap.add_argument(
        "--only",
        choices=["gan", "no-gan", "all"],
        default="all",
        help="Filter by GAN heuristic",
    )
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        raise FileNotFoundError(f"logs-dir not found: {logs_dir}")

    logs = list(iter_logs(logs_dir, args.pattern))
    if not logs:
        print(f"No logs found in {logs_dir} matching {args.pattern}")
        return

    print("log\tstage\tgan\tdataset_h5\tmodel_dir\tmetrics\tbest")
    for log_path in logs:
        info = parse_log(log_path)
        dataset = info.dataset_h5
        gan = is_gan_dataset(dataset) if dataset else False
        if args.only == "gan" and not gan:
            continue
        if args.only == "no-gan" and gan:
            continue

        stage = detect_stage_from_filename(log_path.name)
        gan_txt = "GAN" if gan else "NO_GAN"
        print(
            "\t".join(
                [
                    log_path.name,
                    stage,
                    gan_txt,
                    fmt(dataset),
                    fmt(info.model_dir),
                    fmt(info.metrics_line),
                    fmt(info.best_line),
                ]
            )
        )


if __name__ == "__main__":
    main()
