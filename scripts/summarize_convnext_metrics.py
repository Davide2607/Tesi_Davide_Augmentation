import argparse
import re
from pathlib import Path


FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

EPOCH_METRICS_RE = re.compile(
    rf"loss:\s*({FLOAT_RE})\s*-\s*categorical_accuracy:\s*({FLOAT_RE})\s*-\s*val_loss:\s*({FLOAT_RE})\s*-\s*val_categorical_accuracy:\s*({FLOAT_RE})"
)
EVAL_METRICS_RE = re.compile(
    rf"loss:\s*({FLOAT_RE})\s*-\s*categorical_accuracy:\s*({FLOAT_RE})"
)
EXTRA_METRICS_LINE_RE = re.compile(r"\[METRICS\]\[(?P<stage>[^\]]+)\]\s+(?P<body>.+)$")
KEY_VALUE_RE = re.compile(rf"([a-zA-Z0-9_]+)=({FLOAT_RE})")


def parse_log_file(log_path: Path):
    epoch_rows = []
    eval_rows = []
    extra_metrics = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            epoch_match = EPOCH_METRICS_RE.search(line)
            if epoch_match:
                try:
                    epoch_rows.append(
                        {
                            "train_loss": float(epoch_match.group(1)),
                            "train_acc": float(epoch_match.group(2)),
                            "val_loss": float(epoch_match.group(3)),
                            "val_acc": float(epoch_match.group(4)),
                        }
                    )
                except ValueError:
                    pass

            eval_match = EVAL_METRICS_RE.search(line)
            if eval_match:
                try:
                    eval_rows.append(
                        {
                            "loss": float(eval_match.group(1)),
                            "acc": float(eval_match.group(2)),
                        }
                    )
                except ValueError:
                    pass

            extra_match = EXTRA_METRICS_LINE_RE.search(line)
            if extra_match:
                stage = extra_match.group("stage")
                body = extra_match.group("body")
                metrics = {k: float(v) for k, v in KEY_VALUE_RE.findall(body)}
                extra_metrics[stage] = metrics

    best_val = max(epoch_rows, key=lambda row: row["val_acc"]) if epoch_rows else None
    last_eval = eval_rows[-1] if eval_rows else None

    return {
        "epochs": len(epoch_rows),
        "best_val": best_val,
        "last_eval": last_eval,
        "extra_metrics": extra_metrics,
    }


def fmt(value):
    if value is None:
        return "-"
    return f"{value:.6f}"


def print_summary(title: str, data: dict):
    print(f"\n=== {title} ===")
    print(f"epochs_found: {data['epochs']}")

    best_val = data["best_val"]
    if best_val:
        print(
            "best_val: "
            f"val_acc={fmt(best_val['val_acc'])}, "
            f"val_loss={fmt(best_val['val_loss'])}, "
            f"train_acc={fmt(best_val['train_acc'])}, "
            f"train_loss={fmt(best_val['train_loss'])}"
        )
    else:
        print("best_val: -")

    last_eval = data["last_eval"]
    if last_eval:
        print(f"last_eval: test_acc={fmt(last_eval['acc'])}, test_loss={fmt(last_eval['loss'])}")
    else:
        print("last_eval: -")

    if data["extra_metrics"]:
        for stage, metrics in data["extra_metrics"].items():
            metrics_text = ", ".join(f"{key}={fmt(val)}" for key, val in metrics.items())
            print(f"extra_metrics[{stage}]: {metrics_text}")
    else:
        print("extra_metrics: -")


def main():
    parser = argparse.ArgumentParser(description="Summarize ConvNeXt metrics from Slurm .out logs")
    parser.add_argument("--final-log", type=str, help="Path to final-layers .out file")
    parser.add_argument("--ft-log", type=str, help="Path to finetuning .out file")
    args = parser.parse_args()

    if not args.final_log and not args.ft_log:
        raise ValueError("Passa almeno uno tra --final-log o --ft-log")

    final_data = None
    ft_data = None

    if args.final_log:
        final_log = Path(args.final_log)
        if not final_log.exists():
            raise FileNotFoundError(f"Final-layers log not found: {final_log}")
        final_data = parse_log_file(final_log)
        print_summary("FINAL LAYERS", final_data)

    if args.ft_log:
        ft_log = Path(args.ft_log)
        if not ft_log.exists():
            raise FileNotFoundError(f"Finetuning log not found: {ft_log}")
        ft_data = parse_log_file(ft_log)
        print_summary("FINE TUNING", ft_data)

    if final_data and ft_data and final_data["last_eval"] and ft_data["last_eval"]:
        delta_acc = ft_data["last_eval"]["acc"] - final_data["last_eval"]["acc"]
        delta_loss = ft_data["last_eval"]["loss"] - final_data["last_eval"]["loss"]
        print("\n=== DELTA (finetuning - final_layers) ===")
        print(f"delta_test_acc: {fmt(delta_acc)}")
        print(f"delta_test_loss: {fmt(delta_loss)}")


if __name__ == "__main__":
    main()
