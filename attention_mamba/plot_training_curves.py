"""Generate training curves from a JSONL log produced by train.py.

No experiment values are stored in this source file. Generated figures should
be written to an ignored experiment directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_records(log_path: Path) -> list[dict]:
    records = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {log_path}") from exc
    if not records:
        raise ValueError(f"No epoch records found in {log_path}")
    return records


def save_curve(records: list[dict], keys: tuple[str, ...], ylabel: str, output: Path) -> None:
    epochs = [record["epoch"] for record in records]
    plt.figure()
    plotted = False
    for key in keys:
        if all(key in record for record in records):
            plt.plot(epochs, [record[key] for record in records], label=key)
            plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot curves from an epoch_metrics.jsonl file.")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.log_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_curve(records, ("train_loss", "val_loss"), "Loss", args.output_dir / "loss_curve.png")
    save_curve(records, ("train_acc", "val_acc"), "Accuracy", args.output_dir / "accuracy_curve.png")
    save_curve(records, ("train_f1", "val_f1"), "Macro F1", args.output_dir / "f1_curve.png")
    save_curve(records, ("lr",), "Learning Rate", args.output_dir / "learning_rate_curve.png")


if __name__ == "__main__":
    main()
