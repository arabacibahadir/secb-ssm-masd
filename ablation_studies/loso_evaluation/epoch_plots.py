from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HISTORY_COLUMNS = (
    "seed",
    "test_subject",
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_accuracy",
    "validation_balanced_accuracy",
    "validation_macro_f1",
    "learning_rate",
)


def _run_metadata(history_path: Path) -> tuple[int, str]:
    subject = history_path.parent.name.replace("test_", "")
    seed_match = re.search(r"seed_(\d+)", history_path.parent.parent.name)
    if not seed_match:
        raise ValueError(f"Cannot infer seed from {history_path}")
    return int(seed_match.group(1)), subject


def collect_histories(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    for history_path in sorted(Path(results_root).rglob("history.json")):
        seed, subject = _run_metadata(history_path)
        history = json.loads(history_path.read_text(encoding="utf-8"))
        for item in history:
            rows.append(
                {
                    "seed": seed,
                    "test_subject": subject,
                    **{column: item.get(column) for column in HISTORY_COLUMNS[2:]},
                }
            )
    return rows


def write_epoch_table(rows: list[dict], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in HISTORY_COLUMNS})
    return output_path


def _plot_metric(rows: list[dict], metric: str, output_path: Path, ylabel: str) -> Path:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["test_subject"])].append(row)

    figure, axis = plt.subplots(figsize=(8, 5))
    for subject, subject_rows in sorted(grouped.items()):
        subject_rows = sorted(subject_rows, key=lambda item: int(item["epoch"]))
        axis.plot(
            [int(item["epoch"]) for item in subject_rows],
            [float(item[metric]) for item in subject_rows],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=subject,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(f"LOSO validation {ylabel} by held-out subject")
    axis.grid(True, alpha=0.25)
    axis.legend(title="Test subject")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def _plot_mean_metric(rows: list[dict], metric: str, output_path: Path, ylabel: str) -> Path:
    per_epoch: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        per_epoch[int(row["epoch"])].append(float(row[metric]))
    epochs = sorted(per_epoch)
    means = np.asarray([np.mean(per_epoch[epoch]) for epoch in epochs])
    stds = np.asarray([np.std(per_epoch[epoch], ddof=0) for epoch in epochs])

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, means, color="#1f77b4", linewidth=2, label="Mean")
    axis.fill_between(epochs, means - stds, means + stds, color="#1f77b4", alpha=0.18, label="± SD")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(f"LOSO validation {ylabel}: mean across folds")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def write_epoch_plots(results_root: Path, output_dir: Path | None = None) -> dict[str, Path]:
    rows = collect_histories(results_root)
    if not rows:
        raise ValueError(f"No history.json files found under {results_root}")
    output_dir = Path(output_dir) if output_dir else Path(results_root) / "epoch_plots"
    outputs = {
        "table": write_epoch_table(rows, output_dir / "epoch_metrics.csv"),
        "validation_macro_f1_by_fold": _plot_metric(
            rows,
            "validation_macro_f1",
            output_dir / "validation_macro_f1_by_fold.png",
            "Macro-F1",
        ),
        "validation_accuracy_by_fold": _plot_metric(
            rows,
            "validation_accuracy",
            output_dir / "validation_accuracy_by_fold.png",
            "Accuracy",
        ),
        "loss_by_fold": _plot_metric(
            rows,
            "validation_loss",
            output_dir / "validation_loss_by_fold.png",
            "Loss",
        ),
        "validation_macro_f1_mean": _plot_mean_metric(
            rows,
            "validation_macro_f1",
            output_dir / "validation_macro_f1_mean.png",
            "Macro-F1",
        ),
    }
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create epoch-level LOSO plots from history.json files.")
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    outputs = write_epoch_plots(args.results_root, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
