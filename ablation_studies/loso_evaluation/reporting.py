from __future__ import annotations

import json
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import CLASS_NAMES, compute_metrics


SUMMARY_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr")


def is_run_complete(run_dir: Path) -> bool:
    result_path = Path(run_dir) / "result.json"
    if not result_path.exists():
        return False
    try:
        return json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def aggregate_completed_runs(results_dir: Path, required_subjects: list[str]) -> dict:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for result_path in Path(results_dir).rglob("result.json"):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        predictions_path = Path(payload["predictions"])
        if not predictions_path.is_absolute():
            predictions_path = result_path.parent / predictions_path
        grouped[int(payload["seed"])].append(
            {
                "subject": payload["test_subject"],
                "predictions_path": predictions_path,
            }
        )

    per_seed: list[dict] = []
    required = set(required_subjects)
    for seed in sorted(grouped):
        entries = grouped[seed]
        by_subject = {entry["subject"]: entry for entry in entries}
        if not required.issubset(by_subject):
            continue
        y_true_parts = []
        y_pred_parts = []
        y_proba_parts = []
        for subject in required_subjects:
            with np.load(by_subject[subject]["predictions_path"]) as predictions:
                y_true_parts.append(predictions["y_true"])
                y_pred_parts.append(predictions["y_pred"])
                y_proba_parts.append(predictions["y_proba"])
        metrics = compute_metrics(
            np.concatenate(y_true_parts),
            np.concatenate(y_pred_parts),
            np.concatenate(y_proba_parts),
        )
        per_seed.append({"seed": seed, "metrics": metrics})

    mean_std: dict[str, dict[str, float]] = {}
    for metric_name in SUMMARY_METRICS:
        values = np.asarray(
            [item["metrics"][metric_name] for item in per_seed],
            dtype=np.float64,
        )
        if values.size:
            mean_std[metric_name] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=0)),
            }
    return {
        "required_subjects": required_subjects,
        "complete_seeds": [item["seed"] for item in per_seed],
        "per_seed": per_seed,
        "mean_std": mean_std,
    }


def write_summary_outputs(summary: dict, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    markdown_path = output_dir / "loso_results.md"
    confusion_path = output_dir / "confusion_matrix.png"

    json_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", *SUMMARY_METRICS])
        for item in summary["per_seed"]:
            writer.writerow(
                [f"seed_{item['seed']}", *[item["metrics"][name] for name in SUMMARY_METRICS]]
            )
        writer.writerow(
            ["mean", *[summary["mean_std"].get(name, {}).get("mean", "") for name in SUMMARY_METRICS]]
        )
        writer.writerow(
            ["std", *[summary["mean_std"].get(name, {}).get("std", "") for name in SUMMARY_METRICS]]
        )

    lines = [
        "# Subject-Independent LOSO Results",
        "",
        f"Complete seeds: {', '.join(map(str, summary['complete_seeds'])) or 'none'}",
        f"Required held-out subjects: {', '.join(summary['required_subjects'])}",
        "",
        "| Metric | Mean | SD |",
        "|---|---:|---:|",
    ]
    for metric_name in SUMMARY_METRICS:
        values = summary["mean_std"].get(metric_name)
        if values:
            lines.append(f"| {metric_name} | {values['mean']:.4f} | {values['std']:.4f} |")
    lines.extend(
        [
            "",
            "Each seed metric is computed after concatenating predictions from all required held-out subjects.",
            "The subject-to-record mapping is an explicit dataset assumption documented in `subject_map.csv`.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    confusion = np.zeros((3, 3), dtype=np.int64)
    for item in summary["per_seed"]:
        confusion += np.asarray(item["metrics"]["confusion_matrix"], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, str(confusion[row, column]), ha="center", va="center")
    axis.set_xticks(range(3), CLASS_NAMES, rotation=30, ha="right")
    axis.set_yticks(range(3), CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Pooled LOSO Confusion Matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(confusion_path, dpi=200)
    plt.close(figure)
    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": markdown_path,
        "confusion_matrix": confusion_path,
    }
