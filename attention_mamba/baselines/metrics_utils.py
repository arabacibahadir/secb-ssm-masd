"""
Common metrics helpers for baselines baselines.
Computes accuracy, balanced accuracy, macro-F1, per-class precision/recall, confusion matrix, optional ROC-AUC.
Also aggregates metrics across seeds.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    classes = sorted(np.unique(y_true))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    prec, rec, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    per_class = {}
    for idx, c in enumerate(classes):
        name = class_names[idx] if class_names and idx < len(class_names) else str(c)
        per_class[name] = {
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "f1": float(f1_per_class[idx]),
        }
    metrics["per_class"] = per_class
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    if y_proba is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=classes)
            metrics["roc_auc_ovr"] = float(roc_auc_score(y_true_bin, y_proba, multi_class="ovr"))
        except Exception as exc:  # pragma: no cover - optional
            metrics["roc_auc_ovr_error"] = str(exc)
    return metrics


def aggregate_seed_metrics(seed_metrics: List[Dict[str, float]], exclude_keys: Iterable[str] = ("per_class", "confusion_matrix")) -> Dict:
    if not seed_metrics:
        return {}
    keys = [k for k in seed_metrics[0].keys() if k not in exclude_keys]
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = np.array([m[key] for m in seed_metrics], dtype=float)
        summary[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
        }
    return summary
