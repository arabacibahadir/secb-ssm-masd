from __future__ import annotations

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


CLASS_NAMES = ["Focused", "Unfocused", "Drowsy"]
CLASSES = np.array([0, 1, 2])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(per_class_f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(CLASS_NAMES)
    }
    y_true_binary = label_binarize(y_true, classes=CLASSES)
    try:
        auc = float(roc_auc_score(y_true_binary, y_proba, average="macro", multi_class="ovr"))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc_ovr": auc,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }
