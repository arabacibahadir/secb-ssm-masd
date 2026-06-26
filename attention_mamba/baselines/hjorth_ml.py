"""
Hjorth parameter baselines (LogReg / Linear SVM) on identical EEG epochs.
Outputs per-seed metrics, per-class scores, confusion matrices, and mean/std over seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data_utils import build_dataset, set_seed, split_arrays
from .metrics_utils import aggregate_seed_metrics, compute_metrics


def compute_hjorth_features(signals: np.ndarray) -> np.ndarray:
    feats: List[List[float]] = []
    for ep in signals:  # [T, C]
        ch_feats: List[float] = []
        diff = np.diff(ep, axis=0)
        diff2 = np.diff(diff, axis=0)
        for c in range(ep.shape[1]):
            x = ep[:, c]
            dx = diff[:, c]
            ddx = diff2[:, c]
            var_x = np.var(x)
            var_dx = np.var(dx)
            var_ddx = np.var(ddx)
            activity = var_x
            mobility = np.sqrt(var_dx / (var_x + 1e-8))
            complexity = np.sqrt(var_ddx / (var_dx + 1e-8)) / (mobility + 1e-8)
            ch_feats.extend([activity, mobility, complexity])
        feats.append(ch_feats)
    return np.asarray(feats, dtype=np.float32)


def build_models() -> Dict[str, object]:
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="auto"),
        ),
        "svm_linear": make_pipeline(
            StandardScaler(), SVC(kernel="linear", class_weight="balanced", probability=True)
        ),
    }


def evaluate_models(features: np.ndarray, labels: np.ndarray, seeds: List[int]) -> Dict:
    models = build_models()
    results: Dict[str, Dict] = {}
    for name, model in models.items():
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            splits = split_arrays(features, labels, seed=seed)
            x_train, y_train = splits["train_x"], splits["train_y"]
            x_val, y_val = splits["val_x"], splits["val_y"]
            x_test, y_test = splits["test_x"], splits["test_y"]

            m = model
            if hasattr(m, "random_state"):
                setattr(m, "random_state", seed)
            m.fit(x_train, y_train)

            val_pred = m.predict(x_val)
            te_pred = m.predict(x_test)
            te_proba = m.predict_proba(x_test) if hasattr(m, "predict_proba") else None

            val_metrics = compute_metrics(y_val, val_pred)
            te_metrics = compute_metrics(y_test, te_pred, te_proba)
            per_seed.append({"seed": seed, "val": val_metrics, "test": te_metrics})
        summary = aggregate_seed_metrics([s["test"] for s in per_seed])
        results[name] = {"seeds": per_seed, "test_summary_mean_std": summary}
    return results


def parse_args():
    ap = argparse.ArgumentParser(description="Hjorth parameter baselines (LogReg / Linear SVM).")
    ap.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "EEGData")
    ap.add_argument("--epoch-length", type=float, default=2.0)
    ap.add_argument("--step-size", type=float, default=0.125)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--save-dir", type=Path, default=Path(__file__).resolve().parents[1] / "experiments" / "baselines")
    return ap.parse_args()


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    signals, labels = build_dataset(args.data_root, epoch_length=args.epoch_length, step_size=args.step_size)
    features = compute_hjorth_features(signals)
    results = evaluate_models(features, labels, seeds=args.seeds)

    out_path = args.save_dir / "hjorth_ml.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "data_root": str(args.data_root),
                    "epoch_length": args.epoch_length,
                    "step_size": args.step_size,
                    "seeds": args.seeds,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved Hjorth ML results to {out_path}")


if __name__ == "__main__":
    main()
