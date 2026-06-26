"""
Optional Riemannian covariance + tangent space baselines (LogReg / Linear SVM).
Requires pyriemann. Skips gracefully if dependency is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .data_utils import build_dataset, set_seed, split_arrays
from .metrics_utils import aggregate_seed_metrics, compute_metrics


def build_models():
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"pyriemann not available ({exc}); skipping Riemann baselines.")
        return {}

    return {
        "riemann_logreg": make_pipeline(
            Covariances(estimator="oas"),
            TangentSpace(),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto"),
        ),
        "riemann_svm": make_pipeline(
            Covariances(estimator="oas"),
            TangentSpace(),
            StandardScaler(),
            LinearSVC(class_weight="balanced"),
        ),
    }


def evaluate_models(signals: np.ndarray, labels: np.ndarray, seeds: List[int]) -> Dict:
    models = build_models()
    results: Dict[str, Dict] = {}
    if not models:
        return results

    # pyriemann expects shape [trials, channels, samples]
    signals_c = np.transpose(signals, (0, 2, 1))

    for name, model in models.items():
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            splits = split_arrays(signals_c, labels, seed=seed)
            x_train, y_train = splits["train_x"], splits["train_y"]
            x_val, y_val = splits["val_x"], splits["val_y"]
            x_test, y_test = splits["test_x"], splits["test_y"]

            m = model
            m.fit(x_train, y_train)

            val_pred = m.predict(x_val)
            te_pred = m.predict(x_test)
            val_metrics = compute_metrics(y_val, val_pred)
            te_metrics = compute_metrics(y_test, te_pred)
            per_seed.append({"seed": seed, "val": val_metrics, "test": te_metrics})
        summary = aggregate_seed_metrics([s["test"] for s in per_seed])
        results[name] = {"seeds": per_seed, "test_summary_mean_std": summary}
    return results


def parse_args():
    ap = argparse.ArgumentParser(description="Riemannian covariance + tangent space baselines (LogReg / Linear SVM).")
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
    results = evaluate_models(signals, labels, seeds=args.seeds)
    if not results:
        print("No Riemannian baselines were run (dependency missing).")
        return

    out_path = args.save_dir / "riemann_ml.json"
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
    print(f"Saved Riemannian ML results to {out_path}")


if __name__ == "__main__":
    main()
