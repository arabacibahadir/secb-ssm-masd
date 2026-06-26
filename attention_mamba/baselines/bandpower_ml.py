"""
Band-power classical baselines (SVM / RF / XGBoost) on the same EEG epochs.
Uses identical preprocessing/windowing as attention_mamba/train.py.
Outputs per-seed metrics, macro/balanced metrics, per-class scores, confusion matrices, and aggregate mean/std.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data_utils import build_dataset, set_seed, split_arrays
from .metrics_utils import aggregate_seed_metrics, compute_metrics


def compute_bandpower_features(signals: np.ndarray, hz: int = 128) -> np.ndarray:
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30)]  # delta/theta/alpha/beta
    feats: List[List[float]] = []
    for ep in signals:  # ep: [T, C]
        ch_feats: List[float] = []
        for c in range(ep.shape[1]):
            f, psd = welch(ep[:, c], fs=hz, nperseg=min(256, len(ep)))
            for low, high in bands:
                mask = (f >= low) & (f < high)
                ch_feats.append(float(psd[mask].mean()))
        feats.append(ch_feats)
    return np.asarray(feats, dtype=np.float32)


def build_models() -> Dict[str, object]:
    models: Dict[str, object] = {
        "svm_rbf": make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True)),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=0
        ),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"XGBoost not available ({exc}); skipping xgboost baseline.")
    return models


def evaluate_models(
    features: np.ndarray,
    labels: np.ndarray,
    seeds: List[int],
    verbose: bool = True,
) -> Dict:
    models = build_models()
    results: Dict[str, Dict] = {}
    for name, model in models.items():
        if verbose:
            print(f"Running {name}...")
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
            if verbose:
                print(
                    f"  seed {seed}: val macro_f1={val_metrics['macro_f1']:.4f}, "
                    f"val bal_acc={val_metrics['balanced_accuracy']:.4f} | "
                    f"test macro_f1={te_metrics['macro_f1']:.4f}, "
                    f"test bal_acc={te_metrics['balanced_accuracy']:.4f}"
                )

        summary = aggregate_seed_metrics([s["test"] for s in per_seed])
        results[name] = {"seeds": per_seed, "test_summary_mean_std": summary}
        if verbose:
            print(
                f"  {name} summary: macro_f1 {summary['macro_f1']['mean']:.4f}+/-{summary['macro_f1']['std']:.4f}, "
                f"bal_acc {summary['balanced_accuracy']['mean']:.4f}+/-{summary['balanced_accuracy']['std']:.4f}"
            )
    return results


def parse_args():
    ap = argparse.ArgumentParser(description="Band-power classical ML baselines (SVM/RF/XGBoost).")
    ap.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "EEGData")
    ap.add_argument("--epoch-length", type=float, default=2.0)
    ap.add_argument("--step-size", type=float, default=0.125)
    ap.add_argument("--hz", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--save-dir", type=Path, default=Path(__file__).resolve().parents[1] / "experiments" / "baselines")
    ap.add_argument("--quiet", action="store_true", help="Disable per-seed logging.")
    return ap.parse_args()


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    signals, labels = build_dataset(args.data_root, epoch_length=args.epoch_length, step_size=args.step_size)
    if not args.quiet:
        print(f"Loaded dataset: signals {signals.shape}, labels {labels.shape}")
    features = compute_bandpower_features(signals, hz=args.hz)
    if not args.quiet:
        print(f"Computed bandpower features: {features.shape}")
    results = evaluate_models(features, labels, seeds=args.seeds, verbose=not args.quiet)

    out_path = args.save_dir / "bandpower_ml.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "data_root": str(args.data_root),
                    "epoch_length": args.epoch_length,
                    "step_size": args.step_size,
                    "hz": args.hz,
                    "seeds": args.seeds,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved band-power ML results to {out_path}")


if __name__ == "__main__":
    main()
