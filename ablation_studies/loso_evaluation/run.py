from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import get_profile
from .data import (
    build_record_cache,
    build_window_index,
    compute_channel_stats_from_cache,
    load_subject_map,
)
from .dataset import WindowDataset, limit_window_index
from .model import load_mamba_eeg_class
from .reporting import aggregate_completed_runs, is_run_complete, write_summary_outputs
from .splits import build_loso_split
from .training import evaluate_model, set_seed, train_model


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict subject-independent LOSO evaluation for SECB-SSM.")
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--data-root", type=Path, default=WORKSPACE_ROOT / "EEG Data")
    parser.add_argument("--subject-map", type=Path, default=PACKAGE_ROOT / "subject_map.csv")
    parser.add_argument("--cache-dir", type=Path, default=PACKAGE_ROOT / "cache")
    parser.add_argument("--results-dir", type=Path, default=PACKAGE_ROOT / "results")
    parser.add_argument(
        "--model-source",
        type=Path,
        default=WORKSPACE_ROOT
        / "attentioneeg-main"
        / "attentioneeg-main"
        / "attention_mamba"
        / "train.py",
    )
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--max-windows-per-split", type=int, default=None)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def save_predictions_csv_gz(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    record_ids: np.ndarray,
    starts: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "record_id",
                "start",
                "y_true",
                "y_pred",
                "prob_focused",
                "prob_unfocused",
                "prob_drowsy",
            ]
        )
        for index in range(len(y_true)):
            writer.writerow(
                [
                    int(record_ids[index]),
                    int(starts[index]),
                    int(y_true[index]),
                    int(y_pred[index]),
                    *[float(value) for value in y_proba[index]],
                ]
            )


def _loader(dataset, batch_size: int, shuffle: bool, seed: int, workers: int, pin_memory: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def _filter_index(global_index: list[dict], record_ids: list[int]) -> list[dict]:
    allowed = set(record_ids)
    return [item for item in global_index if item["record_id"] in allowed]


def _run_one(
    *,
    test_subject: str,
    seed: int,
    profile,
    args,
    subject_map: dict[int, str],
    manifest: dict,
    global_index: list[dict],
    output_root: Path,
    model_class,
) -> None:
    run_dir = output_root / f"seed_{seed}" / f"test_{test_subject}"
    if is_run_complete(run_dir) and not args.force:
        print(f"Skipping completed run: seed={seed}, test={test_subject}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    split = build_loso_split(subject_map, test_subject, args.validation_seed)
    mean, std = compute_channel_stats_from_cache(manifest, split["train_records"])

    maximum = (
        args.max_windows_per_split
        if args.max_windows_per_split is not None
        else profile.max_windows_per_split
    )
    train_index = limit_window_index(
        _filter_index(global_index, split["train_records"]),
        maximum,
        seed=seed * 100 + 1,
    )
    validation_index = limit_window_index(
        _filter_index(global_index, split["validation_records"]),
        maximum,
        seed=seed * 100 + 2,
    )
    test_index = limit_window_index(
        _filter_index(global_index, split["test_records"]),
        maximum,
        seed=seed * 100 + 3,
    )
    split_manifest = {
        **split,
        "subject_map": {str(key): value for key, value in subject_map.items()},
        "normalization": {
            "source": "training_records_only",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "window": 256,
        "stride": 240,
        "window_counts": {
            "train": len(train_index),
            "validation": len(validation_index),
            "test": len(test_index),
        },
    }
    (run_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2),
        encoding="utf-8",
    )

    train_dataset = WindowDataset(manifest, train_index, mean, std, window=256)
    validation_dataset = WindowDataset(manifest, validation_index, mean, std, window=256)
    test_dataset = WindowDataset(manifest, test_index, mean, std, window=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or profile.batch_size
    pin_memory = device.type == "cuda"
    train_loader = _loader(
        train_dataset,
        batch_size,
        shuffle=True,
        seed=seed,
        workers=args.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size,
        shuffle=False,
        seed=seed,
        workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = _loader(
        test_dataset,
        batch_size,
        shuffle=False,
        seed=seed,
        workers=args.num_workers,
        pin_memory=pin_memory,
    )

    set_seed(seed)
    model = model_class(
        input_dim=14,
        num_classes=3,
        d_model=profile.d_model,
        layers=profile.layers,
        ff_exp=4,
        conv_kernel=31,
        emb_kernel=7,
        emb_stride=2,
        dropout=0.18,
        bidir=True,
    )
    epochs = args.epochs or profile.epochs
    patience = args.patience or profile.patience
    use_amp = profile.use_amp if args.use_amp is None else args.use_amp
    start_time = time.time()
    model, history, best_epoch = train_model(
        model,
        train_loader,
        validation_loader,
        device,
        epochs=epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=patience,
        use_amp=use_amp,
        log_prefix=f"[seed={seed} test={test_subject}]",
    )
    validation = evaluate_model(model, validation_loader, device)
    test = evaluate_model(model, test_loader, device)
    elapsed_seconds = time.time() - start_time

    predictions_path = run_dir / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        y_true=test["y_true"],
        y_pred=test["y_pred"],
        y_proba=test["y_proba"],
        record_ids=test["record_ids"],
        starts=test["starts"],
    )
    save_predictions_csv_gz(
        run_dir / "predictions.csv.gz",
        test["y_true"],
        test["y_pred"],
        test["y_proba"],
        test["record_ids"],
        test["starts"],
    )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if args.save_checkpoints:
        torch.save(model.state_dict(), run_dir / "best_model.pt")

    result = {
        "status": "complete",
        "profile": profile.name,
        "seed": seed,
        "test_subject": test_subject,
        "best_epoch": best_epoch,
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "amp": bool(use_amp and device.type == "cuda"),
        "validation_metrics": validation["metrics"],
        "test_metrics": test["metrics"],
        "predictions": str(predictions_path.resolve()),
        "split_manifest": str((run_dir / "split_manifest.json").resolve()),
        "history": str((run_dir / "history.json").resolve()),
        "model": {
            "source": str(args.model_source.resolve()),
            "d_model": profile.d_model,
            "layers": profile.layers,
            "ff_expansion": 4,
            "conv_kernel": 31,
            "embedding_kernel": 7,
            "embedding_stride": 2,
            "dropout": 0.18,
            "bidirectional": True,
        },
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    train_dataset.close()
    validation_dataset.close()
    test_dataset.close()
    print(f"Completed seed={seed}, test={test_subject}: {test['metrics']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = get_profile(args.profile)
    subjects = args.subjects or profile.subjects
    seeds = args.seeds or profile.seeds
    subject_map = load_subject_map(args.subject_map)
    known_subjects = set(subject_map.values())
    unknown = set(subjects) - known_subjects
    if unknown:
        raise ValueError(f"unknown subjects: {sorted(unknown)}")

    manifest = build_record_cache(
        args.data_root,
        args.cache_dir,
        record_ids=subject_map.keys(),
        rebuild=args.rebuild_cache,
    )
    global_index = build_window_index(
        manifest,
        record_ids=subject_map.keys(),
        window=256,
        stride=240,
    )
    output_root = args.results_dir / profile.name
    output_root.mkdir(parents=True, exist_ok=True)
    model_class = load_mamba_eeg_class(args.model_source)

    for seed in seeds:
        for test_subject in subjects:
            _run_one(
                test_subject=test_subject,
                seed=seed,
                profile=profile,
                args=args,
                subject_map=subject_map,
                manifest=manifest,
                global_index=global_index,
                output_root=output_root,
                model_class=model_class,
            )
            summary = aggregate_completed_runs(output_root, required_subjects=subjects)
            write_summary_outputs(summary, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
