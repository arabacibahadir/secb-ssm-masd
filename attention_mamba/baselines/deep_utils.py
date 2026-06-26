"""
Shared training/eval utilities for deep baselines.
Each model script calls run_deep_baseline with its model builder to get consistent splits/metrics.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data_utils import build_dataset, create_loaders, set_seed
from .metrics_utils import aggregate_seed_metrics, compute_metrics


def add_common_args(ap, default_save_subdir: str, seeds_default=None):
    ap.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "EEGData")
    ap.add_argument("--epoch-length", type=float, default=2.0)
    ap.add_argument("--step-size", type=float, default=0.125)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5, help="Epoch print frequency (0 disables).")
    seeds_default = seeds_default or [42, 43, 44, 45, 46]
    ap.add_argument("--seeds", type=int, nargs="+", default=seeds_default)
    ap.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed used only for the train/validation/test partition; fixed across model seeds.",
    )
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--save-dir", type=Path, default=Path(__file__).resolve().parents[1] / "experiments" / "baselines" / default_save_subdir)
    return ap


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device, scaler: Optional[torch.cuda.amp.GradScaler], amp: bool):
    model.train()
    total_loss = 0.0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(xb)
            loss = criterion(logits, yb)
        if scaler is not None and amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total += xb.size(0)
    return total_loss / max(total, 1)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, criterion, device) -> Dict:
    model.eval()
    total_loss = 0.0
    total = 0
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    y_proba: List[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        prob = F.softmax(logits, dim=-1)

        total_loss += loss.item() * xb.size(0)
        total += xb.size(0)
        y_true.append(yb.detach().cpu().numpy())
        y_pred.append(logits.argmax(dim=-1).detach().cpu().numpy())
        y_proba.append(prob.detach().cpu().numpy())
    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)
    y_proba_np = np.concatenate(y_proba)
    metrics = compute_metrics(y_true_np, y_pred_np, y_proba_np)
    metrics["loss"] = float(total_loss / max(total, 1))
    return metrics


def run_deep_baseline(
    model_builder: Callable[[int, int], nn.Module],
    args,
    model_name: str,
    extra_config: Optional[Dict] = None,
) -> Dict:
    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.use_amp and device.type == "cuda")
    print(f"Running {model_name} on device {device} (AMP={amp})")

    signals, labels = build_dataset(args.data_root, epoch_length=args.epoch_length, step_size=args.step_size)
    input_dim = signals.shape[2]
    num_classes = int(labels.max() + 1)

    per_seed = []
    for seed in args.seeds:
        set_seed(seed)
        train_loader, val_loader, test_loader = create_loaders(
            signals,
            labels,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.split_seed,
        )
        model = model_builder(input_dim, num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=amp) if device.type == "cuda" else None

        best_state = copy.deepcopy(model.state_dict())
        best_val_score = -1.0
        best_val_metrics = None

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, amp)
            val_metrics = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()

            val_score = val_metrics.get("macro_f1", val_metrics.get("balanced_accuracy", 0.0))
            if val_score > best_val_score:
                best_val_score = val_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_val_metrics = val_metrics

            if args.log_every > 0 and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs):
                print(
                    f"[{model_name}] seed {seed} epoch {epoch}/{args.epochs} "
                    f"train_loss {tr_loss:.4f} val_f1 {val_metrics['macro_f1']:.4f}"
                )

        model.load_state_dict(best_state)
        test_metrics = eval_epoch(model, test_loader, criterion, device)
        per_seed.append(
            {
                "seed": seed,
                "best_val_metrics": best_val_metrics,
                "test": test_metrics,
            }
        )

    validation_summary = aggregate_seed_metrics([s["best_val_metrics"] for s in per_seed])
    test_summary = aggregate_seed_metrics([s["test"] for s in per_seed])
    return {
        "model": model_name,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seeds": args.seeds,
            "split_seed": args.split_seed,
            "extra": extra_config or {},
        },
        "seeds": per_seed,
        "validation_summary_mean_std": validation_summary,
        "test_summary_mean_std": test_summary,
    }


def save_results(save_path: Path, payload: Dict):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {save_path}")
