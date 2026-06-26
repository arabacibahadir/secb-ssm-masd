"""
Ablation study driver for EEG Attention Mamba.

This script reuses the main implementation in `attention_mamba/train.py`
and runs several controlled ablations to quantify where the performance gains
come from (bi-directional SSM, conv mixing, SE, depth, width, etc.).

Usage (from project root):
    python -m ablation_studies.run_ablation

Results:
    - Per-variant training logs and best checkpoint under `ablation_studies/experiments/<variant_name>/`
    - A JSON summary at `ablation_studies/ablation_results.json` with test metrics for all variants

The JSON summary can be used to prepare a model-variant comparison table.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from attention_mamba.train import (
    set_seed,
    build_dataset,
    create_loaders,
    SelectiveSSMLite,
    BiSelectiveSSM,
    FeedForward,
    MambaEEG,
    run_epoch,
    EarlyStopping,
)


ROOT = Path(__file__).resolve().parent


@dataclass
class AblationConfig:
    name: str
    description: str
    epochs: int = 50
    batch_size: int = 256
    lr: float = 4e-4
    weight_decay: float = 1e-2
    d_model: int = 224
    layers: int = 8
    ff_expansion: int = 4
    conv_kernel: int = 31
    emb_kernel: int = 7
    emb_stride: int = 2
    dropout: float = 0.18
    bidir: bool = True
    use_conv_mix: bool = True
    use_se: bool = True
    seed: int = 42
    epoch_length: float = 2.0
    step_size: float = 0.125
    data_root: Path = Path("EEGData")
    num_workers: int = 2
    use_amp: bool = True  # Use mixed precision training if GPU available
    use_early_stopping: bool = True


class ConvEmbeddingNoSE(nn.Module):
    """Conv embedding without squeeze-excitation (for SE ablation)."""

    def __init__(self, in_ch: int, dim: int, kernel: int, stride: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Conv1d(in_ch, dim, kernel_size=kernel, stride=stride, padding=pad)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x.transpose(1, 2)


class MambaBlockNoConv(nn.Module):
    """Mamba block without the ConvMix stage (for conv ablation)."""

    def __init__(self, dim: int, ff_expansion: int, conv_kernel: int, dropout: float, bidir: bool = True):
        super().__init__()
        self.ff1 = FeedForward(dim, ff_expansion, dropout)
        self.ff2 = FeedForward(dim, ff_expansion, dropout)
        self.ssm = BiSelectiveSSM(dim, dropout) if bidir else SelectiveSSMLite(dim, dropout)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.n3 = nn.LayerNorm(dim)
        self.n4 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(self.n1(x))
        x = x + self.drop(self.ssm(self.n2(x)))
        # conv-mix removed: identity
        x = x + 0.0 * self.n3(x)
        x = x + 0.5 * self.ff2(self.n4(x))
        return x


class MambaEEGVariant(nn.Module):
    """
    Wrapper around the original MambaEEG that allows:
      - disabling SE in the embedding
      - disabling conv mixing blocks
      - toggling bidirectional SSM

    When use_conv_mix=True and use_se=True and bidir=True, this reduces to the baseline model.
    """

    def __init__(self, input_dim: int, num_classes: int, cfg: AblationConfig):
        super().__init__()

        if cfg.use_se:
            # Use the original conv embedding with SE from attention_mamba
            self.embed = MambaEEG(
                input_dim=input_dim,
                num_classes=num_classes,
                d_model=cfg.d_model,
                layers=1,
                ff_exp=cfg.ff_expansion,
                conv_kernel=cfg.conv_kernel,
                emb_kernel=cfg.emb_kernel,
                emb_stride=cfg.emb_stride,
                dropout=cfg.dropout,
                bidir=cfg.bidir,
            ).embed
        else:
            self.embed = ConvEmbeddingNoSE(
                in_ch=input_dim,
                dim=cfg.d_model,
                kernel=cfg.emb_kernel,
                stride=cfg.emb_stride,
                dropout=cfg.dropout,
            )

        blocks = []
        for _ in range(cfg.layers):
            if cfg.use_conv_mix:
                blocks.append(
                    # Reuse the full baseline block through a small proxy MambaEEG
                    MambaEEG(
                        input_dim=input_dim,
                        num_classes=num_classes,
                        d_model=cfg.d_model,
                        layers=1,
                        ff_exp=cfg.ff_expansion,
                        conv_kernel=cfg.conv_kernel,
                        emb_kernel=cfg.emb_kernel,
                        emb_stride=cfg.emb_stride,
                        dropout=cfg.dropout,
                        bidir=cfg.bidir,
                    ).blocks[0]
                )
            else:
                blocks.append(
                    MambaBlockNoConv(
                        dim=cfg.d_model,
                        ff_expansion=cfg.ff_expansion,
                        conv_kernel=cfg.conv_kernel,
                        dropout=cfg.dropout,
                        bidir=cfg.bidir,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)


def build_model(input_dim: int, num_classes: int, cfg: AblationConfig) -> nn.Module:
    # Pure baseline: use the original MambaEEG exactly
    if cfg.use_conv_mix and cfg.use_se and cfg.bidir and cfg.layers == 8 and cfg.d_model == 224:
        return MambaEEG(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=cfg.d_model,
            layers=cfg.layers,
            ff_exp=cfg.ff_expansion,
            conv_kernel=cfg.conv_kernel,
            emb_kernel=cfg.emb_kernel,
            emb_stride=cfg.emb_stride,
            dropout=cfg.dropout,
            bidir=True,
        )
    # Otherwise construct a variant model
    return MambaEEGVariant(input_dim=input_dim, num_classes=num_classes, cfg=cfg)


def run_single_ablation(cfg: AblationConfig, resume: bool = False) -> Dict:
    print(f"\n===== Running ablation: {cfg.name} =====")
    print(cfg.description)
    set_seed(cfg.seed)

    signals, labels = build_dataset(
        cfg.data_root,
        epoch_length=cfg.epoch_length,
        step_size=cfg.step_size,
    )
    print(f"Dataset: signals {signals.shape}, labels {labels.shape}")

    train_loader, val_loader, test_loader = create_loaders(
        signals, labels, batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = signals.shape[2]
    num_classes = int(labels.max() + 1)

    model = build_model(input_dim, num_classes, cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if amp_enabled:
        print(f"[{cfg.name}] Using mixed precision training (AMP)")

    save_root = ROOT / "experiments" / cfg.name
    save_root.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_root / "best.pt"
    last_ckpt = save_root / "last.pt"
    epoch_log = save_root / "epoch_metrics.jsonl"

    # History and resume logic
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_epoch = 1
    best_val = -1.0

    if resume:
        # Try to resume model weights from last checkpoint
        if last_ckpt.exists():
            model.load_state_dict(torch.load(last_ckpt, map_location=device))
            print(f"[{cfg.name}] Resumed weights from {last_ckpt}")
        elif best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device))
            print(f"[{cfg.name}] Resumed weights from {best_ckpt}")

        # Resume history and starting epoch from previous logs if available
        if epoch_log.exists():
            lines = epoch_log.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                try:
                    last_rec = json.loads(lines[-1])
                    start_epoch = int(last_rec.get("epoch", 0)) + 1
                    print(f"[{cfg.name}] Resuming from epoch {start_epoch}")
                except Exception:
                    start_epoch = 1

        metrics_path = save_root / "metrics.json"
        if metrics_path.exists():
            try:
                prev = json.loads(metrics_path.read_text(encoding="utf-8"))
                if "history" in prev:
                    history = prev["history"]
                if "best_val_acc" in prev:
                    best_val = float(prev["best_val_acc"])
            except Exception:
                pass
    else:
        # Fresh run: clear epoch log
        epoch_log.write_text("", encoding="utf-8")

    stopper = EarlyStopping(patience=5) if cfg.use_early_stopping else None

    start = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            scaler,
            amp=amp_enabled,
            train=True,
            log_every=10,
            epoch=epoch,
            epochs=cfg.epochs,
        )
        va_loss, va_acc, va_f1, _, _ = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            criterion,
            scaler,
            amp=amp_enabled,
            train=False,
        )

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        scheduler.step()

        torch.save(model.state_dict(), last_ckpt)

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), best_ckpt)

        # Log epoch metrics to JSONL file
        epoch_data = {
            "epoch": epoch,
            "train_loss": float(tr_loss),
            "train_acc": float(tr_acc),
            "train_f1": float(tr_f1),
            "val_loss": float(va_loss),
            "val_acc": float(va_acc),
            "val_f1": float(va_f1),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        with epoch_log.open("a", encoding="utf-8") as f:
            json.dump(epoch_data, f)
            f.write("\n")

        if stopper is not None:
            stopper.step(va_loss, model)
            if stopper.early_stop:
                print(f"[{cfg.name}] Early stopping triggered.")
                break

        print(
            f"[{cfg.name}] Epoch {epoch:03d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}"
        )

    elapsed = time.time() - start
    print(f"[{cfg.name}] Training time: {elapsed/60:.1f} min")

    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"[{cfg.name}] Loaded best checkpoint: {best_ckpt}")

    te_loss, te_acc, te_f1, te_prec, te_rec = run_epoch(
        model,
        test_loader,
        optimizer,
        device,
        criterion,
        scaler,
        amp=amp_enabled,
        train=False,
    )

    metrics = {
        "config": asdict(cfg),
        "history": history,
        "test": {
            "loss": float(te_loss),
            "acc": float(te_acc),
            "f1": float(te_f1),
            "precision": float(te_prec),
            "recall": float(te_rec),
        },
        "best_val_acc": float(best_val),
        "elapsed_min": elapsed / 60.0,
        "epoch_log_file": str(epoch_log),
    }

    with open(save_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def get_ablation_scenarios() -> Dict[str, AblationConfig]:
    base = AblationConfig(
        name="baseline",
        description="Full model: bi-SSM, conv embedding with SE, conv mixing, 8 layers, d_model=224.",
    )
    return {
        "baseline": base,
        "no_bidir": AblationConfig(
            name="no_bidir",
            description="Unidirectional SSM instead of bi-directional.",
            bidir=False,
        ),
        "no_conv": AblationConfig(
            name="no_conv",
            description="Mamba block without the ConvMix temporal convolution.",
            use_conv_mix=False,
        ),
        "no_se": AblationConfig(
            name="no_se",
            description="Conv embedding without squeeze-excitation.",
            use_se=False,
        ),
        "shallow_4layers": AblationConfig(
            name="shallow_4layers",
            description="Reduced depth: 4 stacked Mamba blocks instead of 8.",
            layers=4,
        ),
        "narrow_128": AblationConfig(
            name="narrow_128",
            description="Reduced width: d_model=128 instead of 224.",
            d_model=128,
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation experiments for EEG Attention Mamba.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["all"],
        help=(
            "Which ablation variants to run. "
            "Use names from {baseline, no_bidir, no_conv, no_se, shallow_4layers, narrow_128} "
            "or 'all' to run every variant."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training for the selected variants from their existing checkpoints/logs if available.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=None,
        help="Enable mixed precision training (AMP). Default: True if GPU available.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision training (AMP). Overrides --use-amp.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scenarios = get_ablation_scenarios()

    # Determine AMP usage from CLI
    if args.no_amp:
        use_amp_override = False
    elif args.use_amp is not None:
        use_amp_override = args.use_amp
    else:
        use_amp_override = None  # Use default from config

    # Determine which variants to run
    if "all" in args.variants:
        selected_items = list(scenarios.items())
    else:
        selected_items = []
        for name in args.variants:
            if name not in scenarios:
                print(f"[warn] Unknown variant '{name}' - skipping.")
                continue
            selected_items.append((name, scenarios[name]))

    if not selected_items:
        print("No valid variants selected. Nothing to run.")
        return

    # Override use_amp if specified via CLI
    if use_amp_override is not None:
        for name, cfg in selected_items:
            cfg.use_amp = use_amp_override

    out_path = ROOT / "ablation_results.json"
    if out_path.exists():
        # Merge into existing results so you can run piecewise
        with out_path.open("r", encoding="utf-8") as f:
            summary: Dict[str, Dict] = json.load(f)
    else:
        summary = {}

    for name, cfg in selected_items:
        metrics = run_single_ablation(cfg, resume=args.resume)
        summary[name] = metrics

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nUpdated ablation summary at: {out_path}")
    print("Ablation summary written; use it to prepare a variant-level metrics table.")


if __name__ == "__main__":
    main()
