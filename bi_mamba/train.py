"""
Bi-Mamba-S (Bidirectional Spatial Mixer) implementation for the EEG attention dataset.
This PyTorch-only variant keeps the preprocessing/evaluation pipeline from eegconformer/baseline.py
but replaces custom CUDA kernels with lightweight temporal convolution mixers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from eegconformer import baseline as eeg_base


class TemporalConvUnit(nn.Module):
    """
    Depthwise temporal convolution + GLU gating.
    Entirely PyTorch ops, so no custom CUDA kernels are needed.
    """

    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        xt = x.transpose(1, 2)  # [B, D, T]
        y = self.dw(xt)
        y = self.pw(y)
        y = F.glu(y, dim=1)
        y = self.drop(y)
        return y.transpose(1, 2)


class BiMambaBlock(nn.Module):
    """
    Processes the sequence in both temporal directions with temporal conv units
    and merges the outputs with a lightweight projection.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.fwd = TemporalConvUnit(d_model, kernel_size=d_state, dilation=d_conv, dropout=dropout)
        self.bwd = TemporalConvUnit(d_model, kernel_size=d_state, dilation=d_conv, dropout=dropout)
        hidden = d_model * expand
        self.mix = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.res_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        out_fwd = self.fwd(x)
        out_bwd = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        out = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.mix(out)
        return self.res_norm(out + x)


class BiMambaS(nn.Module):
    """
    Spatial projection + stack of bidirectional temporal conv blocks + classifier head.
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        d_model: int = 64,
        seq_len: int = 512,
        depth: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial_proj = nn.Conv1d(num_channels, d_model, kernel_size=1)
        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, seq_len))
        self.layers = nn.ModuleList(
            [BiMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        x = x.transpose(1, 2)  # -> [B, C, T]
        x = self.spatial_proj(x)
        if x.shape[2] == self.pos_embedding.shape[2]:
            x = x + self.pos_embedding
        x = x.transpose(1, 2)  # -> [B, T, D]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def parse_args():
    ap = argparse.ArgumentParser(description="Bi-Mamba-S (Bidirectional Spatial Mamba) for EEG attention.")
    ap.add_argument("--prepare", action="store_true", help="Only run preprocessing and exit.")
    ap.add_argument("--mat-root", type=Path, default=Path("EEGData"), help="Folder containing eeg_record*.mat")
    ap.add_argument("--out-dir", type=Path, default=Path("data/eeg_attention"), help="Output folder for signals/labels npy")
    ap.add_argument(
        "--label-source",
        choices=["field", "marker", "state"],
        default="state",
        help="Label source: field -> attribute inside mat; marker -> o.marker ratio; state -> timestamp rule",
    )
    ap.add_argument("--label-field", type=str, default=None, help="label-source=field attribute name (e.g., attention_state)")
    ap.add_argument("--marker-thresh", type=float, default=0.1, help="label-source=marker threshold on marker==1 ratio")
    ap.add_argument("--focused-minutes", type=float, default=10.0, help="label-source=state: <= minutes => focused (class 0)")
    ap.add_argument("--unfocused-minutes", type=float, default=20.0, help="label-source=state: > minutes => unfocused (class 1), else class 2")
    ap.add_argument("--window", type=int, default=512, help="Window length in samples (default 4s @128Hz)")
    ap.add_argument("--stride", type=int, default=512, help="Stride in samples (default no overlap)")
    ap.add_argument("--signals", type=Path, help="Existing signals.npy path (default out_dir/signals.npy)")
    ap.add_argument("--labels", type=Path, help="Existing labels.npy path (default out_dir/labels.npy)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--d-state", type=int, default=15, help="Temporal convolution kernel size for Bi blocks.")
    ap.add_argument("--d-conv", type=int, default=2, help="Temporal dilation factor inside Bi blocks.")
    ap.add_argument("--expand", type=int, default=2, help="Hidden expansion multiplier in the fusion MLP.")
    ap.add_argument("--num-classes", type=int, default=None, help="Override class count; otherwise inferred from data")
    ap.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-amp", action="store_true", help="Enable AMP when CUDA is available")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--save-dir", type=Path, default=Path("experiments/bi_mamba"))
    ap.add_argument("--no-normalize", action="store_true", help="Skip z-score in Dataset")
    return ap.parse_args()


def run_prepare(args: argparse.Namespace):
    if args.label_source == "marker":
        return eeg_base.prepare_from_marker(
            mat_root=args.mat_root,
            out_dir=args.out_dir,
            win=args.window,
            stride=args.stride,
            marker_thresh=args.marker_thresh,
        )
    if args.label_source == "state":
        return eeg_base.prepare_from_state(
            mat_root=args.mat_root,
            out_dir=args.out_dir,
            win=args.window,
            stride=args.stride,
            focused_minutes=args.focused_minutes,
            unfocused_minutes=args.unfocused_minutes,
        )
    if eeg_base.run_preprocess is None:
        raise ImportError("src.data.preprocess.run_preprocess not found; install deps or use --label-source marker/state.")
    return eeg_base.run_preprocess(
        mat_root=args.mat_root,
        out_dir=args.out_dir,
        win=args.window,
        stride=args.stride,
        label_field=args.label_field,
        bandpass=False,
        cache_graphs=False,
        thresh=0.5,
        topk=None,
    )


def main():
    args = parse_args()
    eeg_base.set_seed(args.seed)

    if args.prepare:
        run_prepare(args)
        return

    split = tuple(args.split)
    if not np.isclose(sum(split), 1.0):
        raise ValueError(f"Split ratios must sum to 1, got {split}")

    signals_path = args.signals or args.out_dir / "signals.npy"
    labels_path = args.labels or args.out_dir / "labels.npy"
    if not signals_path.exists() or not labels_path.exists():
        raise FileNotFoundError("signals.npy / labels.npy not found; run with --prepare or set --signals/--labels.")

    dataset = eeg_base.EEGConformerDataset(signals_path, labels_path, normalize=not args.no_normalize)
    num_classes = args.num_classes or dataset.num_classes

    train_loader, val_loader, test_loader = eeg_base.create_loaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=split,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.use_amp and device.type == "cuda")
    print(f"Cihaz: {device} | AMP: {amp_enabled}")

    model = BiMambaS(
        num_channels=dataset.num_channels,
        num_classes=num_classes,
        d_model=args.d_model,
        seq_len=dataset.seq_len,
        depth=args.depth,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Bi-Mamba-S parametre sayisi: {params/1e6:.3f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else torch.amp.GradScaler(enabled=False)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.save_dir / "best_bimamba.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = eeg_base.train_one_epoch(
            model, train_loader, optimizer, scaler, device, amp_enabled, args.grad_clip
        )
        val_loss, val_acc, _, _ = eeg_base.evaluate(model, val_loader, device, amp_enabled)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"Epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f}"
        )

    elapsed = time.time() - start
    print(f"Training time: {elapsed/60:.1f} min")

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded best val checkpoint: {ckpt_path}")

    test_loss, test_acc, y_true, y_pred = eeg_base.evaluate(model, test_loader, device, amp_enabled)
    print(f"Test -> loss: {test_loss:.4f} | acc: {test_acc:.4f}")

    serializable_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    metrics = {
        "history": history,
        "best_val_acc": best_val,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "config": serializable_args,
    }
    with open(args.save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    try:
        fig_path = args.save_dir / "training_curves.png"
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="train_loss")
        plt.plot(history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(history["train_acc"], label="train_acc")
        plt.plot(history["val_acc"], label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

        if y_true.size > 0:
            cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(np.unique(y_true)))
            disp.plot(cmap="Blues", colorbar=True)
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(args.save_dir / "confusion_matrix.png", dpi=200)
            plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

    try:
        lines = [
            "# Bi-Mamba-S Results",
            "",
            f"- Best val acc: {best_val:.4f}",
            f"- Test loss: {test_loss:.4f}",
            f"- Test acc: {test_acc:.4f}",
            f"- Epochs: {len(history['train_loss'])}",
            f"- Checkpoint: {ckpt_path}",
            "",
            "## Figures",
            f"- Training curves: {fig_path}",
        ]
        if y_true.size > 0:
            lines.append(f"- Confusion matrix: {args.save_dir / 'confusion_matrix.png'}")
        (args.save_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        print(f"Writing results.md failed: {e}")


if __name__ == "__main__":
    main()
