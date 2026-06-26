"""
Phase-Amplitude Graph Diffusion + Neural CDE EEG model.
- Learnable montage (soft re-referencing)
- Hilbert transform to separate amplitude/phase
- Multi-scale temporal conv per channel
- Channel diffusion via learned spatial kernels
- Discrete Neural CDE integrator for long-range dynamics
Keeps the same signals.npy / labels.npy interface as the baseline.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix

torch.set_float32_matmul_precision("medium")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_from_state(mat_root: Path, out_dir: Path, win: int, stride: int,
                       focused_minutes: float, unfocused_minutes: float, fs: int = 128):
    """
    Timestamp rule (focused / drowsy / unfocused) used in the baseline.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    signals: list[np.ndarray] = []
    labels: list[int] = []

    files = sorted(mat_root.glob("eeg_record*.mat"))
    if not files:
        raise FileNotFoundError(f"No eeg_record*.mat under {mat_root}.")

    import scipy.io as sio  # local import; only needed during prepare

    focused_limit = focused_minutes * 60 * fs
    unfocused_limit = unfocused_minutes * 60 * fs

    for f in files:
        mat = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
        if "o" not in mat:
            raise ValueError(f"'o' key missing in {f}.")
        data = np.asarray(mat["o"].data, dtype=np.float32)  # [T, 25]
        eeg = data[:, 3:17]  # 14 channels

        timestamp = np.arange(eeg.shape[0])
        state = np.full_like(timestamp, 1, dtype=np.int64)  # default unfocused
        state[timestamp <= focused_limit] = 0
        state[timestamp > unfocused_limit] = 2

        scaler = StandardScaler(with_mean=True, with_std=True)
        eeg = scaler.fit_transform(eeg).astype(np.float32)

        total = eeg.shape[0]
        for start in range(0, total - win + 1, stride):
            window = eeg[start:start + win]
            lbl_window = state[start:start + win]
            if lbl_window.size == 0:
                continue
            counts = np.bincount(lbl_window, minlength=3)
            lbl = int(np.argmax(counts))
            signals.append(window.T.astype(np.float32))  # [C, win]
            labels.append(lbl)

    if not signals:
        raise ValueError("No windows were produced; check window/stride.")

    signals_arr = np.stack(signals)
    labels_arr = np.array(labels, dtype=np.int64)
    np.save(out_dir / "signals.npy", signals_arr)
    np.save(out_dir / "labels.npy", labels_arr)
    meta = {
        "files": [str(f) for f in files],
        "window": win,
        "stride": stride,
        "label_source": "state_timestamp",
        "focused_minutes": focused_minutes,
        "unfocused_minutes": unfocused_minutes,
        "signals_shape": signals_arr.shape,
        "label_counts": {int(k): int(v) for k, v in zip(*np.unique(labels_arr, return_counts=True))},
    }
    with open(out_dir / "meta_state.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Timestamp-based labeling done. Label distribution:", meta["label_counts"])
    return meta


class EEGDataset(Dataset):
    """
    Loads signals.npy [N, C, T] and labels.npy [N].
    Filters out negative labels and reindexes remaining to start from zero.
    """

    def __init__(self, signals_path: Path, labels_path: Path, normalize: bool = True):
        self.signals = np.load(signals_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r").astype(np.int64)

        if self.signals.shape[0] != self.labels.shape[0]:
            raise ValueError("signals.npy and labels.npy length mismatch.")

        self.indices = np.where(self.labels >= 0)[0]
        if len(self.indices) == 0:
            raise ValueError("No valid labels (>=0) found.")

        self.label_offset = int(self.labels[self.indices].min())
        self.num_classes = int(self.labels[self.indices].max() - self.label_offset + 1)
        self.num_channels = int(self.signals.shape[1])
        self.seq_len = int(self.signals.shape[2])
        self.normalize = normalize

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        x = np.array(self.signals[real_idx], dtype=np.float32, copy=True)  # [C, T]
        if self.normalize:
            x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.from_numpy(x.T)  # [T, C]
        y = int(self.labels[real_idx] - self.label_offset)
        return x, torch.tensor(y, dtype=torch.long)


class LearnableMontage(nn.Module):
    """
    Soft re-referencing: learn a channel mix matrix applied to every time step.
    """

    def __init__(self, channels: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(channels, channels, bias=bias)
        nn.init.eye_(self.proj.weight)
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        return self.proj(x)


def hilbert_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Hilbert transform along time (dim=1). Returns analytic signal (complex).
    Kept in float32 and no grad to avoid AMP ComplexHalf overhead.
    """
    with torch.no_grad():
        x_f = x.float()
        T = x_f.shape[1]
        Xf = torch.fft.fft(x_f, dim=1)
        h = torch.zeros(T, device=x.device, dtype=Xf.dtype)
        if T % 2 == 0:
            h[0] = h[T // 2] = 1
            h[1:T // 2] = 2
        else:
            h[0] = 1
            h[1:(T + 1) // 2] = 2
        analytic = torch.fft.ifft(Xf * h.view(1, -1, 1), dim=1)
    return analytic


class PhaseAmplitudeEncoder(nn.Module):
    """
    Builds per-channel embeddings from amplitude + phase (sin/cos).
    """

    def __init__(self, channels: int, dim: int, use_hilbert: bool = True):
        super().__init__()
        self.proj = nn.Linear(3, dim)
        self.norm = nn.LayerNorm(dim)
        self.channels = channels
        self.use_hilbert = use_hilbert

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        if self.use_hilbert:
            analytic = hilbert_transform(x)  # complex
            amp = analytic.abs()
            phase = torch.angle(analytic)
        else:
            amp = x.abs()
            phase = torch.zeros_like(x)
        amp = torch.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)
        phase = torch.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
        feat = torch.stack([amp, torch.sin(phase), torch.cos(phase)], dim=-1)  # [B, T, C, 3]
        feat = self.proj(feat)
        feat = self.norm(feat)
        return feat, amp


class MultiScaleTemporalConv(nn.Module):
    """
    Per-channel temporal conv with multiple dilation scales + learned mixing.
    """

    def __init__(self, channels: int, dim: int, kernel_sizes: Iterable[int], dilations: Iterable[int], dropout: float):
        super().__init__()
        self.branches = nn.ModuleList()
        in_ch = channels * dim
        for k, d in zip(kernel_sizes, dilations):
            pad = ((k - 1) // 2) * d
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=pad, dilation=d, groups=channels),
                    nn.GLU(dim=1),
                    nn.Conv1d(in_ch // 2, in_ch, kernel_size=1, groups=channels),
                )
            )
        self.scale_logits = nn.Parameter(torch.zeros(len(self.branches)))
        self.dropout = nn.Dropout(dropout)
        self.channels = channels
        self.dim = dim

    def forward(self, x: torch.Tensor):
        # x: [B, T, C, D]
        B, T, C, D = x.shape
        assert C == self.channels and D == self.dim
        flat = x.permute(0, 2, 3, 1).reshape(B, C * D, T)  # [B, C*D, T]
        outs = []
        for branch in self.branches:
            outs.append(branch(flat))
        stack = torch.stack(outs, dim=-1)  # [B, C*D, T, S]
        weights = F.softmax(self.scale_logits, dim=0)
        mixed = torch.sum(stack * weights.view(1, 1, 1, -1), dim=-1)  # [B, C*D, T]
        mixed = self.dropout(mixed)
        return mixed.view(B, C, D, T).permute(0, 3, 1, 2)  # [B, T, C, D]


class ChannelDiffusion(nn.Module):
    """
    Graph-style diffusion over channels using learned spatial anchors and amplitude-aware gating.
    """

    def __init__(self, channels: int, dim: int, pos_dim: int = 8, temperature: float = 0.7):
        super().__init__()
        self.channel_pos = nn.Parameter(torch.randn(channels, pos_dim) * 0.1)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.amp_proj = nn.Linear(1, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, amp: torch.Tensor):
        # x: [B, T, C, D], amp: [B, T, C]
        B, T, C, D = x.shape
        pos = self.channel_pos  # [C, P]
        dist2 = torch.cdist(pos, pos, p=2).pow(2)  # [C, C]
        temp = torch.clamp(self.temperature.abs(), min=1e-3, max=5.0)
        base_adj = torch.exp(-dist2 / temp)  # [C, C]

        amp_mean = amp.mean(dim=1)  # [B, C]
        amp_gate = torch.tanh(self.amp_proj(amp_mean.unsqueeze(-1))).squeeze(-1)  # [B, C]
        bias = amp_gate.unsqueeze(2) + amp_gate.unsqueeze(1)  # [B, C, C]
        adj = base_adj.unsqueeze(0) * torch.sigmoid(bias)
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        x_norm = self.norm(x)
        mixed = torch.einsum("bij,btjd->btid", adj, x_norm)
        return mixed


class TemporalCDEBlock(nn.Module):
    """
    Discrete Neural CDE-style integrator over time steps.
    """

    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.init = nn.Linear(dim, hidden)
        self.f = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.g = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.drop = nn.Dropout(dropout)
        for m in [self.init, *self.f, *self.g]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, seq: torch.Tensor):
        # seq: [B, T, F]
        B, T, F = seq.shape
        h = self.init(seq[:, 0])
        if T <= 1:
            return self.drop(h)
        dt = 1.0 / float(T - 1)
        for t in range(T - 1):
            delta_x = seq[:, t + 1] - seq[:, t]
            delta_x = torch.clamp(delta_x, -3.0, 3.0)
            drift = self.f(h)
            control = self.g(delta_x)
            h = h + dt * (drift + control)
            h = torch.tanh(h)
            h = self.drop(h)
        return h


class GatedHead(nn.Module):
    def __init__(self, dim: int, classes: int, dropout: float):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, classes))

    def forward(self, pooled: torch.Tensor):
        gate = self.se(pooled)
        pooled = pooled * gate
        return self.classifier(pooled)


class PhaseCDEModel(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        embed_dim: int = 32,
        cde_hidden: int = 128,
        kernel_sizes: Iterable[int] = (7, 15, 31),
        dilations: Iterable[int] = (1, 2, 4),
        dropout: float = 0.1,
        channel_dropout: float = 0.1,
        use_diffusion: bool = True,
        use_hilbert: bool = True,
    ):
        super().__init__()
        self.montage = LearnableMontage(channels)
        self.encoder = PhaseAmplitudeEncoder(channels, embed_dim, use_hilbert=use_hilbert)
        self.temporal = MultiScaleTemporalConv(channels, embed_dim, kernel_sizes, dilations, dropout)
        self.diffusion = ChannelDiffusion(channels, embed_dim) if use_diffusion else None
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.cde = TemporalCDEBlock(dim=channels * embed_dim, hidden=cde_hidden, dropout=dropout)
        self.head = GatedHead(dim=cde_hidden, classes=num_classes, dropout=dropout)
        self.channel_dropout = channel_dropout
        self.use_diffusion = use_diffusion
        self.use_hilbert = use_hilbert

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        if self.training and self.channel_dropout > 0:
            mask = torch.rand(x.shape[0], 1, x.shape[2], device=x.device)
            x = x * (mask > self.channel_dropout).float()

        x = self.montage(x)
        x = torch.clamp(x, -6.0, 6.0)
        feats, amp = self.encoder(x)  # feats [B, T, C, D]
        feats = self.temporal(feats)  # [B, T, C, D]
        feats = torch.clamp(feats, -5.0, 5.0)
        if self.diffusion is not None:
            feats = feats + self.diffusion(feats, amp)  # diffusion residual
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        feats = self.temporal_norm(feats)

        flat = feats.reshape(feats.shape[0], feats.shape[1], -1)  # [B, T, C*D]
        flat = torch.clamp(flat, -5.0, 5.0)
        state = self.cde(flat)
        logits = self.head(state)
        return logits


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return contextlib.nullcontext()


def create_loaders(dataset: EEGDataset, batch_size: int, num_workers: int,
                   split: tuple[float, float, float], seed: int):
    n_total = len(dataset)
    n_train = int(split[0] * n_total)
    n_val = int(split[1] * n_total)
    n_test = n_total - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, device: torch.device, amp: bool, grad_clip: float | None,
                    max_batches: int | None, log_every: int, epoch: int, log_path: Path):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    steps_total = len(loader)
    if max_batches is not None:
        steps_total = min(steps_total, max_batches)
    epoch_start = time.time()
    for step, (xb, yb) in enumerate(loader, 1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
        if not torch.isfinite(loss):
            print(f"Non-finite loss at step {step}, skipping batch.")
            print(f"  xb min/max: {xb.min().item():.4f}/{xb.max().item():.4f}")
            print(f"  logits finite: {torch.isfinite(logits).all().item()} | min/max: {logits.nan_to_num().min().item():.4f}/{logits.nan_to_num().max().item():.4f}")
            print(f"  logits any nan: {torch.isnan(logits).any().item()} inf: {torch.isinf(logits).any().item()}")
            print(f"  targets unique: {yb.unique().tolist()}")
            continue
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
        if log_every and step % log_every == 0:
            elapsed = time.time() - epoch_start
            remaining_steps = max(steps_total - step, 0)
            eta_min = (elapsed / max(step, 1)) * remaining_steps / 60.0 if step > 0 else 0.0
            batch_acc = (preds == yb).float().mean().item()
            print(f"  epoch {epoch} step {step}/{steps_total} loss {loss.item():.4f} acc {batch_acc:.4f} | elapsed {elapsed/60:.2f}m ETA {eta_min:.2f}m")
            log_entry = {
                "epoch": epoch,
                "step": step,
                "steps_total": steps_total,
                "loss": loss.item(),
                "batch_acc": batch_acc,
                "elapsed_min": elapsed / 60.0,
                "eta_min": eta_min,
                "time": time.time(),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        if max_batches is not None and step >= max_batches:
            break
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool, return_details: bool = False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    y_prob = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
        if return_details:
            y_true.append(yb.detach().cpu())
            y_pred.append(preds.detach().cpu())
            y_prob.append(F.softmax(logits, dim=-1).detach().cpu())
    if return_details:
        y_true = torch.cat(y_true).numpy() if y_true else np.array([])
        y_pred = torch.cat(y_pred).numpy() if y_pred else np.array([])
        y_prob = torch.cat(y_prob).numpy() if y_prob else np.array([])
        return total_loss / max(total, 1), correct / max(total, 1), y_true, y_pred, y_prob
    return total_loss / max(total, 1), correct / max(total, 1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int):
    if y_true.size == 0:
        return {}
    metrics: dict[str, float | dict] = {}
    metrics["acc"] = float((y_true == y_pred).mean())
    metrics["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    metrics["per_class"] = {
        "precision": [float(v) for v in prec],
        "recall": [float(v) for v in rec],
        "f1": [float(v) for v in f1],
    }
    try:
        metrics["auroc_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    except Exception:
        metrics["auroc_macro"] = None
    try:
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        metrics["confusion_matrix"] = cm.tolist()
    except Exception:
        metrics["confusion_matrix"] = None
    return metrics


def parse_args():
    ap = argparse.ArgumentParser(description="Phase-Amplitude Graph Diffusion + CDE EEG model.")
    ap.add_argument("--prepare", action="store_true", help="Only run preprocessing and exit.")
    ap.add_argument("--mat-root", type=Path, default=Path("EEGData"), help="Folder containing eeg_record*.mat")
    ap.add_argument("--out-dir", type=Path, default=Path("data/eeg_attention"), help="Output folder for signals/labels npy")
    ap.add_argument("--focused-minutes", type=float, default=10.0)
    ap.add_argument("--unfocused-minutes", type=float, default=20.0)
    ap.add_argument("--window", type=int, default=512, help="Window length in samples (default 4s @128Hz)")
    ap.add_argument("--stride", type=int, default=512, help="Stride in samples")
    ap.add_argument("--signals", type=Path, help="Existing signals.npy path (default out_dir/signals.npy)")
    ap.add_argument("--labels", type=Path, help="Existing labels.npy path (default out_dir/labels.npy)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4, was 2e-4)")
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--cde-hidden", type=int, default=128)
    ap.add_argument("--kernel-sizes", type=int, nargs="+", default=[7, 15, 31])
    ap.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--channel-dropout", type=float, default=0.1)
    ap.add_argument("--no-diffusion", action="store_true", help="Disable channel diffusion (ablation).")
    ap.add_argument("--num-classes", type=int, default=None, help="Override class count; otherwise inferred from data")
    ap.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader worker processes.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-amp", action="store_true", help="Enable AMP when CUDA is available")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--save-dir", type=Path, default=Path("experiments/phase_cde"))
    ap.add_argument("--log-file", type=Path, default=None, help="Path to JSONL training log (default save_dir/train_log.jsonl)")
    ap.add_argument("--no-normalize", action="store_true", help="Skip z-score in Dataset")
    ap.add_argument("--simple-phase", action="store_true", help="Skip Hilbert; use abs(x) and zero phase (faster debug).")
    ap.add_argument("--max-train-batches", type=int, default=None, help="Limit batches per epoch (debug speed).")
    ap.add_argument("--log-every", type=int, default=0, help="Print loss every N steps (debug).")
    ap.add_argument("--force-fp32", action="store_true", help="Disable AMP for stability.")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.prepare:
        meta = prepare_from_state(
            mat_root=args.mat_root,
            out_dir=args.out_dir,
            win=args.window,
            stride=args.stride,
            focused_minutes=args.focused_minutes,
            unfocused_minutes=args.unfocused_minutes,
        )
        print("Prepared:", meta)
        return

    split = tuple(args.split)
    if not np.isclose(sum(split), 1.0):
        raise ValueError(f"Split ratios must sum to 1, got {split}")
    if len(args.kernel_sizes) != len(args.dilations):
        raise ValueError(f"kernel-sizes and dilations must match in length, got {len(args.kernel_sizes)} vs {len(args.dilations)}")

    signals_path = args.signals or args.out_dir / "signals.npy"
    labels_path = args.labels or args.out_dir / "labels.npy"
    if not signals_path.exists() or not labels_path.exists():
        raise FileNotFoundError("signals.npy / labels.npy not found; run with --prepare or set --signals/--labels.")

    dataset = EEGDataset(signals_path, labels_path, normalize=not args.no_normalize)
    num_classes = args.num_classes or dataset.num_classes

    train_loader, val_loader, test_loader = create_loaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=split,
        seed=args.seed,
    )

    # CUDA detection with better WSL support
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {device}")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, using CPU (will be slow!)")
        print("Check: nvidia-smi, CUDA installation, PyTorch CUDA build")
    
    amp_enabled = bool(args.use_amp and device.type == "cuda" and not args.force_fp32)
    print(f"Device: {device} | AMP: {amp_enabled} | num_workers: {args.num_workers}")

    model = PhaseCDEModel(
        channels=dataset.num_channels,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        cde_hidden=args.cde_hidden,
        kernel_sizes=args.kernel_sizes,
        dilations=args.dilations,
        dropout=args.dropout,
        channel_dropout=args.channel_dropout,
        use_diffusion=not args.no_diffusion,
        use_hilbert=not args.simple_phase,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else torch.amp.GradScaler(enabled=False)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or (args.save_dir / "train_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.save_dir / "best_phasecde.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            args.grad_clip,
            max_batches=args.max_train_batches,
            log_every=args.log_every,
            epoch=epoch,
            log_path=log_path,
        )
        val_loss, val_acc, yv_true, yv_pred, yv_prob = evaluate(model, val_loader, device, amp_enabled, return_details=True)
        val_metrics = compute_metrics(yv_true, yv_pred, yv_prob, num_classes)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt_path)

        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f} bal_acc {val_metrics.get('balanced_acc', 0):.4f} f1_macro {val_metrics.get('f1_macro', 0):.4f}")

        # Append epoch log
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_metrics": val_metrics,
            "time": time.time(),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_log) + "\n")

    elapsed = time.time() - start
    print(f"Training time: {elapsed/60:.1f} min")

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded best val checkpoint: {ckpt_path}")

    test_loss, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, device, amp_enabled, return_details=True)
    test_metrics = compute_metrics(y_true, y_pred, y_prob, num_classes)
    print(f"Test -> loss: {test_loss:.4f} | acc: {test_acc:.4f} | bal_acc: {test_metrics.get('balanced_acc'):.4f} | f1_macro: {test_metrics.get('f1_macro'):.4f}")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "test", "test_loss": test_loss, "test_acc": test_acc,
                            "test_metrics": test_metrics, "time": time.time()}) + "\n")

    serializable_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    metrics = {
        "history": history,
        "best_val_acc": best_val,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_metrics": test_metrics,
        "config": serializable_args,
    }
    with open(args.save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
