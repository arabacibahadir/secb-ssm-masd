#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG FusionCNN model.
- Hybrid CNN: time-domain EEGNet+InceptionTCN + freq-domain STFT+2D CNN
- Attentive statistics pooling with mean and standard deviation features.
- Same CLI style as baseline.py (prepare + train/test + metrics + plots)

Expected input: signals.npy [N,C,T], labels.npy [N] in --out-dir (created by --prepare).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Optional: reuse existing preprocessing utilities if available (same pattern as your baseline)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
try:
    from src.data.preprocess import run_preprocess  # type: ignore
except Exception:
    run_preprocess = None


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return contextlib.nullcontext()


# -------------------------
# Prepare (same labeling logic as baseline.py)
# -------------------------
def prepare_from_marker(mat_root: Path, out_dir: Path, win: int, stride: int, marker_thresh: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    signals: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []  # file id to enable group split in paper

    files = sorted(mat_root.glob("eeg_record*.mat"))
    if not files:
        raise FileNotFoundError(f"No eeg_record*.mat under {mat_root}.")

    import scipy.io as sio  # local import

    for fid, f in enumerate(files):
        mat = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
        if "o" not in mat:
            raise ValueError(f"'o' key missing in {f}.")
        o = mat["o"]
        data = np.asarray(o.data, dtype=np.float32)
        marker = np.asarray(o.marker)
        if marker.shape[0] != data.shape[0]:
            raise ValueError(f"marker length mismatch with data in {f}")

        eeg = data[:, 3:17]  # 14 channels
        eeg = (eeg - eeg.mean(axis=0, keepdims=True)) / (eeg.std(axis=0, keepdims=True) + 1e-6)

        total = eeg.shape[0]
        for start in range(0, total - win + 1, stride):
            window = eeg[start:start + win]        # [win, C]
            window = window.T.astype(np.float32)   # [C, win]
            mwin = marker[start:start + win]
            ratio = float((mwin == 1).mean())
            lbl = 1 if ratio >= marker_thresh else 0
            signals.append(window)
            labels.append(lbl)
            groups.append(fid)

    signals_arr = np.stack(signals)
    labels_arr = np.array(labels, dtype=np.int64)
    groups_arr = np.array(groups, dtype=np.int64)

    np.save(out_dir / "signals.npy", signals_arr)
    np.save(out_dir / "labels.npy", labels_arr)
    np.save(out_dir / "groups.npy", groups_arr)

    meta = {
        "files": [str(f) for f in files],
        "window": win,
        "stride": stride,
        "label_source": "marker",
        "marker_thresh": marker_thresh,
        "signals_shape": signals_arr.shape,
        "label_counts": {int(k): int(v) for k, v in zip(*np.unique(labels_arr, return_counts=True))},
        "groups_saved": True,
    }
    with open(out_dir / "meta_marker.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Marker-based labeling done. Label distribution:", meta["label_counts"])
    return meta


def prepare_from_state(mat_root: Path, out_dir: Path, win: int, stride: int,
                       focused_minutes: float, unfocused_minutes: float, fs: int = 128):
    out_dir.mkdir(parents=True, exist_ok=True)
    signals: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []  # file id to enable group split in paper

    files = sorted(mat_root.glob("eeg_record*.mat"))
    if not files:
        raise FileNotFoundError(f"No eeg_record*.mat under {mat_root}.")

    import scipy.io as sio  # local import

    focused_limit = focused_minutes * 60 * fs
    unfocused_limit = unfocused_minutes * 60 * fs

    for fid, f in enumerate(files):
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
            window = eeg[start:start + win]  # [win, C]
            lbl_window = state[start:start + win]
            counts = np.bincount(lbl_window, minlength=3)
            lbl = int(np.argmax(counts))
            signals.append(window.T.astype(np.float32))  # [C, win]
            labels.append(lbl)
            groups.append(fid)

    if not signals:
        raise ValueError("No windows were produced; check window/stride.")

    signals_arr = np.stack(signals)
    labels_arr = np.array(labels, dtype=np.int64)
    groups_arr = np.array(groups, dtype=np.int64)

    np.save(out_dir / "signals.npy", signals_arr)
    np.save(out_dir / "labels.npy", labels_arr)
    np.save(out_dir / "groups.npy", groups_arr)

    meta = {
        "files": [str(f) for f in files],
        "window": win,
        "stride": stride,
        "label_source": "state_timestamp",
        "focused_minutes": focused_minutes,
        "unfocused_minutes": unfocused_minutes,
        "signals_shape": signals_arr.shape,
        "label_counts": {int(k): int(v) for k, v in zip(*np.unique(labels_arr, return_counts=True))},
        "groups_saved": True,
    }
    with open(out_dir / "meta_state.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Timestamp-based labeling done. Label distribution:", meta["label_counts"])
    return meta


class EEGDataset(Dataset):
    """
    Loads signals.npy [N, C, T] and labels.npy [N]
    Returns x: [T, C]
    """
    def __init__(self, signals_path: Path, labels_path: Path, normalize: bool = True, groups_path: Path | None = None):
        self.signals = np.load(signals_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r").astype(np.int64)
        self.groups = None
        if groups_path is not None and groups_path.exists():
            self.groups = np.load(groups_path, mmap_mode="r").astype(np.int64)

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
        x = np.array(self.signals[real_idx], dtype=np.float32, copy=True)  # [C,T]
        if self.normalize:
            x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
        x = torch.from_numpy(x.T)  # [T,C]
        y = int(self.labels[real_idx] - self.label_offset)
        g = int(self.groups[real_idx]) if self.groups is not None else -1
        return x, torch.tensor(y, dtype=torch.long), torch.tensor(g, dtype=torch.long)


def create_loaders(dataset: EEGDataset, batch_size: int, num_workers: int,
                   split: tuple[float, float, float], seed: int):
    # NOTE: window-level random split (baseline-compatible). For paper, prefer group split (by file) externally.
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


# -------------------------
# Model: EEG-FusionCNN
# -------------------------
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=pad, dilation=dilation, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return self.drop(x)


class InceptionTemporalBlock(nn.Module):
    """
    Multi-scale temporal conv (InceptionTime-like) + residual.
    Input/Output: [B, D, T]
    """
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.b1 = DepthwiseSeparableConv1d(dim, dim, k=3, dilation=1, dropout=dropout)
        self.b2 = DepthwiseSeparableConv1d(dim, dim, k=7, dilation=1, dropout=dropout)
        self.b3 = DepthwiseSeparableConv1d(dim, dim, k=15, dilation=1, dropout=dropout)
        self.mix = nn.Conv1d(dim * 3, dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        y = self.mix(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.drop(y)
        return x + y


class EEGNetStem(nn.Module):
    """
    EEGNet-style stem to mix channels and temporal patterns.
    Input:  x [B,T,C]
    Output: f [B,D,T']
    """
    def __init__(self, in_ch: int, d_model: int, dropout: float, temporal_k: int = 64, pool: int = 4):
        super().__init__()
        # Use 2D to do (1,k) temporal then (C,1) spatial depthwise
        F1 = max(8, d_model // 8)
        D = 2
        F2 = d_model

        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, temporal_k), padding=(0, temporal_k // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(in_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=(1, pool)),
            nn.Dropout(dropout),
        )
        # Separable temporal conv
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=(1, pool)),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B,T,C] -> [B,1,C,T]
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)      # [B,*,1,T/4]
        x = self.separable(x)    # [B,D,1,T/16]
        x = x.squeeze(2)         # [B,D,T']
        return x


class AttentiveStatsPooling(nn.Module):
    """
    Attention-weighted mean+std pooling over time.
    Input:  [B,D,T]
    Output: [B,2D]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(dim, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: [B,D,T]
        w = self.att(x)              # [B,1,T]
        w = torch.softmax(w, dim=-1) # [B,1,T]
        mean = torch.sum(w * x, dim=-1)  # [B,D]
        var = torch.sum(w * (x - mean.unsqueeze(-1)) ** 2, dim=-1).clamp(min=1e-6)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=1)  # [B,2D]


class SpectrogramCNN(nn.Module):
    """
    STFT magnitude -> lightweight 2D CNN -> [B, Df]
    """
    def __init__(self, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
            nn.SiLU(),
        )

    def forward(self, mag):
        # mag: [B,1,F,Tf]
        return self.net(mag)


class EEGFusionCNN(nn.Module):
    """
    Hybrid CNN:
      - Time branch: EEGNetStem -> N x InceptionTemporalBlock -> AttentiveStatsPooling
      - Freq branch: STFT magnitude -> 2D CNN -> feature
      - Fuse -> head
    """
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 d_model: int = 128,
                 depth: int = 6,
                 dropout: float = 0.25,
                 use_freq: bool = True,
                 stft_n_fft: int = 128,
                 stft_hop: int = 64,
                 stft_win: int = 128):
        super().__init__()
        self.use_freq = use_freq

        self.stem = EEGNetStem(in_ch=input_channels, d_model=d_model, dropout=dropout, temporal_k=64, pool=4)
        self.blocks = nn.ModuleList([InceptionTemporalBlock(d_model, dropout=dropout) for _ in range(depth)])
        self.pool = AttentiveStatsPooling(d_model)

        self.stft_n_fft = stft_n_fft
        self.stft_hop = stft_hop
        self.stft_win = stft_win
        self.register_buffer("stft_window", torch.hann_window(stft_win), persistent=False)

        freq_dim = d_model // 2
        self.spec = SpectrogramCNN(out_dim=freq_dim, dropout=dropout) if use_freq else None

        fused_dim = (2 * d_model) + (freq_dim if use_freq else 0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def _stft_mag(self, x_bt_c: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,C] -> magnitude spectrogram averaged over channels -> [B,1,F,Tf]
        """
        B, T, C = x_bt_c.shape
        x = x_bt_c.transpose(1, 2).contiguous()  # [B,C,T]
        x = x.reshape(B * C, T)

        stft = torch.stft(
            x,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop,
            win_length=self.stft_win,
            window=self.stft_window.to(x.device),
            center=True,
            return_complex=True,
        )  # [B*C, F, Tf]

        mag = stft.abs()  # [B*C, F, Tf]
        mag = mag.reshape(B, C, mag.size(1), mag.size(2))  # [B,C,F,Tf]
        mag = mag.mean(dim=1)  # average over channels -> [B,F,Tf]
        mag = mag.unsqueeze(1) # [B,1,F,Tf]
        return mag

    def forward(self, x: torch.Tensor):
        # x: [B,T,C]
        # time branch
        f = self.stem(x)  # [B,D,T']
        for blk in self.blocks:
            f = blk(f)
        time_feat = self.pool(f)  # [B,2D]

        if self.use_freq:
            mag = self._stft_mag(x)
            freq_feat = self.spec(mag)  # [B,Df]
            feat = torch.cat([time_feat, freq_feat], dim=1)
        else:
            feat = time_feat

        return self.head(feat)


# -------------------------
# Train/Eval
# -------------------------
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], float(lam)


def loss_mixup(logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float, label_smoothing: float):
    if lam >= 1.0:
        return F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
    la = F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
    lb = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
    return lam * la + (1 - lam) * lb


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, device: torch.device, amp: bool,
                    grad_clip: float | None, mixup_alpha: float, label_smoothing: float):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb, _gb in loader:
        xb = xb.to(device, non_blocking=True)  # [B,T,C]
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            xb2, y_a, y_b, lam = mixup_batch(xb, yb, alpha=mixup_alpha)
            logits = model(xb2)
            loss = loss_mixup(logits, y_a, y_b, lam, label_smoothing=label_smoothing)

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

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool, label_smoothing: float):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    if len(loader) == 0:
        return 0.0, 0.0, np.array([]), np.array([])

    for xb, yb, _gb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
        y_true.append(yb.detach().cpu())
        y_pred.append(preds.detach().cpu())

    y_true = torch.cat(y_true).numpy() if y_true else np.array([])
    y_pred = torch.cat(y_pred).numpy() if y_pred else np.array([])
    return total_loss / max(total, 1), correct / max(total, 1), y_true, y_pred


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train and evaluate the EEG FusionCNN model.")
    ap.add_argument("--prepare", action="store_true", help="Only run preprocessing and exit.")
    ap.add_argument("--mat-root", type=Path, default=Path("EEGData"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/eeg_attention"))

    ap.add_argument("--label-source", choices=["field", "marker", "state"], default="state")
    ap.add_argument("--label-field", type=str, default=None)
    ap.add_argument("--marker-thresh", type=float, default=0.1)
    ap.add_argument("--focused-minutes", type=float, default=10.0)
    ap.add_argument("--unfocused-minutes", type=float, default=20.0)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256, help="default 50% overlap to increase training data")

    ap.add_argument("--signals", type=Path)
    ap.add_argument("--labels", type=Path)

    # training
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0)

    # model
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--no-freq", action="store_true", help="disable STFT+2D CNN branch")
    ap.add_argument("--stft-n-fft", type=int, default=128)
    ap.add_argument("--stft-hop", type=int, default=64)
    ap.add_argument("--stft-win", type=int, default=128)

    ap.add_argument("--num-classes", type=int, default=None)
    ap.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--save-dir", type=Path, default=Path("experiments/eeg_fusioncnn"))
    ap.add_argument("--no-normalize", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.prepare:
        if args.label_source == "marker":
            prepare_from_marker(args.mat_root, args.out_dir, args.window, args.stride, args.marker_thresh)
        elif args.label_source == "state":
            prepare_from_state(args.mat_root, args.out_dir, args.window, args.stride,
                               args.focused_minutes, args.unfocused_minutes)
        else:
            if run_preprocess is None:
                raise ImportError("src.data.preprocess.run_preprocess not found; use --label-source state/marker.")
            run_preprocess(
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
        return

    split = tuple(args.split)
    if not np.isclose(sum(split), 1.0):
        raise ValueError(f"Split ratios must sum to 1, got {split}")

    signals_path = args.signals or (args.out_dir / "signals.npy")
    labels_path = args.labels or (args.out_dir / "labels.npy")
    groups_path = args.out_dir / "groups.npy"
    if not signals_path.exists() or not labels_path.exists():
        raise FileNotFoundError("signals.npy / labels.npy not found; run with --prepare or set --signals/--labels.")

    dataset = EEGDataset(signals_path, labels_path, normalize=not args.no_normalize, groups_path=groups_path)
    num_classes = args.num_classes or dataset.num_classes

    train_loader, val_loader, test_loader = create_loaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=split,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.use_amp and device.type == "cuda")
    print(f"Cihaz: {device} | AMP: {amp_enabled}")
    print(f"Classes: {num_classes} | Channels: {dataset.num_channels} | SeqLen: {dataset.seq_len}")

    model = EEGFusionCNN(
        input_channels=dataset.num_channels,
        num_classes=num_classes,
        d_model=args.d_model,
        depth=args.depth,
        dropout=args.dropout,
        use_freq=not args.no_freq,
        stft_n_fft=args.stft_n_fft,
        stft_hop=args.stft_hop,
        stft_win=args.stft_win,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.cosine else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else torch.amp.GradScaler(enabled=False)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.save_dir / "best_eeg_fusioncnn.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, amp_enabled,
            grad_clip=args.grad_clip,
            mixup_alpha=args.mixup,
            label_smoothing=args.label_smoothing,
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, amp_enabled, label_smoothing=args.label_smoothing)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt_path)

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | lr {lr_now:.2e} | train {train_loss:.4f}/{train_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f}")

    elapsed = time.time() - start
    print(f"Training time: {elapsed/60:.1f} min")

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded best val checkpoint: {ckpt_path}")

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, device, amp_enabled, label_smoothing=args.label_smoothing)
    print(f"Test -> loss: {test_loss:.4f} | acc: {test_acc:.4f}")

    serializable_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    metrics = {
        "history": history,
        "best_val_acc": float(best_val),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "config": serializable_args,
    }
    with open(args.save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # plots
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

    # results.md
    try:
        lines = [
            "# EEG FusionCNN Results",
            "",
            f"- Best val acc: {best_val:.4f}",
            f"- Test loss: {test_loss:.4f}",
            f"- Test acc: {test_acc:.4f}",
            f"- Epochs: {len(history['train_loss'])}",
            f"- Checkpoint: {ckpt_path}",
            "",
            "## Figures",
            f"- Training curves: {args.save_dir / 'training_curves.png'}",
        ]
        if y_true.size > 0:
            lines.append(f"- Confusion matrix: {args.save_dir / 'confusion_matrix.png'}")
        (args.save_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        print(f"Writing results.md failed: {e}")


if __name__ == "__main__":
    main()
