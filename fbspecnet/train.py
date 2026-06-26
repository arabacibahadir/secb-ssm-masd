#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG-FBSpecNet model.
Hybrid CNN:
  Time branch: Learnable FilterBank (Sinc-like) -> per-band spatial conv -> Dilated Res-TCN -> Attentive Stats Pool
  Freq branch: Per-channel STFT magnitude -> 2D CNN (in_channels = C) -> feature
Fusion: gated concat -> MLP head

CLI: --prepare like baseline.py, training + metrics + plots.
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
# Prepare (same logic as baseline.py)
# -------------------------
def prepare_from_state(mat_root: Path, out_dir: Path, win: int, stride: int,
                       focused_minutes: float, unfocused_minutes: float, fs: int = 128):
    out_dir.mkdir(parents=True, exist_ok=True)
    signals, labels, groups = [], [], []

    files = sorted(mat_root.glob("eeg_record*.mat"))
    if not files:
        raise FileNotFoundError(f"No eeg_record*.mat under {mat_root}.")

    import scipy.io as sio

    focused_limit = focused_minutes * 60 * fs
    unfocused_limit = unfocused_minutes * 60 * fs

    for fid, f in enumerate(files):
        mat = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
        if "o" not in mat:
            raise ValueError(f"'o' key missing in {f}.")
        data = np.asarray(mat["o"].data, dtype=np.float32)  # [T, 25]
        eeg = data[:, 3:17]  # 14 ch

        timestamp = np.arange(eeg.shape[0])
        state = np.full_like(timestamp, 1, dtype=np.int64)  # default unfocused
        state[timestamp <= focused_limit] = 0
        state[timestamp > unfocused_limit] = 2

        scaler = StandardScaler(with_mean=True, with_std=True)
        eeg = scaler.fit_transform(eeg).astype(np.float32)

        total = eeg.shape[0]
        for start in range(0, total - win + 1, stride):
            window = eeg[start:start + win]          # [win, C]
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
    Loads signals.npy [N,C,T], labels.npy [N].
    Returns x: [T,C], y, group_id
    """
    def __init__(self, signals_path: Path, labels_path: Path, groups_path: Path | None, normalize: bool):
        self.signals = np.load(signals_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r").astype(np.int64)
        self.groups = np.load(groups_path, mmap_mode="r").astype(np.int64) if (groups_path and groups_path.exists()) else None

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
        ridx = int(self.indices[idx])
        x = np.array(self.signals[ridx], dtype=np.float32, copy=True)  # [C,T]
        if self.normalize:
            x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
        x = torch.from_numpy(x.T)  # [T,C]
        y = int(self.labels[ridx] - self.label_offset)
        g = int(self.groups[ridx]) if self.groups is not None else -1
        return x, torch.tensor(y, dtype=torch.long), torch.tensor(g, dtype=torch.long)


def create_loaders(dataset: EEGDataset, batch_size: int, num_workers: int,
                   split: tuple[float, float, float], seed: int):
    n_total = len(dataset)
    n_train = int(split[0] * n_total)
    n_val = int(split[1] * n_total)
    n_test = n_total - n_train - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader, train_set


# -------------------------
# Loss
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.tensor([]), persistent=False)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        ce = F.cross_entropy(
            logits, target,
            weight=(self.weight if self.weight.numel() else None),
            reduction="none",
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def compute_class_weights_from_subset(subset, num_classes: int, device: torch.device):
    ys = []
    for i in range(len(subset)):
        _x, y, _g = subset[i]
        ys.append(int(y))
    y = torch.tensor(ys, dtype=torch.long)
    counts = torch.bincount(y, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    w = counts.sum() / (num_classes * counts)
    return w.to(device)


# -------------------------
# Augmentations (cheap, effective)
# -------------------------
def augment_batch(x: torch.Tensor, p_shift: float, max_shift: int,
                  p_chdrop: float, chdrop_frac: float,
                  noise_std: float):
    # x: [B,T,C]
    B, T, C = x.shape

    # time shift
    if p_shift > 0 and random.random() < p_shift:
        s = random.randint(-max_shift, max_shift)
        x = torch.roll(x, shifts=s, dims=1)

    # channel dropout
    if p_chdrop > 0 and random.random() < p_chdrop and chdrop_frac > 0:
        k = max(1, int(C * chdrop_frac))
        idx = torch.randperm(C, device=x.device)[:k]
        x[:, :, idx] = 0.0

    # gaussian noise
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)

    return x


# -------------------------
# Model blocks
# -------------------------
class LearnableSincFilterBank(nn.Module):
    """
    Learnable bandpass filterbank (Sinc-style), shared across channels.
    Produces K band-filtered versions per channel using grouped Conv1d.

    Input:  x [B,C,T]
    Output: y [B, C*K, T]
    """
    def __init__(self, channels: int, K: int, kernel_size: int, fs: float):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.C = channels
        self.K = K
        self.ks = kernel_size
        self.fs = fs

        # initialize bands roughly EEG-like: 1..40 Hz split into K bands
        low = torch.linspace(1.0, 30.0, K)
        band = torch.linspace(3.0, 10.0, K)
        self.low_hz = nn.Parameter(low)   # [K]
        self.band_hz = nn.Parameter(band) # [K]

        n = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float()
        self.register_buffer("n", n, persistent=False)
        window = torch.hamming_window(kernel_size, periodic=False)
        self.register_buffer("window", window, persistent=False)

    def _sinc(self, x):
        return torch.where(x == 0, torch.ones_like(x), torch.sin(np.pi * x) / (np.pi * x))

    def build_filters(self, device, dtype):
        low = torch.abs(self.low_hz) + 0.5
        band = torch.abs(self.band_hz) + 1.0
        high = torch.clamp(low + band, max=self.fs / 2 - 1.0)

        t = self.n.to(device=device, dtype=dtype) / self.fs  # seconds
        # bandpass = 2*high*sinc(2*high*t) - 2*low*sinc(2*low*t)
        band_pass = (2 * high[:, None] * self._sinc(2 * high[:, None] * t[None, :]) -
                     2 * low[:, None] * self._sinc(2 * low[:, None] * t[None, :]))
        band_pass = band_pass * self.window.to(device=device, dtype=dtype)[None, :]
        band_pass = band_pass / (band_pass.abs().sum(dim=1, keepdim=True) + 1e-6)  # normalize L1
        return band_pass  # [K, ks]

    def forward(self, x):
        # x: [B,C,T]
        B, C, T = x.shape
        assert C == self.C
        filt = self.build_filters(x.device, x.dtype)  # [K,ks]
        # replicate for grouped conv: weight [C*K, 1, ks]
        w = filt[:, None, :].repeat(C, 1, 1)  # [C*K,1,ks] (same K for each channel)
        y = F.conv1d(x, w, bias=None, stride=1, padding=self.ks // 2, groups=C)  # [B,C*K,T]
        return y


class SpatialBandConv(nn.Module):
    """
    After filterbank, reshape [B,C*K,T] -> [B,K,C,T], do per-band spatial conv across channels.
    Output: [B, K*D, T]
    """
    def __init__(self, C: int, K: int, D: int, dropout: float):
        super().__init__()
        self.C, self.K, self.D = C, K, D
        self.spatial = nn.Conv2d(K, K * D, kernel_size=(C, 1), groups=K, bias=False)
        self.bn = nn.BatchNorm2d(K * D)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C*K, T]
        B, CK, T = x.shape
        x = x.view(B, self.K, self.C, T)     # [B,K,C,T]
        x = self.spatial(x)                  # [B,K*D,1,T]
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x.squeeze(2)                  # [B, K*D, T]


class DWSeparableConv1d(nn.Module):
    def __init__(self, ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv1d(ch, ch, kernel_size=k, padding=pad, dilation=dilation, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return self.drop(x)


class ResTCNBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = DWSeparableConv1d(ch, k=k, dilation=dilation, dropout=dropout)
        self.conv2 = DWSeparableConv1d(ch, k=k, dilation=1, dropout=dropout)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class AttentiveStatsPool(nn.Module):
    """
    x [B,D,T] -> [B,2D]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.Tanh(),
            nn.Conv1d(dim, 1, 1),
        )

    def forward(self, x):
        w = torch.softmax(self.att(x), dim=-1)  # [B,1,T]
        mean = torch.sum(w * x, dim=-1)         # [B,D]
        var = torch.sum(w * (x - mean.unsqueeze(-1)) ** 2, dim=-1).clamp(min=1e-6)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=1)


class SpecCNN(nn.Module):
    """
    Input spectrogram: [B,C,F,Tf] as 2D with in_channels=C.
    Output: [B, Df]
    """
    def __init__(self, in_ch: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class EEGFBSpecNet(nn.Module):
    def __init__(self,
                 C: int,
                 num_classes: int,
                 fs: int = 128,
                 K: int = 8,
                 sinc_k: int = 129,
                 spatial_D: int = 2,
                 tcn_k: int = 3,
                 tcn_depth: int = 8,
                 dropout: float = 0.25,
                 use_freq: bool = True,
                 stft_n_fft: int = 128,
                 stft_hop: int = 64,
                 stft_win: int = 128,
                 spec_dim: int = 128):
        super().__init__()
        self.use_freq = use_freq
        self.fs = fs

        # Time branch
        self.fb = LearnableSincFilterBank(channels=C, K=K, kernel_size=sinc_k, fs=float(fs))
        self.spatial = SpatialBandConv(C=C, K=K, D=spatial_D, dropout=dropout)

        D_time = K * spatial_D
        self.tcn = nn.ModuleList([
            ResTCNBlock(D_time, k=tcn_k, dilation=(2 ** (i % 4)), dropout=dropout)
            for i in range(tcn_depth)
        ])
        self.pool = AttentiveStatsPool(D_time)

        # Freq branch
        self.stft_n_fft = stft_n_fft
        self.stft_hop = stft_hop
        self.stft_win = stft_win
        self.register_buffer("stft_window", torch.hann_window(stft_win), persistent=False)
        self.spec = SpecCNN(in_ch=C, out_dim=spec_dim, dropout=dropout) if use_freq else None

        # Fusion
        time_dim = 2 * D_time
        fused_dim = time_dim + (spec_dim if use_freq else 0)

        self.gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.SiLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def _spec(self, x_bt_c: torch.Tensor):
        # x: [B,T,C] -> STFT per channel -> [B,C,F,Tf]
        B, T, C = x_bt_c.shape
        x = x_bt_c.transpose(1, 2).contiguous().reshape(B * C, T)  # [B*C, T]
        stft = torch.stft(
            x,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop,
            win_length=self.stft_win,
            window=self.stft_window.to(x.device),
            center=True,
            return_complex=True,
        )  # [B*C, F, Tf]
        mag = stft.abs()
        mag = mag.reshape(B, C, mag.size(1), mag.size(2))  # [B,C,F,Tf]
        return mag

    def forward(self, x_bt_c: torch.Tensor):
        # x: [B,T,C] -> time branch expects [B,C,T]
        x_bct = x_bt_c.transpose(1, 2)

        y = self.fb(x_bct)              # [B,C*K,T]
        y = self.spatial(y)             # [B,K*D,T]
        for blk in self.tcn:
            y = blk(y)
        time_feat = self.pool(y)        # [B,2*(K*D)]

        if self.use_freq:
            mag = self._spec(x_bt_c)    # [B,C,F,Tf]
            spec_feat = self.spec(mag)  # [B,spec_dim]
            feat = torch.cat([time_feat, spec_feat], dim=1)
        else:
            feat = time_feat

        g = self.gate(feat)
        feat = feat * g
        return self.head(feat)


# -------------------------
# Train/Eval
# -------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, amp,
                    grad_clip, loss_fn, aug_cfg, log_every: int,
                    epoch: int, epochs: int, global_start_time: float):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    steps = max(1, len(loader))
    t0 = time.time()

    for step, (xb, yb, _gb) in enumerate(loader, start=1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        xb = augment_batch(
            xb,
            p_shift=aug_cfg["p_shift"],
            max_shift=aug_cfg["max_shift"],
            p_chdrop=aug_cfg["p_chdrop"],
            chdrop_frac=aug_cfg["chdrop_frac"],
            noise_std=aug_cfg["noise_std"],
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(xb)
            loss = loss_fn(logits, yb)

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

        bs = xb.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(dim=-1)
        correct += (preds == yb).sum().item()
        total += bs

        if log_every > 0 and (step == 1 or step % log_every == 0 or step == steps):
            elapsed = time.time() - t0
            avg_loss = total_loss / max(total, 1)
            avg_acc = correct / max(total, 1)
            lr_now = optimizer.param_groups[0]["lr"]
            it_s = step / max(elapsed, 1e-9)
            samp_s = total / max(elapsed, 1e-9)
            eta = (steps - step) / max(it_s, 1e-9)

            # Approximate total ETA based on the average epoch duration.
            all_elapsed = time.time() - global_start_time
            avg_epoch_time = all_elapsed / max(epoch, 1)
            total_eta = avg_epoch_time * (epochs - epoch)

            print(
                f"[Epoch {epoch:03d}/{epochs}] "
                f"Step {step:04d}/{steps} | "
                f"loss {avg_loss:.4f} acc {avg_acc:.4f} | "
                f"lr {lr_now:.2e} | "
                f"elapsed {elapsed:6.1f}s ETA(step) {eta:6.1f}s | "
                f"speed {samp_s:7.1f} samp/s | ETA(total) {total_eta/60:5.1f}m",
                flush=True
            )

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp, loss_fn_eval):
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
            loss = loss_fn_eval(logits, yb)

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
    ap = argparse.ArgumentParser(description="Train and evaluate the EEG FBSpecNet model.")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--mat-root", type=Path, default=Path("EEGData"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/eeg_attention"))

    ap.add_argument("--label-source", choices=["field", "marker", "state"], default="state")
    ap.add_argument("--label-field", type=str, default=None)
    ap.add_argument("--focused-minutes", type=float, default=10.0)
    ap.add_argument("--unfocused-minutes", type=float, default=20.0)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)  # more data by default

    ap.add_argument("--signals", type=Path)
    ap.add_argument("--labels", type=Path)

    # train
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    ap.add_argument("--save-dir", type=Path, default=Path("experiments/eeg_fbspecnet"))
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--no-normalize", action="store_true", help="Skip z-score normalization in Dataset.")

    # loss
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--label-smoothing", type=float, default=0.03)

    # augment
    ap.add_argument("--p-shift", type=float, default=0.5)
    ap.add_argument("--max-shift", type=int, default=24)      # samples
    ap.add_argument("--p-chdrop", type=float, default=0.3)
    ap.add_argument("--chdrop-frac", type=float, default=0.15)
    ap.add_argument("--noise-std", type=float, default=0.01)

    # model
    ap.add_argument("--fs", type=int, default=128)
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--sinc-k", type=int, default=129)
    ap.add_argument("--spatial-d", type=int, default=2)
    ap.add_argument("--tcn-depth", type=int, default=10)
    ap.add_argument("--tcn-k", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.25)

    ap.add_argument("--no-freq", action="store_true")
    ap.add_argument("--stft-n-fft", type=int, default=128)
    ap.add_argument("--stft-hop", type=int, default=32)
    ap.add_argument("--stft-win", type=int, default=128)
    ap.add_argument("--spec-dim", type=int, default=128)
    ap.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print loss, accuracy, learning rate, elapsed time, and ETA every N training batches.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.prepare:
        if args.label_source == "state":
            prepare_from_state(
                mat_root=args.mat_root,
                out_dir=args.out_dir,
                win=args.window,
                stride=args.stride,
                focused_minutes=args.focused_minutes,
                unfocused_minutes=args.unfocused_minutes,
                fs=args.fs,
            )
        else:
            if run_preprocess is None:
                raise ImportError("src.data.preprocess.run_preprocess not found; use --label-source state.")
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
    global_start_time = time.time()
    split = tuple(args.split)
    if not np.isclose(sum(split), 1.0):
        raise ValueError("Split must sum to 1.0")

    signals_path = args.signals or (args.out_dir / "signals.npy")
    labels_path = args.labels or (args.out_dir / "labels.npy")
    groups_path = args.out_dir / "groups.npy"

    if not signals_path.exists() or not labels_path.exists():
        raise FileNotFoundError("signals.npy / labels.npy not found; run --prepare first.")

    dataset = EEGDataset(
        signals_path, labels_path, groups_path=groups_path,
        normalize=not args.no_normalize
    )
    num_classes = dataset.num_classes

    train_loader, val_loader, test_loader, train_subset = create_loaders(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        split=split, seed=args.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.use_amp and device.type == "cuda")
    print(f"Cihaz: {device} | AMP: {amp_enabled}")
    print(f"Classes: {num_classes} | Channels: {dataset.num_channels} | SeqLen: {dataset.seq_len}")

    model = EEGFBSpecNet(
        C=dataset.num_channels,
        num_classes=num_classes,
        fs=args.fs,
        K=args.bands,
        sinc_k=args.sinc_k,
        spatial_D=args.spatial_d,
        tcn_k=args.tcn_k,
        tcn_depth=args.tcn_depth,
        dropout=args.dropout,
        use_freq=not args.no_freq,
        stft_n_fft=args.stft_n_fft,
        stft_hop=args.stft_hop,
        stft_win=args.stft_win,
        spec_dim=args.spec_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.cosine else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else torch.amp.GradScaler(enabled=False)

    class_w = None
    if args.class_weights:
        class_w = compute_class_weights_from_subset(train_subset, num_classes, device)

    if args.focal:
        loss_fn = FocalLoss(gamma=args.focal_gamma, weight=class_w, label_smoothing=args.label_smoothing)
        loss_fn_eval = lambda logits, y: F.cross_entropy(logits, y, weight=class_w, label_smoothing=args.label_smoothing)
    else:
        loss_fn = lambda logits, y: F.cross_entropy(logits, y, weight=class_w, label_smoothing=args.label_smoothing)
        loss_fn_eval = loss_fn

    aug_cfg = dict(
        p_shift=args.p_shift, max_shift=args.max_shift,
        p_chdrop=args.p_chdrop, chdrop_frac=args.chdrop_frac,
        noise_std=args.noise_std,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.save_dir / "best_eeg_fbspecnet.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
    model, train_loader, optimizer, scaler, device, amp_enabled,
    grad_clip=args.grad_clip, loss_fn=loss_fn, aug_cfg=aug_cfg,
    log_every=args.log_every, epoch=epoch, epochs=args.epochs,
    global_start_time=global_start_time
)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, amp_enabled, loss_fn_eval)

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

    print(f"Training time: {(time.time()-start)/60:.1f} min")

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded best val checkpoint: {ckpt_path}")

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, device, amp_enabled, loss_fn_eval)
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

    try:
        lines = [
            "# EEG FBSpecNet Results",
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
