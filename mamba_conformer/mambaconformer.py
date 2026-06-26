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

# Optional: reuse existing preprocessing utilities if available (same pattern as baseline.py)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
try:
    from src.data.preprocess import run_preprocess  # type: ignore
except Exception:
    run_preprocess = None


# -------------------------
# Utilities
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
# Data preparation (same as baseline.py)
# -------------------------
def prepare_from_marker(mat_root: Path, out_dir: Path, win: int, stride: int, marker_thresh: float):
    """
    Build labels from o.marker: p(marker==1) >= marker_thresh => label 1 else 0.
    Saves: signals.npy [N, C, T], labels.npy [N]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    signals: list[np.ndarray] = []
    labels: list[int] = []

    files = sorted(mat_root.glob("eeg_record*.mat"))
    if not files:
        raise FileNotFoundError(f"No eeg_record*.mat under {mat_root}.")

    import scipy.io as sio  # local import

    for f in files:
        mat = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
        if "o" not in mat:
            raise ValueError(f"'o' key missing in {f}.")
        o = mat["o"]
        data = np.asarray(o.data, dtype=np.float32)
        marker = np.asarray(o.marker)
        if marker.shape[0] != data.shape[0]:
            raise ValueError(f"marker length mismatch with data in {f}")
        eeg = data[:, 3:17]  # 14 channels

        # Channel-wise z-score
        eeg = (eeg - eeg.mean(axis=0, keepdims=True)) / (eeg.std(axis=0, keepdims=True) + 1e-6)

        total = eeg.shape[0]
        for start in range(0, total - win + 1, stride):
            window = eeg[start:start + win]  # [win, C]
            window = window.T  # [C, win]
            mwin = marker[start:start + win]
            ratio = float((mwin == 1).mean())
            lbl = 1 if ratio >= marker_thresh else 0
            signals.append(window.astype(np.float32))
            labels.append(lbl)

    signals_arr = np.stack(signals)
    labels_arr = np.array(labels, dtype=np.int64)
    np.save(out_dir / "signals.npy", signals_arr)
    np.save(out_dir / "labels.npy", labels_arr)

    meta = {
        "files": [str(f) for f in files],
        "window": win,
        "stride": stride,
        "label_source": "marker",
        "marker_thresh": marker_thresh,
        "signals_shape": signals_arr.shape,
        "label_counts": {int(k): int(v) for k, v in zip(*np.unique(labels_arr, return_counts=True))},
    }
    with open(out_dir / "meta_marker.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Marker-based labeling done. Label distribution:", meta["label_counts"])
    return meta


def prepare_from_state(mat_root: Path, out_dir: Path, win: int, stride: int,
                       focused_minutes: float, unfocused_minutes: float, fs: int = 128):
    """
    Timestamp-derived state labeling:
    - timestamp <= focused_minutes => class 0 (focused)
    - timestamp > unfocused_minutes => class 2 (drowsy)
    - else class 1 (unfocused)
    Window label = majority vote inside window.
    Saves: signals.npy [N, C, T], labels.npy [N]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    signals: list[np.ndarray] = []
    labels: list[int] = []

    files = sorted(mat_root.glob("eeg_record*.mat"))
    if not files:
        raise FileNotFoundError(f"No eeg_record*.mat under {mat_root}.")

    import scipy.io as sio  # local import

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

        # standardize per file per channel
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
    Returns x: [T, C] float32, y: long
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
        x = torch.from_numpy(x.T)  # [T, C]
        y = int(self.labels[real_idx] - self.label_offset)
        return x, torch.tensor(y, dtype=torch.long)


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


# -------------------------
# Model blocks
# -------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-np.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        return x + self.pe[: x.size(1), :].unsqueeze(0)


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class ConvolutionModule(nn.Module):
    """
    Conformer-like conv module (depthwise conv) on [B,T,D].
    """
    def __init__(self, dim: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.pw_conv1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.pw_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pw_conv1(x)
        x = self.glu(x)
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = self.drop(x)
        return x.transpose(1, 2)  # [B, T, D]


class SelectiveSSMLite(nn.Module):
    """
    Dependency-free, Mamba-like selective SSM (Lite).
    - Per time-step input-dependent decay gate.
    - Recurrent state update (linear-time).
    - Compact selective state-space sequence mixer for EEG windows.

    x: [B,T,D] -> y: [B,T,D]
    """
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim * 2)     # gives u and gate
        self.decay_proj = nn.Linear(dim, dim)      # gives per-step decay
        self.out_proj = nn.Linear(dim, dim)
        self.skip = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B,T,D]
        u_gate = self.in_proj(x)
        u, gate = u_gate.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        decay = torch.sigmoid(self.decay_proj(x))  # [B,T,D] in (0,1)

        B, T, D = x.shape
        s = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        ys = []

        # NOTE: T is typically 256-512; this loop is OK for EEG windows.
        for t in range(T):
            s = decay[:, t] * s + u[:, t]
            ys.append(s)

        y = torch.stack(ys, dim=1)  # [B,T,D]
        y = self.out_proj(y)
        y = self.drop(y)
        return (y * gate) + self.skip(x)


class BiSelectiveSSM(nn.Module):
    """
    Bidirectional wrapper: forward SSM + backward SSM, then fuse.
    """
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fwd = SelectiveSSMLite(dim, dropout=dropout)
        self.bwd = SelectiveSSMLite(dim, dropout=dropout)
        self.fuse = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor):
        # x: [B,T,D]
        y_fwd = self.fwd(x)
        y_bwd = self.bwd(torch.flip(x, dims=[1]))
        y_bwd = torch.flip(y_bwd, dims=[1])
        y = torch.cat([y_fwd, y_bwd], dim=-1)
        return self.fuse(y)


class MambaOrLite(nn.Module):
    """
    Dependency-free selective SSM block (BiSelectiveSSM).
    The previous optional mamba-ssm import is removed to avoid build issues.
    """
    def __init__(self, dim: int, dropout: float = 0.0, bidir: bool = True):
        super().__init__()
        self.block = BiSelectiveSSM(dim, dropout=dropout) if bidir else SelectiveSSMLite(dim, dropout=dropout)

    def forward(self, x: torch.Tensor):
        return self.block(x)


class MambaConformerBlock(nn.Module):
    """
    Macaron-style:
      x + 0.5*FFN -> x + SSM(Mamba/Lite) -> x + Conv -> x + 0.5*FFN
    """
    def __init__(self, dim: int, ff_expansion: int, conv_kernel: int, dropout: float,
                 bidir_ssm: bool):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, expansion=ff_expansion, dropout=dropout)
        self.ff2 = FeedForwardModule(dim, expansion=ff_expansion, dropout=dropout)
        self.ssm = MambaOrLite(dim, dropout=dropout, bidir=bidir_ssm)
        self.conv = ConvolutionModule(dim, kernel_size=conv_kernel, dropout=dropout)

        self.norm_ff1 = nn.LayerNorm(dim)
        self.norm_ssm = nn.LayerNorm(dim)
        self.norm_conv = nn.LayerNorm(dim)
        self.norm_ff2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        x = x + self.drop(self.ssm(self.norm_ssm(x)))
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        return x


class TemporalConvEmbedding(nn.Module):
    """
    Strided temporal conv as patch/embedding:
      Input: [B,T,C] -> conv over time with stride -> [B,T',D]
    """
    def __init__(self, in_ch: int, d_model: int, kernel_size: int, stride: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=kernel_size, stride=stride, padding=pad)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B,T,C] -> [B,C,T]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        # [B,D,T'] -> [B,T',D]
        return x.transpose(1, 2)


class EEGMambaConformer(nn.Module):
    """
    End-to-end classifier:
      - temporal conv embedding (downsample time)
      - positional encoding
      - N x MambaConformerBlock
      - mean pool
      - linear head
    """
    def __init__(self, input_dim: int, num_classes: int,
                 d_model: int = 192,
                 num_layers: int = 6,
                 ff_expansion: int = 4,
                 conv_kernel: int = 31,
                 dropout: float = 0.15,
                 emb_kernel: int = 7,
                 emb_stride: int = 2,
                 bidir_ssm: bool = True,
                 max_len: int = 4096):
        super().__init__()
        self.embed = TemporalConvEmbedding(input_dim, d_model, kernel_size=emb_kernel, stride=emb_stride, dropout=dropout)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            MambaConformerBlock(
                dim=d_model,
                ff_expansion=ff_expansion,
                conv_kernel=conv_kernel,
                dropout=dropout,
                bidir_ssm=bidir_ssm,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x: torch.Tensor):
        # x: [B,T,C]
        x = self.embed(x)     # [B,T',D]
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)


# -------------------------
# Training
# -------------------------
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """
    Standard mixup for classification.
    Returns mixed_x, y_a, y_b, lam
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, float(lam)


def loss_with_optional_mixup(logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor,
                            lam: float, label_smoothing: float):
    if lam >= 1.0:
        return F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
    loss_a = F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
    loss_b = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
    return lam * loss_a + (1 - lam) * loss_b


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, device: torch.device, amp: bool,
                    grad_clip: float | None, mixup_alpha: float, label_smoothing: float,
                    log_every: int, epoch: int, epochs: int, global_start_time: float):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    steps = len(loader)
    t0 = time.time()

    for step, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device, non_blocking=True)  # [B,T,C]
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp):
            xb2, y_a, y_b, lam = mixup_batch(xb, yb, alpha=mixup_alpha)
            logits = model(xb2)
            loss = loss_with_optional_mixup(logits, y_a, y_b, lam, label_smoothing=label_smoothing)

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

        # Training accuracy is tracked against original labels when mixup is enabled.
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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool, label_smoothing: float):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    if len(loader) == 0:
        return 0.0, 0.0, np.array([]), np.array([])

    for xb, yb in loader:
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
    ap = argparse.ArgumentParser(description="Train and evaluate the EEG MambaConformer model.")
    ap.add_argument("--prepare", action="store_true", help="Only run preprocessing and exit.")
    ap.add_argument("--mat-root", type=Path, default=Path("EEGData"), help="Folder containing eeg_record*.mat")
    ap.add_argument("--out-dir", type=Path, default=Path("data/eeg_attention"), help="Output folder for signals/labels npy")

    ap.add_argument("--label-source", choices=["field", "marker", "state"], default="state",
                    help="Label source: field -> attribute in mat; marker -> o.marker ratio; state -> timestamp rule.")
    ap.add_argument("--label-field", type=str, default=None, help="label-source=field attribute name (e.g., attention_state)")
    ap.add_argument("--marker-thresh", type=float, default=0.1, help="label-source=marker threshold on marker==1 ratio")
    ap.add_argument("--focused-minutes", type=float, default=10.0, help="label-source=state: <= minutes => focused (0)")
    ap.add_argument("--unfocused-minutes", type=float, default=20.0, help="label-source=state: > minutes => unfocused (1), else 2")
    ap.add_argument("--window", type=int, default=512, help="Window length in samples (default 4s @128Hz)")
    ap.add_argument("--stride", type=int, default=256, help="Stride in samples (default 50%% overlap for more data)")

    ap.add_argument("--signals", type=Path, help="Existing signals.npy path (default out_dir/signals.npy)")
    ap.add_argument("--labels", type=Path, help="Existing labels.npy path (default out_dir/labels.npy)")

    # Training
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4, was 3e-4)")
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--mixup", type=float, default=0.2, help="Mixup alpha (0 disables).")
    ap.add_argument("--log-every", type=int, default=0, help="Print detailed training metrics every N batches; 0 disables logging.")

    # Model
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ff-expansion", type=int, default=4)
    ap.add_argument("--conv-kernel", type=int, default=31)
    ap.add_argument("--emb-kernel", type=int, default=7)
    ap.add_argument("--emb-stride", type=int, default=2)
    ap.add_argument("--bidir-ssm", action="store_true", help="Use bidirectional SSM blocks.")

    ap.add_argument("--num-classes", type=int, default=None, help="Override class count; otherwise inferred from data")
    ap.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader worker processes.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-amp", action="store_true", help="Enable AMP when CUDA is available")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--save-dir", type=Path, default=Path("experiments/eeg_mambaconformer"))
    ap.add_argument("--no-normalize", action="store_true", help="Skip z-score normalization in Dataset.")
    ap.add_argument("--cosine", action="store_true", help="Use cosine LR schedule.")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.prepare:
        if args.label_source == "marker":
            prepare_from_marker(
                mat_root=args.mat_root, out_dir=args.out_dir,
                win=args.window, stride=args.stride, marker_thresh=args.marker_thresh
            )
        elif args.label_source == "state":
            prepare_from_state(
                mat_root=args.mat_root, out_dir=args.out_dir,
                win=args.window, stride=args.stride,
                focused_minutes=args.focused_minutes, unfocused_minutes=args.unfocused_minutes
            )
        else:
            if run_preprocess is None:
                raise ImportError("src.data.preprocess.run_preprocess not found; use --label-source state/marker.")
            run_preprocess(
                mat_root=args.mat_root, out_dir=args.out_dir,
                win=args.window, stride=args.stride,
                label_field=args.label_field,
                bandpass=False, cache_graphs=False,
                thresh=0.5, topk=None,
            )
        return

    split = tuple(args.split)
    if not np.isclose(sum(split), 1.0):
        raise ValueError(f"Split ratios must sum to 1, got {split}")

    signals_path = args.signals or args.out_dir / "signals.npy"
    labels_path = args.labels or args.out_dir / "labels.npy"
    if not signals_path.exists() or not labels_path.exists():
        raise FileNotFoundError("signals.npy / labels.npy not found; run with --prepare or set --signals/--labels.")

    dataset = EEGDataset(signals_path, labels_path, normalize=not args.no_normalize)
    num_classes = args.num_classes or dataset.num_classes

    train_loader, val_loader, test_loader = create_loaders(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        split=split, seed=args.seed
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
    
    amp_enabled = bool(args.use_amp and device.type == "cuda")
    print(f"Device: {device} | AMP: {amp_enabled} | num_workers: {args.num_workers}")
    print(f"Classes: {num_classes} | Channels: {dataset.num_channels} | SeqLen: {dataset.seq_len}")

    model = EEGMambaConformer(
        input_dim=dataset.num_channels,
        num_classes=num_classes,
        d_model=args.d_model,
        num_layers=args.layers,
        ff_expansion=args.ff_expansion,
        conv_kernel=args.conv_kernel,
        dropout=args.dropout,
        emb_kernel=args.emb_kernel,
        emb_stride=args.emb_stride,
        bidir_ssm=bool(args.bidir_ssm),
        max_len=max(4096, dataset.seq_len),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else torch.amp.GradScaler(enabled=False)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.save_dir / "best_eeg_mambaconformer.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val = 0.0
    global_start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, amp_enabled,
            grad_clip=args.grad_clip,
            mixup_alpha=args.mixup,
            label_smoothing=args.label_smoothing,
            log_every=args.log_every,
            epoch=epoch,
            epochs=args.epochs,
            global_start_time=global_start_time,
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

    elapsed = time.time() - global_start_time
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

    # Plots (same style as baseline.py)
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

    # Results markdown
    try:
        lines = [
            "# EEG MambaConformer Results",
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
