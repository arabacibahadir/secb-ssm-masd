"""
Single-file EEG Conformer baseline.
Works with existing signals.npy / labels.npy; can also prepare data from .mat files.
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

# Optional: reuse existing preprocessing utilities if available
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
try:
    from src.data.preprocess import run_preprocess  # type: ignore
except Exception:
    run_preprocess = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_from_marker(mat_root: Path, out_dir: Path, win: int, stride: int, marker_thresh: float):
    """
    Build labels from o.marker: p(marker==1) >= marker_thresh => label 1 else 0.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    signals: list[np.ndarray] = []
    labels: list[int] = []

    files = sorted(mat_root.glob("eeg_record*.mat"))
    if not files:
        raise FileNotFoundError(f"No eeg_record*.mat under {mat_root}.")

    import scipy.io as sio  # local import; only needed during prepare

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
    - Label by timestamp: <= focused_minutes => class 0 (focused),
      > unfocused_minutes => class 2 (drowsy), else class 1 (unfocused).
    - Standardize per file, window with stride, label each window by majority.
    - Save signals [N, C, T] and labels [N].
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

        # timestamp is just sample index
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
            if lbl_window.size == 0:
                continue
            # majority vote
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


class EEGConformerDataset(Dataset):
    """
    Loads signals.npy [N, C, T] and labels.npy [N].
    Filters out negative labels and reindexes remaining to start from zero.
    """

    def __init__(self, signals_path: Path, labels_path: Path, normalize: bool = True):
        self.signals = np.load(signals_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r").astype(np.int64)

        if self.signals.shape[0] != self.labels.shape[0]:
            raise ValueError("signals.npy and labels.npy length mismatch.")

        # Filter negative labels
        self.indices = np.where(self.labels >= 0)[0]
        if len(self.indices) == 0:
            raise ValueError(
                "No valid labels (>=0) found. Re-run with --label-source marker or provide the correct "
                "label field via --label-field."
            )

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


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_expansion: int, conv_kernel: int, dropout: float):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, expansion=ff_expansion, dropout=dropout)
        self.ff2 = FeedForwardModule(dim, expansion=ff_expansion, dropout=dropout)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.conv = ConvolutionModule(dim, kernel_size=conv_kernel, dropout=dropout)
        self.norm_ff1 = nn.LayerNorm(dim)
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_conv = nn.LayerNorm(dim)
        self.norm_ff2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None):
        # Macaron style: 0.5 * FFN -> MHSA -> Conv -> 0.5 * FFN
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        attn_out, _ = self.attn(self.norm_attn(x), self.norm_attn(x), self.norm_attn(x),
                                key_padding_mask=padding_mask, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        return x


class EEGConformer(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, d_model: int = 128, num_layers: int = 4,
                 num_heads: int = 4, ff_expansion: int = 4, conv_kernel: int = 15, dropout: float = 0.1,
                 max_len: int = 4096):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, ff_expansion, conv_kernel, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None):
        # x: [B, T, C]
        x = self.embed(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        x = self.norm(x)
        pooled = x.mean(dim=1)  # global average pooling
        return self.head(pooled)


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return contextlib.nullcontext()


def create_loaders(dataset: EEGConformerDataset, batch_size: int, num_workers: int,
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
                    scaler: torch.amp.GradScaler, device: torch.device, amp: bool, grad_clip: float | None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool):
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
            loss = F.cross_entropy(logits, yb)
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
        y_true.append(yb.detach().cpu())
        y_pred.append(preds.detach().cpu())
    y_true = torch.cat(y_true).numpy() if y_true else np.array([])
    y_pred = torch.cat(y_pred).numpy() if y_pred else np.array([])
    return total_loss / max(total, 1), correct / max(total, 1), y_true, y_pred


def parse_args():
    ap = argparse.ArgumentParser(description="Train and evaluate the EEG Conformer baseline.")
    ap.add_argument("--prepare", action="store_true", help="Only run preprocessing and exit.")
    ap.add_argument("--mat-root", type=Path, default=Path("EEGData"), help="Folder containing eeg_record*.mat")
    ap.add_argument("--out-dir", type=Path, default=Path("data/eeg_attention"), help="Output folder for signals/labels npy")
    ap.add_argument("--label-source", choices=["field", "marker", "state"], default="field",
                    help="Label source: field -> attribute inside mat; marker -> o.marker ratio; state -> timestamp rule (focused/drowsy/unfocused)")
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
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff-expansion", type=int, default=4)
    ap.add_argument("--conv-kernel", type=int, default=15)
    ap.add_argument("--num-classes", type=int, default=None, help="Override class count; otherwise inferred from data")
    ap.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-amp", action="store_true", help="Enable AMP when CUDA is available")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--save-dir", type=Path, default=Path("experiments/eegconformer"))
    ap.add_argument("--no-normalize", action="store_true", help="Skip z-score in Dataset")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.prepare:
        if args.label_source == "marker":
            meta = prepare_from_marker(
                mat_root=args.mat_root,
                out_dir=args.out_dir,
                win=args.window,
                stride=args.stride,
                marker_thresh=args.marker_thresh,
            )
        elif args.label_source == "state":
            meta = prepare_from_state(
                mat_root=args.mat_root,
                out_dir=args.out_dir,
                win=args.window,
                stride=args.stride,
                focused_minutes=args.focused_minutes,
                unfocused_minutes=args.unfocused_minutes,
            )
        else:
            if run_preprocess is None:
                raise ImportError("src.data.preprocess.run_preprocess not found; install deps or use --label-source marker/state.")
            meta = run_preprocess(
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
            print("Prepared:", meta)
        return

    split = tuple(args.split)
    if not np.isclose(sum(split), 1.0):
        raise ValueError(f"Split ratios must sum to 1, got {split}")

    signals_path = args.signals or args.out_dir / "signals.npy"
    labels_path = args.labels or args.out_dir / "labels.npy"
    if not signals_path.exists() or not labels_path.exists():
        raise FileNotFoundError("signals.npy / labels.npy not found; run with --prepare or set --signals/--labels.")

    dataset = EEGConformerDataset(signals_path, labels_path, normalize=not args.no_normalize)
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

    model = EEGConformer(
        input_dim=dataset.num_channels,
        num_classes=num_classes,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        ff_expansion=args.ff_expansion,
        conv_kernel=args.conv_kernel,
        dropout=args.dropout,
        max_len=dataset.seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else torch.amp.GradScaler(enabled=False)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.save_dir / "best_eegconformer.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, amp_enabled, args.grad_clip)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, amp_enabled)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt_path)

        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f}")

    elapsed = time.time() - start
    print(f"Training time: {elapsed/60:.1f} min")

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded best val checkpoint: {ckpt_path}")

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, device, amp_enabled)
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

    # Plots
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
            "# EEG Conformer Results",
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
