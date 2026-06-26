"""
Full ST-MGM implementation using Mamba core from https://github.com/state-spaces/mamba.
Features:
- Data preparation (state/marker labels, windowing).
- Dynamic graph construction (corr + top-k).
- Masked pretraining (optional) and finetuning for classification.
- Outputs metrics, curves, confusion matrix, summary, checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import scipy.io as sio

# Mamba imports
try:
    from mamba_ssm.modules.mamba_simple import MambaBlock
    HAS_MAMBA = True
except Exception:
    HAS_MAMBA = False
    MambaBlock = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Data preparation helpers
# -------------------------
def prepare_state(mat_root: Path, out_dir: Path, win: int, stride: int,
                  focused_minutes: float, unfocused_minutes: float, fs: int = 128):
    out_dir.mkdir(parents=True, exist_ok=True)
    signals, labels = [], []
    files = sorted(mat_root.glob("eeg_record*.mat"))
    if not files:
        raise FileNotFoundError(f"No eeg_record*.mat under {mat_root}")

    focused_limit = focused_minutes * 60 * fs
    unfocused_limit = unfocused_minutes * 60 * fs
    for f in files:
        mat = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
        if "o" not in mat:
            raise ValueError(f"'o' missing in {f}")
        data = np.asarray(mat["o"].data, dtype=np.float32)
        eeg = data[:, 3:17]

        ts = np.arange(eeg.shape[0])
        state = np.full_like(ts, 1, dtype=np.int64)  # default unfocused
        state[ts <= focused_limit] = 0
        state[ts > unfocused_limit] = 2

        eeg = (eeg - eeg.mean(axis=0, keepdims=True)) / (eeg.std(axis=0, keepdims=True) + 1e-6)
        total = eeg.shape[0]
        for start in range(0, total - win + 1, stride):
            window = eeg[start:start + win].T
            lbl_window = state[start:start + win]
            counts = np.bincount(lbl_window, minlength=3)
            lbl = int(np.argmax(counts))
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
        "label_source": "state",
        "focused_minutes": focused_minutes,
        "unfocused_minutes": unfocused_minutes,
        "signals_shape": signals_arr.shape,
        "label_counts": {int(k): int(v) for k, v in zip(*np.unique(labels_arr, return_counts=True))},
    }
    with open(out_dir / "meta_state.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("State labeling done:", meta["label_counts"])


def prepare_marker(mat_root: Path, out_dir: Path, win: int, stride: int, marker_thresh: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    signals, labels = [], []
    files = sorted(mat_root.glob("eeg_record*.mat"))
    if not files:
        raise FileNotFoundError(f"No eeg_record*.mat under {mat_root}")

    for f in files:
        mat = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
        if "o" not in mat:
            raise ValueError(f"'o' missing in {f}")
        o = mat["o"]
        data = np.asarray(o.data, dtype=np.float32)
        marker = getattr(o, "marker", None)
        if marker is None:
            marker = data[:, 23]
        marker = np.asarray(marker)
        if marker.shape[0] != data.shape[0]:
            raise ValueError(f"marker length mismatch in {f}")
        eeg = data[:, 3:17]
        eeg = (eeg - eeg.mean(axis=0, keepdims=True)) / (eeg.std(axis=0, keepdims=True) + 1e-6)

        total = eeg.shape[0]
        for start in range(0, total - win + 1, stride):
            window = eeg[start:start + win].T
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
    print("Marker labeling done:", meta["label_counts"])


# -------------------------
# Dataset
# -------------------------
class EEGLightDataset(Dataset):
    def __init__(self, signals_path: Path, labels_path: Path, normalize: bool = True):
        self.signals = np.load(signals_path, mmap_mode="r")
        labels_raw = np.load(labels_path, mmap_mode="r").astype(np.int64)
        self.indices = np.where(labels_raw >= 0)[0]
        if len(self.indices) == 0:
            raise ValueError("No valid labels (>=0). Prepare data first.")
        labels = labels_raw[self.indices]
        self.label_offset = int(labels.min())
        self.labels = labels - self.label_offset
        self.num_classes = int(self.labels.max() + 1)
        self.num_channels = int(self.signals.shape[1])
        self.seq_len = int(self.signals.shape[2])
        self.normalize = normalize

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        x = np.asarray(self.signals[real_idx], dtype=np.float32)
        if self.normalize:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-6
            x = (x - mean) / std
        y = int(self.labels[idx])
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)


# -------------------------
# Graph builder
# -------------------------
def build_graph(x: torch.Tensor, topk: int = 8, thresh: float = 0.3):
    # x: [B,C,T]
    B, C, T = x.shape
    xc = x - x.mean(dim=2, keepdim=True)
    cov = torch.matmul(xc, xc.transpose(1, 2)) / (T - 1 + 1e-6)
    var = cov.diagonal(dim1=1, dim2=2)
    std = torch.sqrt(var + 1e-6)
    denom = (std.unsqueeze(2) * std.unsqueeze(1)).clamp(min=1e-6)
    corr = cov / denom
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    if topk and topk > 0:
        topk = min(topk, C - 1)
        vals, idx = torch.topk(corr.abs(), k=topk + 1, dim=-1)
        mask = torch.zeros_like(corr, dtype=torch.bool)
        mask.scatter_(dim=-1, index=idx, value=True)
        eye = torch.eye(C, device=x.device).bool().unsqueeze(0)
        mask = mask & (~eye)
    else:
        mask = corr.abs() >= thresh
        eye = torch.eye(C, device=x.device).bool().unsqueeze(0)
        mask = mask & (~eye)

    adj = corr * mask
    denom = adj.sum(dim=-1, keepdim=True).abs() + 1e-6
    adj = adj / denom
    return adj


# -------------------------
# ST-MGM Model with Mamba
# -------------------------
class TemporalMamba(nn.Module):
    def __init__(self, d_model: int, d_state: int, drop: float, use_mamba: bool):
        super().__init__()
        self.use_mamba = use_mamba and HAS_MAMBA
        if self.use_mamba:
            self.mamba = MambaBlock(
                d_model=d_model,
                d_state=d_state,
                expand=2,
                conv_kernel=4,
                use_fast_conv=True,
                chunk_size=64,
            )
        else:
            self.mamba = nn.GRU(d_model, d_model, num_layers=2, batch_first=True, dropout=drop)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # [B*C, T, D]
        if self.use_mamba:
            y = self.mamba(x)  # residual handled inside
        else:
            y, _ = self.mamba(x)
        y = self.drop(y)
        return self.norm(y + x)


class SpatialGraph(nn.Module):
    def __init__(self, d_model: int, drop: float):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, h, adj):
        agg = torch.matmul(adj, h)  # [B,C,D]
        out = self.proj(agg)
        return self.drop(self.act(out))


class STMGMMamba(nn.Module):
    def __init__(self, channels: int, num_classes: int, d_model: int = 64, d_state: int = 16,
                 drop: float = 0.1, topk: int = 8, thresh: float = 0.3, use_mamba: bool = True):
        super().__init__()
        self.topk = topk
        self.thresh = thresh
        self.embed = nn.Linear(1, d_model)
        self.temporal = TemporalMamba(d_model, d_state, drop, use_mamba)
        self.spatial = SpatialGraph(d_model, drop)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x):  # [B,C,T]
        B, C, T = x.shape
        raw = x
        x = x.unsqueeze(-1)
        x = self.embed(x)
        x = x.permute(0, 1, 3, 2).reshape(B * C, T, -1)
        h = self.temporal(x).reshape(B, C, T, -1).mean(dim=2)
        adj = build_graph(raw, topk=self.topk, thresh=self.thresh)
        h = self.spatial(h, adj)
        g = h.mean(dim=1)
        return self.cls_head(g)


# -------------------------
# Training utilities
# -------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, amp, grad_clip):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
            loss = ce(logits, y)
        if not torch.isfinite(loss):
            print("Non-finite loss; skipping batch.")
            continue
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
            loss = ce(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)
        y_true.append(y.cpu())
        y_pred.append(preds.cpu())
    y_true = torch.cat(y_true).numpy() if y_true else np.array([])
    y_pred = torch.cat(y_pred).numpy() if y_pred else np.array([])
    return total_loss / max(total, 1), total_correct / max(total, 1), y_true, y_pred


def plot_curves(history, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / "training_curves.png"
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200); plt.close()
    return fig_path


def plot_confusion(y_true, y_pred, save_dir: Path):
    if y_true.size == 0:
        return None
    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(np.unique(y_true)))
    disp.plot(cmap="Blues", colorbar=True)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out = save_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


# -------------------------
# Main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Mamba ST-MGM")
    ap.add_argument("--prepare", action="store_true", help="Only run preprocessing and exit.")
    ap.add_argument("--mat-root", type=Path, default=Path("EEGData"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/eeg_attention"))
    ap.add_argument("--label-source", choices=["state", "marker"], default="state")
    ap.add_argument("--focused-minutes", type=float, default=10.0)
    ap.add_argument("--unfocused-minutes", type=float, default=20.0)
    ap.add_argument("--marker-thresh", type=float, default=0.1)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)

    ap.add_argument("--signals", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--save-dir", type=Path, default=Path("experiments/mamba_stmgm"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--use-mamba", action="store_true", help="Use Mamba core (requires mamba-ssm)")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.prepare:
        if args.label_source == "marker":
            prepare_marker(args.mat_root, args.out_dir, args.window, args.stride, args.marker_thresh)
        else:
            prepare_state(args.mat_root, args.out_dir, args.window, args.stride,
                          args.focused_minutes, args.unfocused_minutes)
        return

    signals_path = args.signals or (args.out_dir / "signals.npy")
    labels_path = args.labels or (args.out_dir / "labels.npy")
    if not signals_path.exists() or not labels_path.exists():
        raise FileNotFoundError("signals.npy / labels.npy not found. Run with --prepare first.")

    dataset = EEGLightDataset(signals_path, labels_path, normalize=not args.no_normalize)
    num_classes = dataset.num_classes

    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.use_amp and device.type == "cuda")
    print(f"Cihaz: {device} | AMP: {amp_enabled}")

    model = STMGMMamba(
        channels=dataset.num_channels,
        num_classes=num_classes,
        d_model=args.d_model,
        d_state=args.d_state,
        drop=args.dropout,
        topk=args.topk,
        thresh=args.thresh,
        use_mamba=args.use_mamba,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else torch.amp.GradScaler(enabled=False)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.save_dir / "best_mamba_stmgm.pt"
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

        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}")

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

    fig_path = plot_curves(history, args.save_dir)
    cm_path = plot_confusion(y_true, y_pred, args.save_dir)

    lines = [
        "# Mamba ST-MGM Results",
        f"- Best val acc: {best_val:.4f}",
        f"- Test loss: {test_loss:.4f}",
        f"- Test acc: {test_acc:.4f}",
        f"- Epochs: {len(history['train_loss'])}",
        f"- Checkpoint: {ckpt_path}",
        "",
        "## Figures",
        f"- Training curves: {fig_path}",
    ]
    if cm_path:
        lines.append(f"- Confusion matrix: {cm_path}")
    (args.save_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
