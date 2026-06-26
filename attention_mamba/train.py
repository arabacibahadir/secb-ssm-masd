"""
EEG Attention Mamba model.
- Preprocess: timestamp proxy labels, z-score, 2 s window, 0.125 s overlap (16 samples; stride 240).
- Model: Conv embed w/ SE -> Stacked bi-SSM blocks (Mamba-lite) + depthwise conv mixing.
- Outputs: metrics, curves, checkpoint under experiments/attention_mamba.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


FOCUSED_CLASS = 0
UNFOCUSED_CLASS = 1
DROWSY_CLASS = 2


def get_state(timestamp: int, hz: int = 128, focused_min: float = 10.0, unfocused_min: float = 20.0):
    if timestamp <= focused_min * 60 * hz:
        return FOCUSED_CLASS
    if timestamp > unfocused_min * 60 * hz:
        return DROWSY_CLASS
    return UNFOCUSED_CLASS


# -----------------------------
# Preprocess
# -----------------------------
def load_file(path: Path, scaler: StandardScaler) -> pd.DataFrame:
    mat = loadmat(path)
    data = mat["o"]["data"][0, 0]
    cols = [
        "ED_COUNTER",
        "ED_INTERPOLATED",
        "ED_RAW_CQ",
        "ED_AF3",
        "ED_F7",
        "ED_F3",
        "ED_FC5",
        "ED_T7",
        "ED_P7",
        "ED_O1",
        "ED_O2",
        "ED_P8",
        "ED_T8",
        "ED_FC6",
        "ED_F4",
        "ED_F8",
        "ED_AF4",
        "ED_GYROX",
        "ED_GYROY",
        "ED_TIMESTAMP",
        "ED_ES_TIMESTAMP",
        "ED_FUNC_ID",
        "ED_FUNC_VALUE",
        "ED_MARKER",
        "ED_SYNC_SIGNAL",
    ]
    eeg_df = pd.DataFrame(data, columns=cols)
    eeg_df = eeg_df[
        ["ED_AF3", "ED_F7", "ED_F3", "ED_FC5", "ED_T7", "ED_P7", "ED_O1", "ED_O2", "ED_P8", "ED_T8", "ED_FC6", "ED_F4", "ED_F8", "ED_AF4"]
    ]
    eeg_df.columns = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    eeg_df = pd.DataFrame(scaler.fit_transform(eeg_df), columns=eeg_df.columns)
    eeg_df.reset_index(inplace=True)
    eeg_df.rename(columns={"index": "timestamp"}, inplace=True)
    eeg_df["state"] = eeg_df["timestamp"].apply(get_state)
    return eeg_df


def split_epochs(df: pd.DataFrame, hz: int = 128, epoch_length: float = 2.0, step_size: float = 0.125) -> List[pd.DataFrame]:
    step = int(epoch_length * hz - step_size * hz)
    win = int(epoch_length * hz)
    starts = []
    cur = 0
    while cur + win <= df.shape[0]:
        starts.append(cur)
        cur += step
    epochs = []
    for s in starts:
        epochs.append(df.iloc[s : s + win])
    return epochs


def build_dataset(data_root: Path, epoch_length: float, step_size: float) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    signals: List[np.ndarray] = []
    labels: List[int] = []
    files = sorted(data_root.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files under {data_root}")
    for f in files:
        df = load_file(f, scaler)
        epochs = split_epochs(df, hz=128, epoch_length=epoch_length, step_size=step_size)
        for ep in epochs:
            feat = ep.drop(columns=["state", "timestamp"], errors="ignore").values.astype(np.float32)
            label = int(ep["state"].mode()[0])
            signals.append(feat)  # [T, C]
            labels.append(label)
    signals_arr = np.stack(signals)  # [N, T, C]
    labels_arr = np.array(labels, dtype=np.int64)
    return signals_arr, labels_arr


class EpochDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.signals[idx])
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def create_loaders(signals: np.ndarray, labels: np.ndarray, batch_size: int, num_workers: int, seed: int):
    ds = EpochDataset(signals, labels)
    n_total = len(ds)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=g)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


# -----------------------------
# SSM and convolution blocks
# -----------------------------
class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B,C,T]
        w = self.pool(x).squeeze(-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w


class SelectiveSSMLite(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim * 2)
        self.decay_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.skip = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        u_gate = self.in_proj(x)
        u, gate = u_gate.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        decay = torch.sigmoid(self.decay_proj(x))

        B, T, D = x.shape
        s = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            s = decay[:, t] * s + u[:, t]
            ys.append(s)
        y = torch.stack(ys, dim=1)
        y = self.out_proj(y)
        y = self.drop(y)
        return y * gate + self.skip(x)


class BiSelectiveSSM(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.fwd = SelectiveSSMLite(dim, dropout)
        self.bwd = SelectiveSSMLite(dim, dropout)
        self.fuse = nn.Linear(dim * 2, dim)

    def forward(self, x):
        y_fwd = self.fwd(x)
        y_bwd = self.bwd(torch.flip(x, dims=[1]))
        y_bwd = torch.flip(y_bwd, dims=[1])
        return self.fuse(torch.cat([y_fwd, y_bwd], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class ConvMix(nn.Module):
    def __init__(self, dim: int, kernel: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) // 2
        self.dw = nn.Conv1d(dim, dim, kernel, padding=pad, groups=dim)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.drop(x)
        return x.transpose(1, 2)


class MambaBlock(nn.Module):
    def __init__(self, dim: int, ff_expansion: int, conv_kernel: int, dropout: float, bidir: bool = True):
        super().__init__()
        self.ff1 = FeedForward(dim, ff_expansion, dropout)
        self.ff2 = FeedForward(dim, ff_expansion, dropout)
        self.ssm = BiSelectiveSSM(dim, dropout) if bidir else SelectiveSSMLite(dim, dropout)
        self.conv = ConvMix(dim, conv_kernel, dropout)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.n3 = nn.LayerNorm(dim)
        self.n4 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + 0.5 * self.ff1(self.n1(x))
        x = x + self.drop(self.ssm(self.n2(x)))
        x = x + self.conv(self.n3(x))
        x = x + 0.5 * self.ff2(self.n4(x))
        return x


class ConvEmbedding(nn.Module):
    def __init__(self, in_ch: int, dim: int, kernel: int, stride: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Conv1d(in_ch, dim, kernel_size=kernel, stride=stride, padding=pad)
        self.se = SqueezeExcite1d(dim, reduction=4)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.se(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x.transpose(1, 2)


class MambaEEG(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, d_model: int, layers: int, ff_exp: int, conv_kernel: int,
                 emb_kernel: int, emb_stride: int, dropout: float, bidir: bool = True):
        super().__init__()
        self.embed = ConvEmbedding(input_dim, d_model, emb_kernel, emb_stride, dropout)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, ff_expansion=ff_exp, conv_kernel=conv_kernel, dropout=dropout, bidir=bidir)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)


# -----------------------------
# Train / Eval
# -----------------------------
def run_epoch(model, loader, optimizer, device, criterion, scaler, amp, train: bool,
              log_every: int = 0, epoch: int = 0, epochs: int = 0):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    total = 0
    correct = 0
    for step, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device)
        yb = yb.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.autocast(device_type="cuda", enabled=amp):
            logits = model(xb)
            loss = criterion(logits, yb)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
        all_labels.append(yb.detach().cpu())
        all_preds.append(pred.detach().cpu())
        if train and log_every > 0 and (step == 1 or step % log_every == 0 or step == len(loader)):
            avg_loss = total_loss / max(total, 1)
            avg_acc = correct / max(total, 1)
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[Epoch {epoch:03d}/{epochs}] Step {step:04d}/{len(loader)} | "
                f"loss {avg_loss:.4f} acc {avg_acc:.4f} | lr {lr_now:.2e}",
                flush=True,
            )
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    acc = correct / max(total, 1)
    f1 = f1_score(all_labels, all_preds, average="macro")
    prec = precision_score(all_labels, all_preds, average="macro")
    rec = recall_score(all_labels, all_preds, average="macro")
    return total_loss / max(total, 1), acc, f1, prec, rec


class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best = None
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module):
        score = -val_loss
        if self.best is None or score > self.best + self.delta:
            self.best = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def plot_curves(history: dict, save_dir: Path):
    fig = plt.figure(figsize=(10, 4))
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
    fig_path = save_dir / "training_curves.png"
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    return fig_path


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train and evaluate the EEG Attention Mamba model.")
    ap.add_argument("--data-root", type=Path, default=Path("EEGData"), help="Path to folder with .mat files.")
    ap.add_argument("--epoch-length", type=float, default=2.0, help="Epoch length seconds.")
    ap.add_argument(
        "--step-size",
        type=float,
        default=0.125,
        help="Overlap duration in seconds; stride = epoch length - overlap (default stride: 240 samples).",
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--d-model", type=int, default=224)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ff-expansion", type=int, default=4)
    ap.add_argument("--conv-kernel", type=int, default=31)
    ap.add_argument("--emb-kernel", type=int, default=7)
    ap.add_argument("--emb-stride", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.18)
    ap.add_argument(
        "--bidir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bidirectional SSM blocks (default: enabled).",
    )
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=10, help="Print training stats every N steps (0 disables).")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--save-dir", type=Path, default=Path(__file__).parent / "experiments" / "attention_mamba")
    ap.add_argument("--resume", action="store_true", help="Resume from last_mamba.pt if present.")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = args.data_root
    print(f"Using data from: {data_root}")
    signals, labels = build_dataset(Path(data_root), epoch_length=args.epoch_length, step_size=args.step_size)
    print(f"Dataset built: signals {signals.shape}, labels {labels.shape}")

    train_loader, val_loader, test_loader = create_loaders(
        signals, labels, batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.use_amp and device.type == "cuda")
    input_dim = signals.shape[2]
    num_classes = int(labels.max() + 1)

    model = MambaEEG(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=args.d_model,
        layers=args.layers,
        ff_exp=args.ff_expansion,
        conv_kernel=args.conv_kernel,
        emb_kernel=args.emb_kernel,
        emb_stride=args.emb_stride,
        dropout=args.dropout,
        bidir=args.bidir,
    ).to(device)

    if args.resume:
        last_ckpt = Path(args.save_dir) / "last_mamba.pt"
        best_ckpt = Path(args.save_dir) / "best_mamba.pt"
        loaded = False
        if last_ckpt.exists():
            model.load_state_dict(torch.load(last_ckpt, map_location=device))
            print(f"Resumed from {last_ckpt}")
            loaded = True
        elif best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device))
            print(f"Resumed from {best_ckpt}")
            loaded = True
        if not loaded:
            print("Resume requested but no checkpoint found; training from scratch.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if device.type == "cuda" else torch.amp.GradScaler(enabled=False)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.save_dir / "best_mamba.pt"
    last_ckpt = args.save_dir / "last_mamba.pt"
    epoch_log = args.save_dir / "epoch_metrics.jsonl"
    epoch_log.write_text("", encoding="utf-8")
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }
    best_val = -1.0
    start = time.time()
    stopper = EarlyStopping(patience=args.patience)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(
            model, train_loader, optimizer, device, criterion, scaler, amp,
            train=True, log_every=args.log_every, epoch=epoch, epochs=args.epochs
        )
        va_loss, va_acc, va_f1, _, _ = run_epoch(
            model, val_loader, optimizer, device, criterion, scaler, amp, train=False
        )
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)
        scheduler.step()

        torch.save(model.state_dict(), last_ckpt)  # save latest for resume

        # per-epoch logging for reproducibility
        with epoch_log.open("a", encoding="utf-8") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "train_f1": tr_f1,
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                    "val_f1": va_f1,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                f,
            )
            f.write("\n")

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), ckpt)

        stopper.step(va_loss, model)
        if stopper.early_stop:
            print("Early stopping triggered.")
            break

        print(
            f"Epoch {epoch:03d} | lr {optimizer.param_groups[0]['lr']:.2e} | "
            f"train {tr_loss:.4f}/{tr_acc:.4f}/{tr_f1:.4f} | "
            f"val {va_loss:.4f}/{va_acc:.4f}/{va_f1:.4f}"
        )

    elapsed = time.time() - start
    print(f"Training time: {elapsed/60:.1f} min")

    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded best checkpoint: {ckpt}")

    te_loss, te_acc, te_f1, te_prec, te_rec = run_epoch(
        model, test_loader, optimizer, device, criterion, scaler, amp, train=False
    )
    print(f"Test -> loss {te_loss:.4f} acc {te_acc:.4f} f1 {te_f1:.4f} prec {te_prec:.4f} rec {te_rec:.4f}")

    fig_path = plot_curves(history, args.save_dir)
    serializable_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    metrics = {
        "history": history,
        "test": {"loss": te_loss, "acc": te_acc, "f1": te_f1, "precision": te_prec, "recall": te_rec},
        "config": serializable_args,
        "checkpoint": str(ckpt),
        "data_root": str(data_root),
    }
    with open(args.save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    lines = [
        "# EEG Attention Mamba Results",
        f"- Test acc: {te_acc:.4f}, f1: {te_f1:.4f}, prec: {te_prec:.4f}, rec: {te_rec:.4f}",
        f"- Checkpoint: {ckpt}",
        f"- Curves: {fig_path}",
        f"- Data root: {data_root}",
    ]
    (args.save_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
