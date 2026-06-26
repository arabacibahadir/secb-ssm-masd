"""
EEG attention classification baseline pipeline.
- Downloads the public dataset via kagglehub when requested.
- Preprocess: z-score channels, timestamp-based labels (focused/unfocused/drowsy), window into epochs.
- Models: BiLSTM or CNN+BiLSTM.
- Saves metrics/plots/checkpoints under experiments/attention_baseline next to this file.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.autocast("cpu", enabled=False)


FOCUSED_CLASS = 0
UNFOCUSED_CLASS = 1
DROWSY_CLASS = 2


def get_state(timestamp: int, hz: int = 128, focused_min: float = 10.0, unfocused_min: float = 20.0):
    if timestamp <= focused_min * 60 * hz:
        return FOCUSED_CLASS
    if timestamp > unfocused_min * 60 * hz:
        return DROWSY_CLASS
    return UNFOCUSED_CLASS


def download_public_dataset() -> Path:
    """
    Downloads the public dataset via kagglehub.
    Returns Path to 'EEG Data' folder.
    """
    import kagglehub

    path = kagglehub.dataset_download("inancigdem/eeg-data-for-mental-attention-state-detection")
    eeg_data = Path(path) / "EEG Data"
    if not eeg_data.exists():
        raise FileNotFoundError(f"'EEG Data' not found under downloaded path {path}")
    return eeg_data


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
    """
    Sliding window epoching with slight overlap.
    """
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
    all_signals: List[np.ndarray] = []
    all_labels: List[int] = []
    files = sorted(data_root.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found under {data_root}")
    for f in files:
        df = load_file(f, scaler)
        epochs = split_epochs(df, hz=128, epoch_length=epoch_length, step_size=step_size)
        for ep in epochs:
            feat = ep.drop(columns=["state", "timestamp"], errors="ignore").values.astype(np.float32)
            label = int(ep["state"].mode()[0])
            all_signals.append(feat)  # [T, C]
            all_labels.append(label)
    signals = np.stack(all_signals)  # [N, T, C]
    labels = np.array(all_labels, dtype=np.int64)
    return signals, labels


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


class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = F.relu(self.fc1(out[:, -1, :]))
        out = self.drop(out)
        out = self.fc2(out)
        return out


class CNNBiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # [B, T', C']
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = F.relu(self.fc1(self.drop(out[:, -1, :])))
        out = self.fc2(self.drop(out))
        return out


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


def parse_args():
    ap = argparse.ArgumentParser(description="Train and evaluate EEG attention baseline models.")
    ap.add_argument("--download", action="store_true", help="Download dataset via kagglehub.")
    ap.add_argument("--data-root", type=Path, default=Path("EEGData"), help="Path to folder with .mat files (e.g., existing EEGData).")
    ap.add_argument("--epoch-length", type=float, default=2.0, help="Epoch length in seconds.")
    ap.add_argument(
        "--step-size",
        type=float,
        default=0.125,
        help="Overlap duration in seconds; stride = epoch length - overlap (default stride: 240 samples).",
    )
    ap.add_argument("--model", choices=["bilstm", "cnn-bilstm"], default="bilstm")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=10, help="Print training stats every N steps (0 disables).")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--save-dir", type=Path, default=Path(__file__).parent / "experiments" / "attention_baseline")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.download:
        data_root = download_public_dataset()
    else:
        data_root = args.data_root

    print(f"Using data from: {data_root}")
    signals, labels = build_dataset(Path(data_root), epoch_length=args.epoch_length, step_size=args.step_size)
    print(f"Dataset built: signals {signals.shape}, labels {labels.shape}")

    train_loader, val_loader, test_loader = create_loaders(
        signals, labels, batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.use_amp and device.type == "cuda")
    input_size = signals.shape[2]
    num_classes = int(labels.max() + 1)

    if args.model == "bilstm":
        model = BiLSTM(input_size, args.hidden_size, args.layers, num_classes, args.dropout).to(device)
    else:
        model = CNNBiLSTM(input_size, args.hidden_size, args.layers, num_classes, args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if device.type == "cuda" else torch.amp.GradScaler(enabled=False)
    stopper = EarlyStopping(patience=args.patience)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.save_dir / f"best_{args.model}.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(
            model, train_loader, optimizer, device, criterion, scaler, amp,
            train=True, log_every=args.log_every, epoch=epoch, epochs=args.epochs
        )
        va_loss, va_acc, va_f1, _, _ = run_epoch(model, val_loader, optimizer, device, criterion, scaler, amp, train=False)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        stopper.step(va_loss, model)
        if stopper.best_state is not None:
            torch.save(stopper.best_state, ckpt)
        print(
            f"Epoch {epoch:03d} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
            f"val_loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}"
        )
        if stopper.early_stop:
            print("Early stopping triggered.")
            break

    stopper.load_best(model)
    elapsed = time.time() - start
    print(f"Training time: {elapsed/60:.1f} min")

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
        "# EEG Attention Baseline Results",
        f"- Model: {args.model}",
        f"- Test acc: {te_acc:.4f}, f1: {te_f1:.4f}, prec: {te_prec:.4f}, rec: {te_rec:.4f}",
        f"- Checkpoint: {ckpt}",
        f"- Curves: {fig_path}",
        f"- Data root: {data_root}",
    ]
    (args.save_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
