"""
Shared preprocessing and dataset utilities for baselines baselines.
- Mirrors attention_mamba/train.py preprocessing so all baselines use identical
  2 s windows, 0.125 s overlap (16 samples), stride-240 segmentation, and splits.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset


# -----------------------------
# Seeds / labels
# -----------------------------
FOCUSED_CLASS = 0
UNFOCUSED_CLASS = 1
DROWSY_CLASS = 2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


# -----------------------------
# Splits / loaders
# -----------------------------
def get_split_indices(n_total: int, seed: int, train_ratio: float, val_ratio: float):
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).numpy()
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def split_arrays(
    signals: np.ndarray,
    labels: np.ndarray,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, np.ndarray]:
    train_idx, val_idx, test_idx = get_split_indices(len(labels), seed, train_ratio, val_ratio)
    return {
        "train_x": signals[train_idx],
        "train_y": labels[train_idx],
        "val_x": signals[val_idx],
        "val_y": labels[val_idx],
        "test_x": signals[test_idx],
        "test_y": labels[test_idx],
    }


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


def create_loaders(
    signals: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds = EpochDataset(signals, labels)
    train_idx, val_idx, test_idx = get_split_indices(len(ds), seed, train_ratio, val_ratio)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_ds = Subset(ds, train_idx.tolist())
    val_ds = Subset(ds, val_idx.tolist())
    test_ds = Subset(ds, test_idx.tolist())
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )
