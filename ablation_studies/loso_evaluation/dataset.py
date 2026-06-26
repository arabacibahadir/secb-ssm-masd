from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import normalize_window


def limit_window_index(index: list[dict], maximum: int | None, seed: int) -> list[dict]:
    if maximum is None or len(index) <= maximum:
        return list(index)
    selected_positions = sorted(random.Random(seed).sample(range(len(index)), maximum))
    return [index[position] for position in selected_positions]


class WindowDataset(Dataset):
    def __init__(
        self,
        manifest: dict,
        index: list[dict],
        mean: np.ndarray,
        std: np.ndarray,
        window: int = 256,
    ):
        self.manifest = manifest
        self.index = index
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.window = int(window)
        self._records: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _record(self, record_id: int) -> np.ndarray:
        if record_id not in self._records:
            path = self.manifest["records"][str(record_id)]["path"]
            self._records[record_id] = np.load(path, mmap_mode="r")
        return self._records[record_id]

    def __getitem__(self, position: int):
        item = self.index[position]
        record_id = int(item["record_id"])
        start = int(item["start"])
        raw = self._record(record_id)[start : start + self.window]
        normalized = normalize_window(raw, self.mean, self.std)
        return (
            torch.from_numpy(normalized),
            torch.tensor(int(item["label"]), dtype=torch.long),
            record_id,
            start,
        )

    def close(self) -> None:
        self._records.clear()
