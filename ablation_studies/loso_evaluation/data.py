from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np

EEG_CHANNEL_SLICE = slice(3, 17)


def load_subject_map(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record_id = int(row["record_id"])
            if record_id in mapping:
                raise ValueError(f"duplicate record_id: {record_id}")
            mapping[record_id] = row["subject_id"].strip()
    return mapping


def label_for_sample(
    sample_index: int,
    hz: int = 128,
    focused_minutes: float = 10.0,
    unfocused_minutes: float = 20.0,
) -> int:
    if sample_index <= focused_minutes * 60 * hz:
        return 0
    if sample_index > unfocused_minutes * 60 * hz:
        return 2
    return 1


def make_window_starts(sample_count: int, window: int = 256, stride: int = 240) -> list[int]:
    return list(range(0, sample_count - window + 1, stride))


def compute_channel_stats(arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    arrays = list(arrays)
    if not arrays:
        raise ValueError("at least one training array is required")
    combined = np.concatenate(arrays, axis=0).astype(np.float64, copy=False)
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def compute_channel_stats_from_cache(
    manifest: dict,
    record_ids: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    total_count = 0
    channel_sum: np.ndarray | None = None
    channel_square_sum: np.ndarray | None = None
    for record_id in record_ids:
        metadata = manifest["records"][str(record_id)]
        if "channel_sum" in metadata and "channel_square_sum" in metadata:
            current_sum = np.asarray(metadata["channel_sum"], dtype=np.float64)
            current_square_sum = np.asarray(metadata["channel_square_sum"], dtype=np.float64)
            count = int(metadata["samples"])
        else:
            array = np.load(metadata["path"], mmap_mode="r")
            current_sum, current_square_sum = _array_moments(array)
            count = int(array.shape[0])
            del array
        if channel_sum is None:
            channel_sum = np.zeros_like(current_sum)
            channel_square_sum = np.zeros_like(current_square_sum)
        channel_sum += current_sum
        channel_square_sum += current_square_sum
        total_count += count
    if total_count == 0 or channel_sum is None or channel_square_sum is None:
        raise ValueError("at least one non-empty training record is required")
    mean = channel_sum / total_count
    variance = np.maximum(channel_square_sum / total_count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def extract_eeg_channels(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 2 or raw.shape[1] < EEG_CHANNEL_SLICE.stop:
        raise ValueError(f"expected a 2D MATLAB data array with at least 17 columns, got {raw.shape}")
    return np.asarray(raw[:, EEG_CHANNEL_SLICE], dtype=np.float32)


def load_mat_eeg(path: Path) -> np.ndarray:
    from scipy.io import loadmat

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "o" not in mat:
        raise ValueError(f"{path} does not contain MATLAB variable 'o'")
    obj = mat["o"]
    raw = obj.data if hasattr(obj, "data") else mat["o"]["data"][0, 0]
    return extract_eeg_channels(np.asarray(raw))


def _record_id_from_path(path: Path) -> int:
    match = re.fullmatch(r"eeg_record(\d+)\.mat", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"unexpected recording filename: {path.name}")
    return int(match.group(1))


def _array_moments(array: np.ndarray, chunk_size: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    channel_sum = np.zeros(array.shape[1], dtype=np.float64)
    channel_square_sum = np.zeros(array.shape[1], dtype=np.float64)
    for start in range(0, array.shape[0], chunk_size):
        chunk = np.asarray(array[start : start + chunk_size], dtype=np.float64)
        channel_sum += chunk.sum(axis=0)
        channel_square_sum += np.square(chunk).sum(axis=0)
    return channel_sum, channel_square_sum


def build_record_cache(
    data_root: Path,
    cache_dir: Path,
    record_ids: Iterable[int] | None = None,
    rebuild: bool = False,
) -> dict:
    data_root = Path(data_root).resolve()
    cache_dir = Path(cache_dir).resolve()
    records_dir = cache_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    requested = set(record_ids) if record_ids is not None else None
    source_files = sorted(
        data_root.glob("eeg_record*.mat"),
        key=_record_id_from_path,
    )
    manifest: dict = {
        "data_root": str(data_root),
        "cache_dir": str(cache_dir),
        "dtype": "float32",
        "channels": 14,
        "records": {},
    }
    for source_path in source_files:
        record_id = _record_id_from_path(source_path)
        if requested is not None and record_id not in requested:
            continue
        cache_path = records_dir / f"eeg_record{record_id}.npy"
        if rebuild or not cache_path.exists():
            np.save(cache_path, load_mat_eeg(source_path), allow_pickle=False)
        cached = np.load(cache_path, mmap_mode="r")
        channel_sum, channel_square_sum = _array_moments(cached)
        manifest["records"][str(record_id)] = {
            "record_id": record_id,
            "source_path": str(source_path.resolve()),
            "path": str(cache_path.resolve()),
            "samples": int(cached.shape[0]),
            "channels": int(cached.shape[1]),
            "channel_sum": channel_sum.tolist(),
            "channel_square_sum": channel_square_sum.tolist(),
        }
        del cached
    if requested is not None:
        missing = requested - {int(key) for key in manifest["records"]}
        if missing:
            raise FileNotFoundError(f"missing eeg_record files: {sorted(missing)}")
    manifest_path = cache_dir / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _window_label(start: int, window: int) -> int:
    labels = np.fromiter(
        (label_for_sample(index) for index in range(start, start + window)),
        dtype=np.int64,
        count=window,
    )
    return int(np.bincount(labels, minlength=3).argmax())


def build_window_index(
    manifest: dict,
    record_ids: Iterable[int],
    window: int = 256,
    stride: int = 240,
) -> list[dict[str, int]]:
    index: list[dict[str, int]] = []
    for record_id in sorted(record_ids):
        metadata = manifest["records"][str(record_id)]
        for start in make_window_starts(metadata["samples"], window=window, stride=stride):
            index.append(
                {
                    "record_id": int(record_id),
                    "start": int(start),
                    "label": _window_label(start, window),
                }
            )
    return index


def normalize_window(window: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.asarray((window - mean) / std, dtype=np.float32)
