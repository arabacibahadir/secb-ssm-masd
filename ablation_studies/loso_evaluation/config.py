from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    subjects: list[str]
    seeds: list[int]
    epochs: int
    batch_size: int
    patience: int
    use_amp: bool
    max_windows_per_split: int | None
    d_model: int
    layers: int


def get_profile(name: str) -> Profile:
    if name == "smoke":
        return Profile(
            name="smoke",
            subjects=["S1"],
            seeds=[42],
            epochs=2,
            batch_size=16,
            patience=2,
            use_amp=True,
            max_windows_per_split=64,
            d_model=32,
            layers=1,
        )
    if name == "full":
        return Profile(
            name="full",
            subjects=["S1", "S2", "S3", "S4", "S5"],
            seeds=[42, 43, 44, 45, 46],
            epochs=60,
            batch_size=64,
            patience=5,
            use_amp=True,
            max_windows_per_split=None,
            d_model=224,
            layers=8,
        )
    raise ValueError(f"unknown profile: {name}")
