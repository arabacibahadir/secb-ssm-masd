"""
Temporal Convolutional Network (TCN) baseline.
Provides a strong yet simple 1D alternative to CNNs/transformers on the same EEG windows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from .deep_utils import add_common_args, run_deep_baseline, save_results


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.activation(y)
        y = self.drop(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.activation(y)
        y = self.drop(y)

        res = x if self.downsample is None else self.downsample(x)
        return self.activation(y + res)


class TCNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, channels: int, levels: int, kernel: int, dropout: float):
        super().__init__()
        chs: List[int] = [channels] * levels
        layers: List[nn.Module] = []
        in_ch = input_dim
        for i, out_ch in enumerate(chs):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel=kernel, dilation=dilation, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.tcn(x)
        x = x.mean(dim=-1)
        return self.head(x)


def parse_args():
    ap = argparse.ArgumentParser(description="TCN baseline")
    add_common_args(ap, default_save_subdir="tcn")
    ap.add_argument("--channels", type=int, default=64, help="Hidden channel width for TCN blocks.")
    ap.add_argument("--levels", type=int, default=5, help="Number of residual TCN blocks (dilations double each layer).")
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.2)
    return ap.parse_args()


def main():
    args = parse_args()

    def builder(input_dim: int, num_classes: int):
        return TCNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            channels=args.channels,
            levels=args.levels,
            kernel=args.kernel,
            dropout=args.dropout,
        )

    result = run_deep_baseline(
        model_builder=builder,
        args=args,
        model_name="TCN",
        extra_config={
            "channels": args.channels,
            "levels": args.levels,
            "kernel": args.kernel,
            "dropout": args.dropout,
        },
    )
    payload = {
        "data": {
            "data_root": str(args.data_root),
            "epoch_length": args.epoch_length,
            "step_size": args.step_size,
        },
        **result,
    }
    save_results(args.save_dir / "tcn.json", payload)


if __name__ == "__main__":
    main()
