"""
EEGNet baseline using the same preprocessing/splits as attention_mamba/train.py.
Evaluates 5 seeds by default and reports balanced/macro metrics, per-class scores, confusion matrices, and mean/std.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .deep_utils import add_common_args, run_deep_baseline, save_results


class EEGNet(nn.Module):
    def __init__(self, n_chans: int, n_classes: int, dropout: float = 0.5, temporal_kernel: int = 64):
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.depthwise = nn.Conv2d(8, 16, kernel_size=(n_chans, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.act = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.drop = nn.Dropout(dropout)

        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 32, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2).unsqueeze(1)  # [B, 1, C, T]
        x = self.temporal(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool1(x)
        x = self.drop(x)
        x = self.separable(x)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)


def parse_args():
    ap = argparse.ArgumentParser(description="EEGNet baseline")
    add_common_args(ap, default_save_subdir="eegnet")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--temporal-kernel", type=int, default=64)
    return ap.parse_args()


def main():
    args = parse_args()

    def builder(input_dim: int, num_classes: int):
        return EEGNet(
            n_chans=input_dim,
            n_classes=num_classes,
            dropout=args.dropout,
            temporal_kernel=args.temporal_kernel,
        )

    result = run_deep_baseline(
        model_builder=builder,
        args=args,
        model_name="EEGNet",
        extra_config={"dropout": args.dropout, "temporal_kernel": args.temporal_kernel},
    )
    payload = {
        "data": {
            "data_root": str(args.data_root),
            "epoch_length": args.epoch_length,
            "step_size": args.step_size,
        },
        **result,
    }
    save_results(args.save_dir / "eegnet.json", payload)


if __name__ == "__main__":
    main()
