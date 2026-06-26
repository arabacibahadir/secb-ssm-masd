"""
DeepConvNet and ShallowConvNet baselines.
Choose --model deepconvnet/shallowconvnet/both; defaults run both with shared training hyperparameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from .deep_utils import add_common_args, run_deep_baseline, save_results


class DeepConvNet(nn.Module):
    def __init__(self, n_chans: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5)),
            nn.Conv2d(25, 25, kernel_size=(n_chans, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout),

            nn.Conv2d(25, 50, kernel_size=(1, 5)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout),

            nn.Conv2d(50, 100, kernel_size=(1, 5)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout),

            nn.Conv2d(100, 200, kernel_size=(1, 5)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(200, n_classes)

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2).unsqueeze(1)  # [B, 1, C, T]
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


class ShallowConvNet(nn.Module):
    def __init__(self, n_chans: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 13), padding=(0, 6), bias=False),
            nn.Conv2d(40, 40, kernel_size=(n_chans, 1), bias=False),
            nn.BatchNorm2d(40),
        )
        self.pool = nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7))
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(40, n_classes)

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.conv(x)
        x = x ** 2  # square nonlinearity
        x = self.pool(x)
        x = torch.log(x + 1e-6)
        x = self.drop(x)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)


def parse_args():
    ap = argparse.ArgumentParser(description="DeepConvNet / ShallowConvNet baselines")
    add_common_args(ap, default_save_subdir="convnets")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--model", type=str, choices=["deepconvnet", "shallowconvnet", "both"], default="both")
    return ap.parse_args()


def main():
    args = parse_args()
    models_to_run: List[str] = ["deepconvnet", "shallowconvnet"] if args.model == "both" else [args.model]
    results: Dict[str, Dict] = {}

    for name in models_to_run:
        if name == "deepconvnet":
            builder = lambda input_dim, num_classes: DeepConvNet(input_dim, num_classes, dropout=args.dropout)
        else:
            builder = lambda input_dim, num_classes: ShallowConvNet(input_dim, num_classes, dropout=args.dropout)
        res = run_deep_baseline(
            model_builder=builder,
            args=args,
            model_name=name,
            extra_config={"dropout": args.dropout},
        )
        results[name] = res

    payload = {
        "data": {
            "data_root": str(args.data_root),
            "epoch_length": args.epoch_length,
            "step_size": args.step_size,
        },
        "results": results,
    }
    save_path = args.save_dir / ("convnets.json" if args.model == "both" else f"{args.model}.json")
    save_results(save_path, payload)


if __name__ == "__main__":
    main()
