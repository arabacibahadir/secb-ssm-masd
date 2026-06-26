"""
EEG Conformer baseline: conv embedding + multi-head attention + depthwise conv.
Runs multi-seed training with shared preprocessing/splits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deep_utils import add_common_args, run_deep_baseline, save_results


class ConvModule(nn.Module):
    def __init__(self, dim: int, kernel: int, dropout: float):
        super().__init__()
        self.pw_in = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        pad = (kernel - 1) // 2
        self.dw = nn.Conv1d(dim, dim, kernel_size=kernel, padding=pad, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.pw_out = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, T, D]
        x = x.transpose(1, 2)
        x = self.pw_in(x)
        x = self.glu(x)
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw_out(x)
        x = self.drop(x)
        return x.transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int, conv_kernel: int, dropout: float):
        super().__init__()
        self.ff1 = FeedForward(dim, ff_mult, dropout)
        self.ff2 = FeedForward(dim, ff_mult, dropout)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.conv = ConvModule(dim, conv_kernel, dropout)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.n3 = nn.LayerNorm(dim)
        self.n4 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ff1(self.n1(x))
        attn_out, _ = self.mha(self.n2(x), self.n2(x), self.n2(x))
        x = x + attn_out
        x = x + self.conv(self.n3(x))
        x = x + 0.5 * self.ff2(self.n4(x))
        return x


class EEGConformer(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, d_model: int, layers: int, heads: int, ff_mult: int, conv_kernel: int, dropout: float):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ConformerBlock(d_model, heads=heads, ff_mult=ff_mult, conv_kernel=conv_kernel, dropout=dropout) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.embed(x).transpose(1, 2)  # [B, T, D]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def parse_args():
    ap = argparse.ArgumentParser(description="EEG Conformer baseline")
    add_common_args(ap, default_save_subdir="conformer")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff-mult", type=int, default=4)
    ap.add_argument("--conv-kernel", type=int, default=15)
    ap.add_argument("--dropout", type=float, default=0.2)
    return ap.parse_args()


def main():
    args = parse_args()

    def builder(input_dim: int, num_classes: int):
        return EEGConformer(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            ff_mult=args.ff_mult,
            conv_kernel=args.conv_kernel,
            dropout=args.dropout,
        )

    result = run_deep_baseline(
        model_builder=builder,
        args=args,
        model_name="EEGConformer",
        extra_config={
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "ff_mult": args.ff_mult,
            "conv_kernel": args.conv_kernel,
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
    save_results(args.save_dir / "conformer.json", payload)


if __name__ == "__main__":
    main()
