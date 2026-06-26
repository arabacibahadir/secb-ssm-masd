# Mamba ST-MGM

This directory hosts an ST-MGM implementation aligned with the [state-spaces/mamba](https://github.com/state-spaces/mamba) reference components. It integrates an SSM core, convolution branch, and gating within a spatio-temporal pipeline for EEG attention classification.

Key features:
- Mamba-based temporal encoder (SSM core, conv branch, gating) inspired by state-spaces/mamba repo.
- Graph construction via dynamic correlation; optional top-k pruning.
- Masked pretraining (optional) and finetuning head; outputs metrics, curves, confusion matrix, summary, checkpoint.

Commands from the repository root:
```
# Prepare windows/labels (state rule by default)
python mamba_stmgm/train.py --prepare --mat-root EEGData --out-dir data/eeg_attention \
  --label-source state --focused-minutes 10 --unfocused-minutes 20 --window 512 --stride 256

# Train ST-MGM with Mamba core (requires mamba-ssm installation)
python mamba_stmgm/train.py --out-dir data/eeg_attention --epochs 50 --batch-size 64 --use-amp --use-mamba
```
