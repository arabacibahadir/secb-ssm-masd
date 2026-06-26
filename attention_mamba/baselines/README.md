# Baseline Models

These scripts evaluate classical machine-learning and deep-learning baselines with the same preprocessing, windowing, labels, and train/validation/test split policy used by `attention_mamba/train.py`.

Generated JSON summaries include test accuracy, balanced accuracy, macro-F1, per-class precision/recall/F1, confusion matrices, optional one-vs-rest ROC-AUC, and per-seed aggregate statistics.

## Classical Baselines

```bash
python -m attention_mamba.baselines.bandpower_ml --data-root EEGData
python -m attention_mamba.baselines.hjorth_ml --data-root EEGData
python -m attention_mamba.baselines.riemann_ml --data-root EEGData
```

The Riemannian baseline requires the optional `pyriemann` package.

## Deep Baselines

```bash
python -m attention_mamba.baselines.eegnet_baseline --data-root EEGData --epochs 50 --use-amp
python -m attention_mamba.baselines.convnet_baselines --data-root EEGData --model both
python -m attention_mamba.baselines.conformer_baseline --data-root EEGData --epochs 50
python -m attention_mamba.baselines.eegconformerplus_baseline --data-root EEGData --epochs 50
python -m attention_mamba.baselines.tcn_baseline --data-root EEGData --epochs 50
```

Default seeds are `42 43 44 45 46`; override them with `--seeds`. By default, outputs are written under `attention_mamba/experiments/baselines/`.

## Paired Statistical Tests

```bash
python -m attention_mamba.baselines.stat_tests --metric macro_f1 --files attention_mamba/experiments/baselines/eegnet/eegnet.json attention_mamba/experiments/baselines/convnets/convnets.json
```

Use the same seed set across result files before applying paired tests.
