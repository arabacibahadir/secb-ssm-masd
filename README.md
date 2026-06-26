# EEG Attention State Classification

This repository contains PyTorch training and evaluation scripts for classifying attention states from passive EEG recordings. It includes the proposed Attention Mamba model, ablation studies, and comparison baselines.

## Dataset

The scripts expect MATLAB `.mat` files under `EEGData/`. Each file should contain an object `o` with `o.data`, sampled at 128 Hz. The 14 EEG channels are columns 4 through 17 of `o.data`: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, and AF4.

The MASD EEG dataset is not redistributed in this repository and should be obtained from the original Kaggle dataset page: https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection. Users are responsible for complying with the dataset license and terms of use.

## License

The source code in this repository is released under the MIT License.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```


## Main Model

Train the Attention Mamba model with local data:

```bash
python attention_mamba/train.py --data-root EEGData --epochs 100 --batch-size 64 --use-amp --lr 4e-4 --log-every 10 --bidir
```

Run a short smoke test:

```bash
python attention_mamba/train.py --data-root EEGData --epochs 10 --batch-size 32 --log-every 5 --save-dir attention_mamba/experiments/debug_run
```

## Baselines and Ablations

Baseline scripts are organized by model family:

- `attention_baseline/`: BiLSTM and CNN-BiLSTM baselines.
- `eegconformer/`: EEG Conformer baseline.
- `eegconformer_plus/`: EEG Conformer Plus variant.
- `fusion_cnn/`: Fusion CNN model.
- `fbspecnet/`: FBSpecNet model.
- `mamba_conformer/` and `mamba_conformer_plus/`: Mamba-Conformer variants.
- `bi_mamba/`, `stmgm/`, and `mamba_stmgm/`: additional sequence and graph baselines.
- `phase_cde/`: PhaseCDE model.
- `attention_mamba/baselines/`: classical ML and deep-learning baselines for the main model protocol.
- `ablation_studies/`: ablation driver for the Attention Mamba architecture.

Each model directory includes a `commands.txt` file with reproducible command examples.

Run the ablation study driver:

```bash
python -m ablation_studies.run_ablation --data-root EEGData
```

## Outputs

Training scripts write metrics, training curves, confusion matrices, and checkpoints to an experiment directory, usually under `experiments/` or the model-specific `experiments/` folder. Generated experiment artifacts are intentionally excluded from this clean source package; rerun the corresponding script to regenerate them.

## Notes

- The default split policy in these scripts is window-level random splitting unless a specific script documents otherwise.
- Use the same preprocessing, seeds, and split settings when comparing models.
- Checkpoint files are ignored by Git (`*.pt`, `*.pth`, `*.ckpt`) to keep the repository lightweight.
