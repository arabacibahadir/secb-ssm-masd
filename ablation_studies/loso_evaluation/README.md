# SECB-SSM Subject-Independent LOSO Evaluation

This directory provides a strict leave-one-subject-out (LOSO) evaluation
pipeline for the proposed SECB-SSM model. It loads the `MambaEEG` class from
`attention_mamba/train.py` without modifying the original training source file.

## Protocol

- Five outer folds are used: S1-S5 are held out as the test subject in turn.
- In each fold, all recordings from the held-out subject are used only for
  testing.
- Validation is selected at the full-record level: one complete recording is
  selected from each of the four remaining subjects.
- Windows from the same recording are never split between training and
  validation.
- Channel-wise mean and standard deviation are computed only from the true
  training records.
- The window length is 256 samples, the stride is 240 samples, and the overlap
  is 16 samples.
- Proxy labels are kept consistent with the manuscript protocol:
  0-10 min = Focused, 10-20 min = Unfocused, and >20 min = Drowsy.
- The full profile runs five seeds, 42-46, across five LOSO folds, for 25 total
  training runs.

## Subject-to-record mapping assumption

The public `.mat` files do not include explicit participant identifiers.
Therefore, `subject_map.csv` defines a transparent mapping assumption based on
the five-participant acquisition protocol reported in the original study and
the chronological order of the public files:

- S1: records 1-7
- S2: records 8-14
- S3: records 15-21
- S4: records 22-28
- S5: records 29-34

The final block contains six recordings. This mapping should be reported as an
explicit metadata assumption when discussing the LOSO results.

## Installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\ablation_studies\loso_evaluation\requirements.txt
```

If CUDA-enabled PyTorch is required, install the appropriate PyTorch build from
the official PyTorch installation page. The code automatically uses CUDA when
available; otherwise, it runs on CPU.

## License and dataset access

The source code in this repository is released under the MIT License.
The MASD EEG dataset is not redistributed in this repository and should be
obtained from the original Kaggle dataset page:
https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection.
Users are responsible for complying with the dataset license and terms of use.

## Smoke test

The smoke profile runs one fold (S1), one seed (42), two epochs, and at most
64 windows per split. This profile is intended only to verify that the pipeline
runs correctly; it should not be reported as a scientific result.

```powershell
.\.venv\Scripts\python.exe -m ablation_studies.loso_evaluation.run --profile smoke
```

## Full LOSO run

```powershell
.\.venv\Scripts\python.exe -m ablation_studies.loso_evaluation.run --profile full
```

Interrupted runs can be resumed with the same command. Runs with
`status: complete` in `result.json` are skipped automatically. Use `--force` to
rerun completed folds and `--rebuild-cache` to regenerate the `.mat` cache.

Example for a single fold or seed:

```powershell
.\.venv\Scripts\python.exe -m ablation_studies.loso_evaluation.run --profile full --subjects S3 --seeds 42
```

## Outputs

Each fold/seed directory is written under
`results/<profile>/seed_<seed>/test_<subject>/` and contains:

- `result.json`: validation/test metrics and run settings
- `split_manifest.json`: record-level splits and train-only normalization values
- `history.json`: epoch-level training history
- `predictions.npz`: compressed prediction arrays
- `predictions.csv.gz`: prediction table with recording and window identifiers
- optional `best_model.pt` when `--save-checkpoints` is used

The profile-level result directory contains:

- `summary.json`
- `summary.csv`
- `loso_results.md`
- `confusion_matrix.png`

The main LOSO result is computed by concatenating the predictions from all five
held-out subjects within each seed and then reporting the mean and standard
deviation across seeds.

## Files that should not be included in source archives

Generated `cache/`, `results/`, `__pycache__/`, model checkpoints, and data
files should not be included in journal or supplementary code archives. The
code expects users to download the public MASD `.mat` files locally and place
them under an `EEG Data/` directory or pass the dataset path through
`--data-root`.

## Interpretation in the manuscript

This LOSO pipeline produces subject-independent results only for SECB-SSM.
EEGNet, TCN, and the other baseline results in the manuscript belong to the
original epoch-wise protocol. Therefore, the LOSO result should not be used to
claim superiority over baselines. Baseline comparisons should be explicitly
limited to the original epoch-wise protocol.
