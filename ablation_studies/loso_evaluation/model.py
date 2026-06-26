from __future__ import annotations

import importlib.util
from pathlib import Path


def load_mamba_eeg_class(source_path: Path):
    source_path = Path(source_path).resolve()
    spec = importlib.util.spec_from_file_location("_loso_source_attention_mamba", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load model source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MambaEEG
