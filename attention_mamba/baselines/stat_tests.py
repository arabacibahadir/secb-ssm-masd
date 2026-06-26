"""
Simple statistical tests over per-seed scores from baseline result JSON files.
Supports paired t-test and Wilcoxon signed-rank across methods using the same seeds.
Usage:
python stat_tests.py --metric macro_f1 --files path/to/method1.json path/to/method2.json [...]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def extract_runs(path: Path, metric: str) -> List[Tuple[str, List[float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    runs: List[Tuple[str, List[float]]] = []
    if "seeds" in data:
        name = data.get("model", path.stem)
        vals = [s["test"][metric] for s in data["seeds"] if metric in s.get("test", {})]
        runs.append((name, vals))
    if "results" in data:
        for name, res in data["results"].items():
            if "seeds" in res:
                vals = [s["test"][metric] for s in res["seeds"] if metric in s.get("test", {})]
                runs.append((name, vals))
    return runs


def paired_tests(scores_a: List[float], scores_b: List[float]) -> Dict[str, float]:
    scores_a = np.array(scores_a, dtype=float)
    scores_b = np.array(scores_b, dtype=float)
    if scores_a.shape != scores_b.shape:
        raise ValueError("Score vectors must have the same shape for paired tests.")
    t_stat, t_p = ttest_rel(scores_a, scores_b)
    w_stat, w_p = wilcoxon(scores_a, scores_b, zero_method="wilcox", correction=False)
    return {"ttest_stat": float(t_stat), "ttest_p": float(t_p), "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p)}


def parse_args():
    ap = argparse.ArgumentParser(description="Paired tests over per-seed results from baseline JSON files.")
    ap.add_argument("--metric", type=str, default="macro_f1", help="Metric key inside per-seed test results.")
    ap.add_argument("--files", type=Path, nargs="+", required=True, help="JSON result files to compare.")
    return ap.parse_args()


def main():
    args = parse_args()
    collected: List[Tuple[str, List[float]]] = []
    for fp in args.files:
        runs = extract_runs(fp, args.metric)
        collected.extend(runs)

    if len(collected) < 2:
        print("Need at least two runs to compare.")
        return

    print(f"Paired tests on metric '{args.metric}'")
    for (name_a, scores_a), (name_b, scores_b) in itertools.combinations(collected, 2):
        try:
            stats = paired_tests(scores_a, scores_b)
            print(f"{name_a} vs {name_b}: t={stats['ttest_stat']:.4f} (p={stats['ttest_p']:.4g}), "
                  f"wilcoxon={stats['wilcoxon_stat']:.4f} (p={stats['wilcoxon_p']:.4g})")
        except Exception as exc:
            print(f"Skipping {name_a} vs {name_b}: {exc}")


if __name__ == "__main__":
    main()
