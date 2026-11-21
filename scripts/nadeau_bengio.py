#!/usr/bin/env python3
"""Run the Nadeau–Bengio corrected paired t-test with Holm step-down.

Expects one JSON per run with fold-level macro-F1 (Section 5.6)::

    [
      {"fold": 0, "seed": 42, "macro_f1": 90.6},
      {"fold": 1, "seed": 42, "macro_f1": 90.8},
      ...
    ]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from pigmentnet.utils.stats import (
    nadeau_bengio_t,
    holm_step_down,
    wilcoxon_signed_rank,
    bootstrap_ci,
    paired_differences,
)


def _load(path: Path):
    """Load either one JSON file or a directory of per-run JSONs."""
    if path.is_dir():
        recs = []
        for p in sorted(path.glob("*.json")):
            d = json.loads(p.read_text())
            if isinstance(d, list):
                recs.extend(d)
            elif "folds" in d:
                recs.extend(d["folds"])
            else:
                recs.append(d)
        return recs
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary", required=True, type=Path)
    ap.add_argument("--competitors", nargs="+", required=True, type=Path)
    ap.add_argument("--n-train", type=int, required=True,
                    help="Training-fold size for the Nadeau–Bengio inflation factor.")
    ap.add_argument("--n-test", type=int, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    primary = _load(args.primary)
    comp_recs = {str(p): _load(p) for p in args.competitors}
    diffs = paired_differences(primary, comp_recs, metric="macro_f1")

    results = {}
    for name, d in diffs.items():
        nb = nadeau_bengio_t(d, n_train=args.n_train, n_test=args.n_test, k=args.k)
        w_stat, w_p = wilcoxon_signed_rank(d)
        lo, hi = bootstrap_ci(d, B=10_000)
        results[name] = {
            "mean_diff": nb.mean_diff,
            "sd_diff":   nb.sd_diff,
            "t":         nb.t,
            "df":        nb.df,
            "p":         nb.p_value,
            "wilcoxon":  {"stat": w_stat, "p": w_p},
            "bootstrap_ci": [lo, hi],
        }
    p_values = {name: r["p"] for name, r in results.items()}
    adj = holm_step_down(p_values, alpha=args.alpha)
    for name, r in results.items():
        r["p_holm"] = adj[name]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
