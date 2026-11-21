"""Statistical protocol (Section 5.6).

Nadeau–Bengio corrected paired t-test with Holm step-down over the family of
competitors, Wilcoxon signed-rank as a distribution-free secondary, and
percentile bootstrap intervals over fold-level paired differences.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import stats as _stats


@dataclass
class NBResult:
    t: float
    df: int
    p_value: float
    mean_diff: float
    sd_diff: float
    inflation: float
    k: int


def nadeau_bengio_t(diffs: np.ndarray, n_train: int, n_test: int, k: int = 5) -> NBResult:
    """Corrected paired t-test for resampled validation.

    Parameters
    ----------
    diffs : (k,) array of fold-level paired differences (one score per fold).
    n_train, n_test : fold sizes on the shared protocol.
    k : number of folds.
    """
    d = np.asarray(diffs, dtype=np.float64)
    mean = float(d.mean())
    var = float(d.var(ddof=1))
    inflation = 1.0 / k + n_test / max(1, n_train)
    denom = np.sqrt(inflation * var) if var > 0 else np.inf
    t = mean / denom if denom > 0 else 0.0
    df = k - 1
    p = float(2.0 * _stats.t.sf(abs(t), df))
    return NBResult(t=float(t), df=df, p_value=p, mean_diff=mean,
                    sd_diff=float(np.sqrt(var)), inflation=inflation, k=k)


def holm_step_down(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, float]:
    """Return Holm-adjusted p-values for a family keyed by method name."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        adj = p * (m - i)
        running = max(running, adj)
        adjusted[name] = min(1.0, running)
    return adjusted


def wilcoxon_signed_rank(diffs: np.ndarray):
    """Exact two-sided Wilcoxon signed-rank test over paired fold differences."""
    try:
        res = _stats.wilcoxon(diffs, alternative="two-sided", mode="exact")
        return float(res.statistic), float(res.pvalue)
    except TypeError:
        res = _stats.wilcoxon(diffs, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)


def bootstrap_ci(diffs: np.ndarray, B: int = 10_000,
                 alpha: float = 0.05, rng: np.random.Generator | None = None):
    """Percentile bootstrap CI over fold-level paired differences."""
    rng = rng or np.random.default_rng(0)
    d = np.asarray(diffs, dtype=np.float64)
    boot = rng.choice(d, size=(B, d.size), replace=True).mean(axis=1)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_differences(primary: list[dict], competitors: dict[str, list[dict]],
                       metric: str = "macro_f1") -> dict[str, np.ndarray]:
    """Build fold-level paired difference vectors (primary − competitor).

    ``primary`` and each competitor entry is a list of fold-dicts with the
    requested metric. Seeds are averaged inside a fold before pairing.
    """
    def _fold_mean(records):
        per_fold: dict[int, list[float]] = {}
        for r in records:
            per_fold.setdefault(int(r["fold"]), []).append(float(r[metric]))
        return np.array([np.mean(per_fold[k]) for k in sorted(per_fold)])

    p = _fold_mean(primary)
    out = {}
    for name, recs in competitors.items():
        c = _fold_mean(recs)
        if len(c) != len(p):
            raise ValueError(f"Fold count mismatch for {name}")
        out[name] = p - c
    return out
