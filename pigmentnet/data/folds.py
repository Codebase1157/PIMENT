"""Group-aware, stratified five-fold splits (Section 5.2).

Whole perceptual-hash groups are assigned to a single fold, so no two
exposures of one blade can fall on opposite sides of a split.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


# Canonical class ordering across the paper's experiments
D1_CLASSES = [
    "brown_spot", "yellow_leaf", "mosaic", "sett_rot", "banded_chlorosis",
    "healthy", "grassy_shoot", "dried_leaves", "smut", "brown_rust",
    "pokkah_boeng",
]
D2_CLASSES = ["healthy", "yellow", "mosaic", "rust", "red_rot"]
D3_CLASSES = ["healthy", "red_rot", "ring_spot", "eye_spot", "red_leaf_spot"]

HARMONISED_CLASSES = ["healthy", "yellow_leaf", "mosaic", "rust", "red_rot"]
# Harmonisation map from the native class name to the shared label
HARMONISATION = {
    # D1 -> harmonised
    "yellow_leaf": "yellow_leaf",
    "brown_rust": "rust",
    "healthy": "healthy",
    "mosaic": "mosaic",
    # D2 -> harmonised
    "yellow": "yellow_leaf",
    "rust": "rust",
    "red_rot": "red_rot",
    # D3 -> harmonised
}


def build_folds(items: Sequence[dict], groups: Sequence[int], n_splits: int = 5,
                seed: int = 42, val_share: float = 0.15) -> dict:
    """Return a leakage-controlled fold assignment.

    Parameters
    ----------
    items : list of dicts with keys ``path`` and ``label`` (int).
    groups : parallel sequence of group ids (from perceptual-hash grouping).
    n_splits : number of folds.
    seed : RNG seed.
    val_share : fraction of the training fold held out for validation.

    Returns
    -------
    folds : dict with ``train``, ``val`` and ``test`` lists per fold index.
    """
    labels = np.asarray([it["label"] for it in items], dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = {}
    rng = np.random.default_rng(seed)
    for f, (tr_idx, te_idx) in enumerate(sgkf.split(np.zeros(len(labels)), labels, groups=groups)):
        tr_groups = np.unique(groups[tr_idx])
        rng.shuffle(tr_groups)
        n_val_groups = max(1, int(round(val_share * len(tr_groups))))
        val_groups = set(tr_groups[:n_val_groups].tolist())
        val_mask = np.array([g in val_groups for g in groups[tr_idx]])
        val_idx = tr_idx[val_mask]
        tr_idx = tr_idx[~val_mask]
        folds[str(f)] = {
            "train": [items[i]["path"] for i in tr_idx],
            "val":   [items[i]["path"] for i in val_idx],
            "test":  [items[i]["path"] for i in te_idx],
        }
    return folds


def load_folds(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
