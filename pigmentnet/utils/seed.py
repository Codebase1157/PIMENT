"""Deterministic seeding for reproducible 5 × 5 grids.

The paper reports five seeds × five folds. These are the fixed seeds.
"""
from __future__ import annotations

import os
import random
import numpy as np

# The canonical seed set used throughout the paper. Any change breaks
# cross-run comparability with the reported numbers.
SEEDS = (42, 1337, 2024, 20260101, 73856093)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        try:
            torch.use_deterministic_algorithms(deterministic, warn_only=True)
        except Exception:
            pass
    except ImportError:
        pass
