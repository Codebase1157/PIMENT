"""Optical density conversion (Section 4.2.4, Eq. 11).

OD_c(x) = -log10(max(D_c(x), 1e-3) / I_{0,c}) with I_0 = 1 after correction.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-3
_I0 = 1.0


def optical_density(diffuse: np.ndarray) -> np.ndarray:
    """Convert a diffuse linear-sRGB image to optical density."""
    d = np.clip(np.asarray(diffuse, dtype=np.float32), _EPS, None)
    return -np.log10(d / _I0).astype(np.float32)


def from_optical_density(od: np.ndarray) -> np.ndarray:
    """Inverse of :func:`optical_density`."""
    return np.clip((10.0 ** (-od)) * _I0, 0.0, 1.0).astype(np.float32)
