"""sRGB transfer-curve inversion (Eq. 4).

Both the dichromatic model and the Beer–Lambert relation hold for linear
intensity only, so the encoded sRGB value must be inverted channel-wise before
anything else.
"""
from __future__ import annotations

import numpy as np

try:
    import torch  # torch is optional for the decomposition
except ImportError:
    torch = None  # type: ignore

_SRGB_THRESH = 0.04045
_SRGB_A = 1.055
_SRGB_K = 12.92
_SRGB_GAMMA = 2.4


def linearise_srgb(img):
    """Invert the sRGB transfer curve (piecewise, Eq. 4).

    Accepts NumPy arrays or torch tensors with values in [0, 1].
    """
    if torch is not None and isinstance(img, torch.Tensor):
        u = img.clamp(0.0, 1.0)
        return torch.where(
            u <= _SRGB_THRESH,
            u / _SRGB_K,
            ((u + (_SRGB_A - 1.0)) / _SRGB_A).pow(_SRGB_GAMMA),
        )
    u = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
    out = np.where(
        u <= _SRGB_THRESH,
        u / _SRGB_K,
        ((u + (_SRGB_A - 1.0)) / _SRGB_A) ** _SRGB_GAMMA,
    )
    return out.astype(img.dtype if hasattr(img, "dtype") else np.float32)


def encode_srgb(lin):
    """Forward sRGB transfer, the inverse of :func:`linearise_srgb`."""
    if torch is not None and isinstance(lin, torch.Tensor):
        u = lin.clamp(0.0, 1.0)
        return torch.where(
            u <= _SRGB_THRESH / _SRGB_K,
            u * _SRGB_K,
            _SRGB_A * u.pow(1.0 / _SRGB_GAMMA) - (_SRGB_A - 1.0),
        )
    u = np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    out = np.where(
        u <= _SRGB_THRESH / _SRGB_K,
        u * _SRGB_K,
        _SRGB_A * (u ** (1.0 / _SRGB_GAMMA)) - (_SRGB_A - 1.0),
    )
    return out.astype(lin.dtype if hasattr(lin, "dtype") else np.float32)
