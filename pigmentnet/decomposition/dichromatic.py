"""Dichromatic glare split (Section 4.2.2, Eqs. 5–7)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from .white_point import gloss_white_point, von_kries_correct


def _diffuse_floor(img_corr: np.ndarray, mask: np.ndarray) -> float:
    """σ_d: 25th percentile of minimum chromaticity over Ω, clipped to [0, 0.32]."""
    pix = img_corr[mask]
    if pix.size == 0:
        return 0.0
    energy = pix.sum(axis=1, keepdims=True) + 1e-8
    chroma = pix / energy
    min_chroma = chroma.min(axis=1)
    sigma_d = float(np.percentile(min_chroma, 25))
    return float(np.clip(sigma_d, 0.0, 0.32))


def dichromatic_split(
    img_linear: np.ndarray,
    mask: np.ndarray,
    gamma: np.ndarray | None = None,
    median_size: int = 3,
):
    """Return the illuminant-corrected diffuse image, the specular magnitude
    and the glare mask (Eqs. 6, 7).

    Parameters
    ----------
    img_linear : (H, W, 3)
        Linearised image in [0, 1].
    mask : (H, W) bool
        Blade segmentation Ω.
    gamma : (3,) or None
        Illuminant chromaticity. If ``None`` it is estimated internally by
        :func:`gloss_white_point`.
    median_size : int
        Median filter radius applied to the specular layer.
    """
    if gamma is None:
        gamma, _ = gloss_white_point(img_linear, mask)

    img_c = von_kries_correct(img_linear, gamma)
    sigma_d = _diffuse_floor(img_c, mask)

    energy = img_c.sum(axis=-1, keepdims=False)
    min_c = img_c.min(axis=-1)
    denom = max(1e-6, 1.0 - 3.0 * sigma_d)
    specular = (min_c - sigma_d * energy) / denom
    specular = np.clip(specular, 0.0, min_c)
    specular = median_filter(specular, size=median_size)

    diffuse = img_c - specular[..., None]
    diffuse = np.clip(diffuse, 0.0, 1.0)
    diffuse[~mask] = 0.0

    gmask = glare_mask(specular, energy, mask)
    return diffuse, specular, gmask, float(sigma_d)


def glare_mask(specular: np.ndarray, energy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Eq. 7 glare reliability flag."""
    cond1 = (3.0 * specular) > (0.2 * energy)
    cond2 = specular > 0.025
    g = (cond1 & cond2 & mask).astype(np.float32)
    return g


def glare_fraction(g: np.ndarray, mask: np.ndarray) -> float:
    n = int(mask.sum())
    return float(g.sum() / n) if n else 0.0
