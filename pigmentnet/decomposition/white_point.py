"""Gloss white point via inverse-intensity chromaticity (Section 4.2.3).

Specular pixels are themselves a white reference: in the inverse-intensity
chromaticity space, the chromaticity of a highlight pixel is affine in the
reciprocal of its total intensity, and the intercept is the illuminant
chromaticity (Tan, Nishino & Ikeuchi 2004).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def _huber_fit(x: np.ndarray, y: np.ndarray, delta: float = 0.01):
    """Robust affine fit y = γ + β x with a Huber penalty."""

    def residuals(p):
        return p[0] + p[1] * x - y

    sol = least_squares(residuals, x0=[float(y.mean()), 0.0], loss="huber", f_scale=delta)
    return float(sol.x[0]), float(sol.x[1])


def _highlight_mask(img: np.ndarray, mask: np.ndarray, q: float = 0.90,
                    spread: float = 0.22, clip: float = 0.985) -> np.ndarray:
    """Eq. 9. Pixels above the q-th intensity percentile with low chromaticity
    spread and no clipped channel."""
    energy = img.sum(axis=-1)
    if mask.sum() == 0:
        return np.zeros_like(mask)
    q_val = np.quantile(energy[mask], q)
    chroma = img / (energy[..., None] + 1e-8)
    chroma_spread = chroma.max(axis=-1) - chroma.min(axis=-1)
    max_c = img.max(axis=-1)
    hl = mask & (energy > q_val) & (chroma_spread < spread) & (max_c < clip)
    return hl


def gloss_white_point(img: np.ndarray, mask: np.ndarray,
                      q: float = 0.90, spread: float = 0.22,
                      clip: float = 0.985, min_hl: int = 60) -> tuple[np.ndarray, bool]:
    """Estimate the illuminant chromaticity γ.

    Returns (γ, fallback_flag). If fewer than ``min_hl`` valid highlights are
    found the estimate falls back to grey-world and ``fallback_flag`` is True.
    """
    hl = _highlight_mask(img, mask, q, spread, clip)
    if int(hl.sum()) < 150:  # relaxation step
        hl = _highlight_mask(img, mask, q, spread, clip=1.0)

    if int(hl.sum()) < min_hl:
        return grey_world(img, mask), True

    pix = img[hl]
    energy = pix.sum(axis=1) + 1e-8
    inv_e = 1.0 / energy
    gamma = np.zeros(3, dtype=np.float32)
    for c in range(3):
        sigma_c = pix[:, c] / energy
        g, _ = _huber_fit(inv_e, sigma_c)
        gamma[c] = g
    s = gamma.sum()
    if s <= 0:
        return grey_world(img, mask), True
    gamma = gamma / s  # keep chromaticity on the simplex
    return gamma.astype(np.float32), False


def grey_world(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pix = img[mask]
    if pix.size == 0:
        return np.ones(3, dtype=np.float32) / 3.0
    mean = pix.mean(axis=0)
    gamma = mean / (mean.sum() + 1e-8)
    return gamma.astype(np.float32)


def von_kries_correct(img: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """I'_c = I_c / (3 γ_c). Division keeps the diffuse floor comparable
    across illuminants."""
    g = np.asarray(gamma, dtype=np.float32).reshape(1, 1, 3)
    return (img / (3.0 * g + 1e-8)).astype(np.float32)
