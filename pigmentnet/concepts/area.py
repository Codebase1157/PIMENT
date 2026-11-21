"""Area concepts c_1, c_2, c_3 (Eq. 26).

Each measures the share of Ω dominated by one pigment composition, through
a soft threshold:

    c_k = (1/|Ω|) Σ_{x ∈ Ω} σ((p_k(x) − t_k) / T)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _soft_share(p_k: torch.Tensor, mask: torch.Tensor, t_k: torch.Tensor, T: torch.Tensor):
    denom = mask.sum(dim=(-2, -1)).clamp_min(1.0)
    score = torch.sigmoid((p_k - t_k.view(-1, 1, 1, 1).squeeze(-1)) / T.clamp_min(1e-4))
    score = score * mask
    return score.sum(dim=(-2, -1)) / denom


def area_concepts(p: torch.Tensor, mask: torch.Tensor,
                  thresholds: torch.Tensor, temperature: torch.Tensor):
    """Compute (c1, c2, c3) from the pigment composition p (K, H, W).

    Parameters
    ----------
    p : (B, K, H, W) composition map with K = 4 (chl, ylw, red, brn)
    mask : (B, 1, H, W) blade mask Ω
    thresholds : (3,) learnable t_k for yellow, red, necrotic
    temperature : (1,) learnable T
    """
    t_ylw, t_red, t_brn = thresholds[0], thresholds[1], thresholds[2]
    T = temperature
    c1 = _single_fraction(p[:, 1:2], mask, t_ylw, T)
    c2 = _single_fraction(p[:, 2:3], mask, t_red, T)
    c3 = _single_fraction(p[:, 3:4], mask, t_brn, T)
    return c1, c2, c3


def _single_fraction(p_k: torch.Tensor, mask: torch.Tensor, t_k, T) -> torch.Tensor:
    score = torch.sigmoid((p_k - t_k) / T.clamp_min(1e-4))
    score = score * mask
    denom = mask.sum(dim=(-2, -1, -3)).clamp_min(1.0)
    return score.sum(dim=(-2, -1, -3)) / denom


def chlorotic_fraction(p, mask, t, T): return _single_fraction(p[:, 1:2], mask, t, T)
def red_fraction      (p, mask, t, T): return _single_fraction(p[:, 2:3], mask, t, T)
def necrotic_fraction (p, mask, t, T): return _single_fraction(p[:, 3:4], mask, t, T)
