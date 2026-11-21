"""Texture concepts c_4 (heterogeneity), c_5 (patchiness), c_6 (lesion
density) and c_7 (stripe coherence) (Eqs. 27–30)."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Helpers
def _local_mean(x: torch.Tensor, win: int, mask: torch.Tensor) -> torch.Tensor:
    pad = win // 2
    k = torch.ones(1, 1, win, win, device=x.device, dtype=x.dtype) / (win * win)
    denom = F.conv2d(mask, k, padding=pad).clamp_min(1e-4)
    num = F.conv2d(x * mask, k, padding=pad)
    return num / denom


def _local_var(x, win, mask):
    m = _local_mean(x, win, mask)
    m2 = _local_mean(x * x, win, mask)
    return (m2 - m * m).clamp_min(0.0), m


# ----------------------------------------------------------------------
def heterogeneity(p_chl: torch.Tensor, mask: torch.Tensor, win: int = 9, eps: float = 1e-4) -> torch.Tensor:
    """c_4 — local coefficient of variation of the chlorophyll composition.

    c_4 = mean_{x ∈ Ω} sd_N(x)(p_chl) / (mean_N(x)(p_chl) + ε)
    """
    var, mean = _local_var(p_chl, win, mask)
    cov = torch.sqrt(var) / (mean + eps)
    denom = mask.sum(dim=(-2, -1, -3)).clamp_min(1.0)
    return (cov * mask).sum(dim=(-2, -1, -3)) / denom


def patchiness_moran(p_chl: torch.Tensor, mask: torch.Tensor,
                     subsample: int = 4) -> torch.Tensor:
    """c_5 — global Moran's I of p_chl with 4-neighbour weights.

    Downsampled by ``subsample`` to keep memory bounded.
    """
    x = F.avg_pool2d(p_chl, subsample)
    m = F.avg_pool2d(mask, subsample)
    m = (m > 0.5).float()
    mean = (x * m).sum(dim=(-2, -1, -3)) / m.sum(dim=(-2, -1, -3)).clamp_min(1.0)
    dx = (x - mean.view(-1, 1, 1, 1)) * m
    # four-neighbour shift sum
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neigh = torch.zeros_like(dx)
    w_sum = torch.zeros_like(m)
    for dy, ddx in shifts:
        shifted = torch.roll(dx, shifts=(dy, ddx), dims=(-2, -1))
        wshift = torch.roll(m, shifts=(dy, ddx), dims=(-2, -1)) * m
        neigh = neigh + shifted * wshift
        w_sum = w_sum + wshift
    num = (dx * neigh).sum(dim=(-2, -1, -3))
    denom_var = (dx * dx).sum(dim=(-2, -1, -3)).clamp_min(1e-6)
    denom_w = w_sum.sum(dim=(-2, -1, -3)).clamp_min(1.0)
    N = m.sum(dim=(-2, -1, -3)).clamp_min(1.0)
    return (N / denom_w) * (num / denom_var)


def _laplacian_of_gaussian(x: torch.Tensor, sigma: float = 2.5) -> torch.Tensor:
    size = max(5, int(2 * (3 * sigma) + 1))
    size |= 1
    coords = torch.arange(size, dtype=x.dtype, device=x.device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    gy = g.view(1, 1, -1, 1)
    gx = g.view(1, 1, 1, -1)
    xs = F.conv2d(x, gy, padding=(size // 2, 0))
    xs = F.conv2d(xs, gx, padding=(0, size // 2))
    lap_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    lap = F.conv2d(xs, lap_k, padding=1)
    return -(sigma ** 2) * lap


def lesion_density(p_brn: torch.Tensor, mask: torch.Tensor, t: torch.Tensor,
                   T: torch.Tensor, area_norm: float = 1e-4, sigma: float = 2.5) -> torch.Tensor:
    """c_6 — soft Laplacian-of-Gaussian peaks of p_brn per unit area."""
    L = _laplacian_of_gaussian(p_brn, sigma=sigma)
    # soft local-maximum indicator via max pooling comparison
    max3 = F.max_pool2d(L, kernel_size=3, stride=1, padding=1)
    is_peak = (L >= max3 - 1e-5).float()
    score = is_peak * torch.sigmoid((L - t) / T.clamp_min(1e-4))
    area = mask.sum(dim=(-2, -1, -3)).clamp_min(1.0)
    return (area_norm / 1.0) * (score * mask).sum(dim=(-2, -1, -3)) / area * 1e4


def stripe_coherence(p_chl: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """c_7 — ((λ1 − λ2) / (λ1 + λ2))^2 for the blade-averaged structure
    tensor J of p_chl."""
    # Sobel gradients
    sx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=p_chl.dtype,
                      device=p_chl.device).view(1, 1, 3, 3) / 4.0
    sy = sx.transpose(-1, -2)
    gx = F.conv2d(p_chl, sx, padding=1)
    gy = F.conv2d(p_chl, sy, padding=1)
    # Gaussian smoothing on products (ρ)
    def _blur(t):
        k = torch.tensor([[1, 4, 6, 4, 1]], dtype=p_chl.dtype, device=p_chl.device) / 16.0
        k2 = (k.T @ k).view(1, 1, 5, 5)
        return F.conv2d(t, k2, padding=2)
    Jxx = _blur(gx * gx) * mask
    Jyy = _blur(gy * gy) * mask
    Jxy = _blur(gx * gy) * mask
    denom = mask.sum(dim=(-2, -1, -3)).clamp_min(1.0)
    jxx = Jxx.sum(dim=(-2, -1, -3)) / denom
    jyy = Jyy.sum(dim=(-2, -1, -3)) / denom
    jxy = Jxy.sum(dim=(-2, -1, -3)) / denom
    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = (tr * tr - 4.0 * det).clamp_min(0.0).sqrt()
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    num = (l1 - l2)
    den = (l1 + l2).clamp_min(1e-8)
    return (num / den) ** 2
