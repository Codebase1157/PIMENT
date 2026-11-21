"""Anchored sparse pigment factorisation (Section 4.2.5, Eqs. 12–15).

Solves
    min_{E, C ≥ 0}  ||OD − E C||_F^2 + λ ||C||_1 + μ ||E − E_0||_F^2
    s.t.  ||e_k||_2 = 1

by projected gradient descent. C is updated at every step, E every tenth step
(Algorithm 1).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class FactoriseConfig:
    iterations: int = 200
    sparsity: float = 0.05        # λ
    anchoring: float = 1.0        # μ
    dict_update_every: int = 10
    eta_c: float = 0.10
    eta_e: float = 0.02
    non_negative_sphere: bool = True


def _project_sphere(E: np.ndarray, non_negative: bool = True) -> np.ndarray:
    if non_negative:
        E = np.clip(E, 0.0, None)
    norm = np.linalg.norm(E, axis=0, keepdims=True) + 1e-8
    return E / norm


def _init_C(E0: np.ndarray, OD: np.ndarray) -> np.ndarray:
    # [E0+ OD]+ as in Algorithm 1 line 6
    pinv = np.linalg.pinv(E0)
    C0 = pinv @ OD
    return np.clip(C0, 0.0, None)


def anchored_sparse_factorise(
    od: np.ndarray,
    mask: np.ndarray,
    E0: np.ndarray,
    cfg: FactoriseConfig | None = None,
):
    """Factorise OD in Ω into E (3 × K) and C (K × H × W).

    Parameters
    ----------
    od : (H, W, 3)
    mask : (H, W) bool, blade Ω
    E0 : (3, K) column-normalised prior absorbance matrix
    cfg : :class:`FactoriseConfig`

    Returns
    -------
    C : (K, H, W) float32, non-negative pigment-proxy maps
    E : (3, K) float32, estimated absorbance matrix
    """
    cfg = cfg or FactoriseConfig()
    K = E0.shape[1]
    H, W, _ = od.shape

    pix = od[mask]                     # (N, 3)
    OD = pix.T.astype(np.float32)      # (3, N)
    E = _project_sphere(E0.astype(np.float32).copy(), cfg.non_negative_sphere)
    C = _init_C(E, OD).astype(np.float32)

    for t in range(1, cfg.iterations + 1):
        grad_C = 2.0 * (E.T @ (E @ C - OD)) + cfg.sparsity
        C = np.clip(C - cfg.eta_c * grad_C, 0.0, None)
        if t % cfg.dict_update_every == 0:
            grad_E = 2.0 * ((E @ C - OD) @ C.T) + 2.0 * cfg.anchoring * (E - E0)
            E = E - cfg.eta_e * grad_E
            E = _project_sphere(E, cfg.non_negative_sphere)

    maps = np.zeros((K, H, W), dtype=np.float32)
    idx = np.where(mask)
    for k in range(K):
        maps[k][idx] = C[k]
    return maps, E.astype(np.float32)


def leaf_relative_deviation(C: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ΔC_k(x) = C_k(x) - median_Ω C_k (Eq. 16)."""
    K = C.shape[0]
    out = np.zeros_like(C)
    for k in range(K):
        pix = C[k][mask]
        if pix.size == 0:
            continue
        m = float(np.median(pix))
        out[k][mask] = C[k][mask] - m
    return out


def pigment_composition(C: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """p_k(x) = C_k(x) / (sum_j C_j(x) + ε) (Eq. 17).

    Cancels leaf-thickness and exposure factors that survive the logarithm.
    """
    total = C.sum(axis=0, keepdims=True) + eps
    return (C / total).astype(np.float32)


def E0_from_pixels(anchors: dict[str, np.ndarray]) -> np.ndarray:
    """Build the prior absorbance matrix from agronomist-marked RGB pixels.

    `anchors` maps pigment name -> (N, 3) linear RGB values. The returned
    columns are ordered (chl, ylw, red, brn) and unit-normalised.
    """
    from .optical_density import optical_density
    order = ["chl", "ylw", "red", "brn"]
    cols = []
    for name in order:
        px = np.asarray(anchors[name], dtype=np.float32).reshape(-1, 3)
        mean_rgb = px.mean(axis=0, keepdims=True).reshape(1, 1, 3)
        od = optical_density(mean_rgb).reshape(3)
        od = od / (np.linalg.norm(od) + 1e-8)
        cols.append(od)
    return np.stack(cols, axis=1).astype(np.float32)
