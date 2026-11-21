"""End-to-end photometric decomposition pipeline (Algorithm 1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .linearise import linearise_srgb
from .segmentation import segment_blade
from .dichromatic import dichromatic_split, glare_fraction
from .white_point import gloss_white_point
from .optical_density import optical_density
from .factorisation import (
    anchored_sparse_factorise,
    FactoriseConfig,
    leaf_relative_deviation,
    pigment_composition,
)


@dataclass
class Decomposition:
    """Container for the outputs of :func:`photometric_decompose`."""
    diffuse: np.ndarray        # (H, W, 3) linear sRGB after illuminant correction
    specular: np.ndarray       # (H, W)
    glare_mask: np.ndarray     # (H, W) float32 {0, 1}
    glare_fraction: float      # scalar reliability flag c_8
    mask: np.ndarray           # (H, W) bool blade Ω
    gamma: np.ndarray          # (3,) illuminant chromaticity
    illum_fallback: bool       # True if grey-world fallback was used
    C: np.ndarray              # (K, H, W) pigment-proxy maps
    E: np.ndarray              # (3, K) estimated absorbance matrix
    delta_C: np.ndarray        # (K, H, W) leaf-relative deviation
    p: np.ndarray              # (K, H, W) pigment composition
    od: np.ndarray             # (H, W, 3) optical density


def photometric_decompose(
    img_srgb: np.ndarray,
    E0: np.ndarray,
    mask: Optional[np.ndarray] = None,
    factorise_cfg: Optional[FactoriseConfig] = None,
) -> Decomposition:
    """Run Algorithm 1 on one sRGB image.

    Parameters
    ----------
    img_srgb : (H, W, 3) encoded sRGB in [0, 1]
    E0 : (3, K) prior absorbance matrix
    mask : optional (H, W) bool blade mask; segmented on the fly when absent
    factorise_cfg : :class:`FactoriseConfig` or None (defaults to paper settings)
    """
    img_lin = linearise_srgb(img_srgb)
    if mask is None:
        mask = segment_blade(img_srgb)
    gamma, fallback = gloss_white_point(img_lin, mask)
    diffuse, specular, gmask, _sigma_d = dichromatic_split(img_lin, mask, gamma=gamma)
    od = optical_density(diffuse)
    C, E = anchored_sparse_factorise(od, mask, E0=E0, cfg=factorise_cfg)
    delta_C = leaf_relative_deviation(C, mask)
    p = pigment_composition(C)
    g_frac = glare_fraction(gmask, mask)
    return Decomposition(
        diffuse=diffuse,
        specular=specular,
        glare_mask=gmask,
        glare_fraction=g_frac,
        mask=mask,
        gamma=gamma,
        illum_fallback=bool(fallback),
        C=C,
        E=E,
        delta_C=delta_C,
        p=p,
        od=od,
    )
