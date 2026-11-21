"""PhysAug — physics-consistent augmentation (Section 4.3, Eqs. 18–21).

Colour jitter multiplies RGB channels at random, moving a leaf along exactly
the axis that encodes its diagnosis. PhysAug holds pigment concentrations
fixed and redraws the imaging conditions (illuminant, exposure, device
response), re-synthesising the observation from the decomposition. Because
C(x̃) = C(x) the concept vector — and with it the class label — is preserved
by construction up to the device perturbation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .decomposition.linearise import encode_srgb

# sRGB D65 primary matrix (XYZ to linear sRGB, IEC 61966-2-1)
_M_XYZ_TO_RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
], dtype=np.float32)


def planckian_xyz(Tc: float) -> np.ndarray:
    """Approximate CIE 1931 chromaticity xyz of a black-body at Tc K.

    Follows Krystek's rational approximation (2,800–7,500 K range), returning
    an XYZ-like triple normalised so x + y + z ≈ 1.
    """
    Tc = float(np.clip(Tc, 1000.0, 15000.0))
    if Tc < 4000.0:
        xc = (-0.2661239e9 / Tc**3 - 0.2343580e6 / Tc**2 + 0.8776956e3 / Tc + 0.179910)
    else:
        xc = (-3.0258469e9 / Tc**3 + 2.1070379e6 / Tc**2 + 0.2226347e3 / Tc + 0.240390)
    if Tc < 2222.0:
        yc = -1.1063814 * xc**3 - 1.34811020 * xc**2 + 2.18555832 * xc - 0.20219683
    elif Tc < 4000.0:
        yc = -0.9549476 * xc**3 - 1.37418593 * xc**2 + 2.09137015 * xc - 0.16748867
    else:
        yc = 3.0817580 * xc**3 - 5.87338670 * xc**2 + 3.75112997 * xc - 0.37001483
    zc = 1.0 - xc - yc
    xyz = np.array([xc, yc, zc], dtype=np.float32)
    xyz = xyz / (xyz.sum() + 1e-8)
    return xyz


def planckian_illuminant(Tc: float) -> np.ndarray:
    """Chromaticity γ of a black-body illuminant at Tc K (linear sRGB simplex)."""
    xyz = planckian_xyz(Tc)
    rgb = _M_XYZ_TO_RGB @ xyz
    rgb = np.clip(rgb, 1e-6, None)
    rgb = rgb / (np.linalg.norm(rgb, 1) + 1e-8)
    return rgb.astype(np.float32)


@dataclass
class PhysAugConfig:
    Tc_low: float = 2800.0
    Tc_high: float = 7500.0
    EV_low: float = -1.5
    EV_high: float = 1.5
    sigma_E: float = 0.03
    glare_jitter: float = 0.15   # ±fractional jitter of m_s
    probability: float = 1.0


class PhysAug:
    """Draw a physics-consistent augmentation (Eq. 18).

    The caller passes either an already-decomposed scene (E, C, specular,
    original illuminant implicit in the diffuse term) or the raw outputs of
    :func:`pigmentnet.decomposition.photometric_decompose`.
    """

    def __init__(self, cfg: PhysAugConfig | None = None, rng: np.random.Generator | None = None):
        self.cfg = cfg or PhysAugConfig()
        self.rng = rng if rng is not None else np.random.default_rng()

    # -----------------------------------------------------------------
    def sample(self):
        """Draw one triple (γ', τ, device perturbation per column)."""
        c = self.cfg
        Tc = float(self.rng.uniform(c.Tc_low, c.Tc_high))
        gamma = planckian_illuminant(Tc)
        ev = float(self.rng.uniform(c.EV_low, c.EV_high))
        tau = ev * np.log(2.0)
        return {
            "gamma": gamma,
            "exposure_tau": tau,
            "EV": ev,
            "Tc": Tc,
        }

    # -----------------------------------------------------------------
    def apply(self, decomposition, draw=None):
        """Re-synthesise one augmented view from a decomposition.

        Parameters
        ----------
        decomposition : :class:`pigmentnet.decomposition.Decomposition`
        draw : dict or None (one :meth:`sample`)
        """
        d = decomposition
        if draw is None:
            draw = self.sample()

        E = d.E.copy()
        # device response perturbation: rotate each absorbance column slightly
        if self.cfg.sigma_E > 0:
            eps = self.rng.normal(0.0, self.cfg.sigma_E, size=E.shape).astype(np.float32)
            Ep = E * (1.0 + eps)
            Ep = Ep / (np.linalg.norm(Ep, axis=0, keepdims=True) + 1e-8)
        else:
            Ep = E

        gamma = draw["gamma"].reshape(1, 1, 3)
        tau = draw["exposure_tau"]
        exposure = float(np.exp(tau))

        K, H, W = d.C.shape
        C_flat = d.C.reshape(K, -1)                  # (K, N)
        OD_new = Ep @ C_flat                         # (3, N)
        OD_new = OD_new.reshape(3, H, W).transpose(1, 2, 0)
        body = 10.0 ** (-OD_new)                     # linear reflectance

        specular = d.specular[..., None]
        if self.cfg.glare_jitter > 0:
            specular = specular * (1.0 + self.rng.uniform(
                -self.cfg.glare_jitter, self.cfg.glare_jitter))

        linear = exposure * gamma * body + specular * gamma
        linear = np.clip(linear, 0.0, 1.0)
        out = encode_srgb(linear)
        out[~d.mask] = 0.0
        return out.astype(np.float32), draw
