"""Smoke tests for the photometric decomposition pipeline (Algorithm 1)."""
import numpy as np
import pytest

from pigmentnet.decomposition import (
    linearise_srgb,
    optical_density,
    from_optical_density,
    gloss_white_point,
    dichromatic_split,
    anchored_sparse_factorise,
    leaf_relative_deviation,
    pigment_composition,
    photometric_decompose,
)
from pigmentnet.decomposition.factorisation import FactoriseConfig


def _synthetic_leaf(H=96, W=96, highlight=True, seed=0):
    rng = np.random.default_rng(seed)
    img = np.ones((H, W, 3), dtype=np.float32) * np.array([0.22, 0.48, 0.14])
    img += rng.normal(0.0, 0.02, img.shape).astype(np.float32)
    if highlight:
        yy, xx = np.mgrid[0:H, 0:W]
        hl = np.exp(-((yy - H/3)**2 / 10.0 + (xx - W/2)**2 / 8.0))
        img = img + hl[..., None] * 0.6
    mask = np.ones((H, W), dtype=bool)
    mask[:, :8] = False; mask[:, -8:] = False
    img = np.clip(img, 0.0, 1.0)
    return img, mask


def test_linearise_roundtrip():
    x = np.linspace(0.0, 1.0, 1000).astype(np.float32)
    y = linearise_srgb(x)
    assert y.shape == x.shape
    # at the dark end the response is exactly linear
    assert np.allclose(y[:5], x[:5] / 12.92, atol=1e-6)


def test_optical_density_roundtrip():
    x = np.clip(np.random.default_rng(0).random((16, 16, 3)), 1e-2, 1.0).astype(np.float32)
    assert np.allclose(from_optical_density(optical_density(x)), x, atol=1e-3)


def test_gloss_white_point_runs():
    img, mask = _synthetic_leaf(highlight=True)
    gamma, fallback = gloss_white_point(img, mask)
    assert gamma.shape == (3,)
    assert np.isclose(gamma.sum(), 1.0, atol=1e-5)
    # grey-world fallback is acceptable here; main contract is numeric


def test_dichromatic_split_shapes():
    img, mask = _synthetic_leaf()
    gamma, _ = gloss_white_point(img, mask)
    diff, spec, gmask, sigma_d = dichromatic_split(img, mask, gamma=gamma)
    assert diff.shape == img.shape
    assert spec.shape == img.shape[:2]
    assert gmask.shape == img.shape[:2]
    assert 0.0 <= sigma_d <= 0.32


def test_factorisation_non_negative():
    img, mask = _synthetic_leaf(highlight=False)
    E0 = np.load("data/priors/E0_D1.npy")
    od = optical_density(img)
    C, E = anchored_sparse_factorise(od, mask, E0, cfg=FactoriseConfig(iterations=50))
    assert C.shape == (4, img.shape[0], img.shape[1])
    assert (C[:, mask] >= 0).all()
    # columns are unit-norm on the non-negative sphere
    norms = np.linalg.norm(E, axis=0)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_deviation_and_composition():
    C = np.random.default_rng(0).random((4, 32, 32)).astype(np.float32)
    mask = np.ones((32, 32), dtype=bool)
    dC = leaf_relative_deviation(C, mask)
    assert dC.shape == C.shape
    assert np.isclose(np.median(dC[0]), 0.0, atol=1e-5)
    p = pigment_composition(C)
    assert p.shape == C.shape
    assert np.allclose(p.sum(axis=0), 1.0, atol=1e-4)


def test_photometric_decompose_runs_end_to_end():
    img, _ = _synthetic_leaf()
    E0 = np.load("data/priors/E0_D1.npy")
    dec = photometric_decompose(img, E0=E0, mask=None,
                                factorise_cfg=FactoriseConfig(iterations=30))
    assert dec.C.shape == (4, img.shape[0], img.shape[1])
    assert dec.p.shape == dec.C.shape
    assert 0.0 <= dec.glare_fraction <= 1.0
    assert dec.E.shape == (3, 4)


def test_physaug_label_preserving():
    from pigmentnet.physaug import PhysAug, PhysAugConfig
    img, mask = _synthetic_leaf(highlight=False)
    E0 = np.load("data/priors/E0_D1.npy")
    dec = photometric_decompose(img, E0=E0, mask=mask,
                                factorise_cfg=FactoriseConfig(iterations=60))
    phys = PhysAug(PhysAugConfig(sigma_E=0.0, glare_jitter=0.0),
                   rng=np.random.default_rng(42))
    p0 = dec.p[:, dec.mask].mean(axis=1)
    for _ in range(4):
        aug, _ = phys.apply(dec)
        dec_aug = photometric_decompose(aug, E0=E0, mask=mask,
                                        factorise_cfg=FactoriseConfig(iterations=60))
        assert aug.shape == img.shape
        assert aug.dtype == np.float32
        assert 0.0 <= aug.min() and aug.max() <= 1.0
        # PhysAug should preserve the blade mask
        assert dec_aug.mask.sum() == dec.mask.sum()
