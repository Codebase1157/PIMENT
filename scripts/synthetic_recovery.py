#!/usr/bin/env python3
"""Synthetic recovery of known mixing coefficients (Section 6.9, Table 16).

Construct known pigment maps, mix them with the measured prior E0, corrupt
at a specified SNR, run the anchored sparse factorisation and measure R² per
pigment.
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from pigmentnet.decomposition.factorisation import (
    anchored_sparse_factorise, FactoriseConfig,
)
from pigmentnet.decomposition.optical_density import from_optical_density


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def _rotate_column(E: np.ndarray, col: int, deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    Rm = np.array([[np.cos(th), -np.sin(th), 0.0],
                   [np.sin(th),  np.cos(th), 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)
    E2 = E.copy()
    E2[:, col] = Rm @ E2[:, col]
    E2 = E2 / (np.linalg.norm(E2, axis=0, keepdims=True) + 1e-8)
    return E2


def generate_scene(H, W, K, rng):
    maps = np.abs(rng.standard_normal((K, H, W))).astype(np.float32)
    maps = (maps - maps.mean(axis=(-2, -1), keepdims=True)) * 0.3 + 0.5
    maps = np.clip(maps, 0.0, None)
    return maps


def _add_noise(x: np.ndarray, snr_db: float, rng) -> np.ndarray:
    sig = float(np.mean(x * x))
    n_pow = sig / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(n_pow)
    return x + noise


def run_one(E0, snr_db=30.0, prior_rotation=0.0, rotate_col=3,
            sparsity=0.05, anchoring=1.0, H=96, W=96, scenes=5, seed=0):
    rng = np.random.default_rng(seed)
    K = E0.shape[1]
    r2s = []
    E_truth = _rotate_column(E0, rotate_col, 0.0)
    E_prior = _rotate_column(E0, rotate_col, prior_rotation)
    for _ in range(scenes):
        maps = generate_scene(H, W, K, rng)
        OD = np.einsum("ck,khw->chw", E_truth, maps)
        OD = _add_noise(OD, snr_db=snr_db, rng=rng)
        img = from_optical_density(np.moveaxis(OD, 0, -1))
        from pigmentnet.decomposition.optical_density import optical_density
        OD_hat = optical_density(img)
        mask = np.ones((H, W), dtype=bool)
        cfg = FactoriseConfig(sparsity=sparsity, anchoring=anchoring)
        recovered, _ = anchored_sparse_factorise(OD_hat, mask, E_prior, cfg=cfg)
        per_pig = [float(_r2(maps[k], recovered[k])) for k in range(K)]
        r2s.append(per_pig)
    r2s = np.mean(r2s, axis=0).tolist()
    return r2s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--out", default="results/synthetic_recovery.json")
    args = ap.parse_args()

    E0 = np.load(args.prior).astype(np.float32)
    settings = [
        ("Reference",           dict(snr_db=30.0, prior_rotation=0.0,  sparsity=0.05, anchoring=1.0)),
        ("Prior rotated 5°",    dict(snr_db=30.0, prior_rotation=5.0)),
        ("Prior rotated 10°",   dict(snr_db=30.0, prior_rotation=10.0)),
        ("Sparsity λ = 0.01",   dict(snr_db=30.0, sparsity=0.01)),
        ("Sparsity λ = 0.20",   dict(snr_db=30.0, sparsity=0.20)),
        ("Anchoring μ = 0.1",   dict(snr_db=30.0, anchoring=0.1)),
        ("Anchoring μ = 10",    dict(snr_db=30.0, anchoring=10.0)),
        ("Noise 40 dB SNR",     dict(snr_db=40.0)),
        ("Noise 20 dB SNR",     dict(snr_db=20.0)),
    ]
    results = {}
    for name, kw in settings:
        r2s = run_one(E0, **kw)
        results[name] = {"R2": {k: r for k, r in zip(["chl", "ylw", "red", "brn"], r2s)},
                         "mean_R2": float(np.mean(r2s))}
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
