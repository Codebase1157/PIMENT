"""Image preprocessing and decomposition caching."""
from __future__ import annotations

import io
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image

from ..decomposition import photometric_decompose
from ..decomposition.factorisation import FactoriseConfig


def prepare_image(path: str | Path, target: int = 384) -> np.ndarray:
    """Load an image, resize to ``target × target`` with aspect preserved
    (letterbox), and return a float32 sRGB array in [0, 1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = target / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (target, target), (0, 0, 0))
    canvas.paste(img, ((target - nw) // 2, (target - nh) // 2))
    arr = np.asarray(canvas).astype(np.float32) / 255.0
    return arr


def decompose_and_cache(path: str | Path, out_dir: str | Path, E0: np.ndarray,
                        target: int = 384) -> dict:
    """Run Algorithm 1 on one image and write its outputs to ``out_dir``.

    Returns a JSON-serialisable record with per-image stats.
    """
    arr = prepare_image(path, target=target)
    dec = photometric_decompose(arr, E0=E0, factorise_cfg=FactoriseConfig())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(path).stem
    np.savez_compressed(
        out_dir / f"{name}.npz",
        diffuse=(dec.diffuse * 255).astype(np.uint8),
        specular=(dec.specular * 255).astype(np.uint8),
        glare=dec.glare_mask.astype(np.uint8),
        mask=dec.mask.astype(np.uint8),
        gamma=dec.gamma.astype(np.float32),
        fallback=np.array([int(dec.illum_fallback)], dtype=np.uint8),
        C=dec.C.astype(np.float32),
        E=dec.E.astype(np.float32),
    )
    return {
        "name": name,
        "gamma": [float(x) for x in dec.gamma],
        "fallback": bool(dec.illum_fallback),
        "glare_fraction": float(dec.glare_fraction),
    }
