"""Perceptual-hash de-duplication (Section 5.2, leakage control).

Near-duplicates are removed across folds by perceptual hashing, and repeat
exposures of one leaf are confined to a single fold. The audit on D1 found
1,051 exact-duplicate groups and 71 perceptual-hash repeat groups that
straddled random split boundaries.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image


def perceptual_hash(image_path: str | Path, hash_size: int = 16) -> str:
    """Return a hexadecimal pHash of the image at ``image_path``."""
    try:
        import imagehash
        img = Image.open(image_path).convert("L")
        return str(imagehash.phash(img, hash_size=hash_size))
    except ImportError:
        # fallback – resize + DCT via numpy
        img = np.asarray(Image.open(image_path).convert("L").resize((32, 32)))
        dct = _dct2(img.astype(np.float32))
        low = dct[:hash_size, :hash_size]
        med = np.median(low[1:])        # skip DC
        bits = (low > med).astype(np.uint8).flatten()
        return _bits_to_hex(bits)


def _dct2(x: np.ndarray) -> np.ndarray:
    from scipy.fftpack import dct
    return dct(dct(x, axis=0, norm="ortho"), axis=1, norm="ortho")


def _bits_to_hex(bits: np.ndarray) -> str:
    n = (len(bits) + 3) // 4
    bstr = "".join(str(b) for b in bits)
    return hex(int(bstr, 2))[2:].zfill(n)


def hamming(h1: str, h2: str) -> int:
    n = max(len(h1), len(h2))
    b1 = int(h1, 16)
    b2 = int(h2, 16)
    return bin(b1 ^ b2).count("1")


def compute_phash(paths: Iterable[str | Path], hash_size: int = 16) -> dict[str, str]:
    """Return a mapping ``{path: hash}``."""
    out = {}
    for p in paths:
        out[str(p)] = perceptual_hash(p, hash_size=hash_size)
    return out


def group_by_phash(hashes: dict[str, str], threshold: int = 6) -> list[list[str]]:
    """Union–find grouping on pHash Hamming distance."""
    paths = list(hashes.keys())
    parent = {p: p for p in paths}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # O(N^2) naive — fine for a few thousand images; swap in BK-tree for large sets.
    for i, p in enumerate(paths):
        h1 = hashes[p]
        for q in paths[i + 1:]:
            if hamming(h1, hashes[q]) <= threshold:
                union(p, q)

    groups: dict[str, list[str]] = {}
    for p in paths:
        groups.setdefault(find(p), []).append(p)
    return list(groups.values())
