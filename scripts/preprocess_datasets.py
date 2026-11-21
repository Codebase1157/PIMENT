#!/usr/bin/env python3
"""Decompose every image in each corpus and cache the result.

Usage
-----
    python scripts/preprocess_datasets.py --root raw --out processed [--priors data/priors]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from pigmentnet.data.preprocess import decompose_and_cache
from pigmentnet.data.folds import D1_CLASSES, D2_CLASSES, D3_CLASSES


def _collect(root: Path, class_list) -> list[dict]:
    items = []
    for cls_idx, cls in enumerate(class_list):
        cls_dir = root / cls
        if not cls_dir.exists():
            # case-insensitive and hyphen/underscore-insensitive fallback
            for cand in root.iterdir():
                if cand.is_dir() and cand.name.lower().replace("-", "_") == cls.lower():
                    cls_dir = cand
                    break
        if not cls_dir.exists():
            print(f"warn: class '{cls}' not found under {root}")
            continue
        for p in sorted(cls_dir.glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                items.append({"path": str(p), "label": cls_idx, "class": cls})
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--priors", type=Path, default=Path("data/priors"))
    args = ap.parse_args()

    for corpus, classes in (("D1", D1_CLASSES), ("D2", D2_CLASSES), ("D3", D3_CLASSES)):
        src = args.root / corpus
        if not src.exists():
            continue
        items = _collect(src, classes)
        if not items:
            continue
        E0 = np.load(args.priors / f"E0_{corpus}.npy")
        out_dir = args.out / corpus / "cache"
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for it in tqdm(items, desc=corpus):
            info = decompose_and_cache(it["path"], out_dir, E0=E0)
            records.append({**it, "cache": str(out_dir / f"{info['name']}.npz"),
                            "gamma": info["gamma"], "fallback": info["fallback"],
                            "glare_fraction": info["glare_fraction"]})
        (args.out / corpus / "index.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
