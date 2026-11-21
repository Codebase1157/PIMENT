#!/usr/bin/env python3
"""Compute perceptual hashes and group near-duplicates, as used for leakage
control (Section 5.2)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from pigmentnet.data.phash import perceptual_hash, group_by_phash


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/hash_groups"))
    ap.add_argument("--threshold", type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for corpus_dir in sorted(args.root.iterdir()):
        if not corpus_dir.is_dir():
            continue
        name = corpus_dir.name
        idx_path = corpus_dir / "index.json"
        if not idx_path.exists():
            continue
        items = json.loads(idx_path.read_text())
        hashes = {}
        for it in tqdm(items, desc=f"pHash {name}"):
            hashes[it["path"]] = perceptual_hash(it["path"])
        groups = group_by_phash(hashes, threshold=args.threshold)
        (args.out / f"{name}_phash_groups.json").write_text(json.dumps({
            "threshold": args.threshold,
            "hash_size": 16,
            "groups": groups,
        }, indent=2))
        print(f"{name}: {len(items)} images -> {len(groups)} groups")


if __name__ == "__main__":
    main()
