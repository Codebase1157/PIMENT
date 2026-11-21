#!/usr/bin/env python3
"""Build leakage-controlled stratified group k-fold assignments (Section 5.2)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from pigmentnet.data.folds import build_folds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/folds"))
    ap.add_argument("--hash-groups", type=Path, default=Path("data/hash_groups"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-share", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for corpus_dir in sorted(args.root.iterdir()):
        name = corpus_dir.name
        idx_path = corpus_dir / "index.json"
        if not idx_path.exists():
            continue
        items = json.loads(idx_path.read_text())
        groups_path = args.hash_groups / f"{name}_phash_groups.json"
        if groups_path.exists():
            raw = json.loads(groups_path.read_text())["groups"]
            path_to_group = {}
            for i, g in enumerate(raw):
                for p in g:
                    path_to_group[p] = i
            groups = [path_to_group.get(it["path"], idx) for idx, it in enumerate(items)]
        else:
            # fall back to one group per image (no leakage control)
            groups = list(range(len(items)))
        folds = build_folds(items, groups, n_splits=args.folds,
                            seed=args.seed, val_share=args.val_share)
        (args.out / f"{name}_folds.json").write_text(json.dumps(folds, indent=2))
        print(f"{name}: {len(items)} items, {max(groups)+1} groups, {args.folds} folds")


if __name__ == "__main__":
    main()
