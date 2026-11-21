#!/usr/bin/env python3
"""Cross-dataset (E3) and external (E4) evaluation on the harmonised classes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml

from pigmentnet.data import SugarcaneLeafDataset, HARMONISED_CLASSES
from pigmentnet.data.folds import HARMONISATION
from pigmentnet.evaluate import collect_predictions, score_predictions
from pigmentnet.models import build_model


def _harmonised_subset(index: list[dict], keep: list[str]) -> list[dict]:
    kept = []
    for rec in index:
        if "class" in rec and rec["class"] in HARMONISATION:
            new = HARMONISATION[rec["class"]]
            if new in keep:
                rec["label"] = keep.index(new)
                kept.append(rec)
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-checkpoint", type=Path, required=True)
    ap.add_argument("--target-index", type=Path, required=True)
    ap.add_argument("--target-cache", type=Path, required=True)
    ap.add_argument("--classes", nargs="+", default=HARMONISED_CLASSES)
    ap.add_argument("--model-name", default="pigmentnet")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    index = json.loads(args.target_index.read_text())
    items = _harmonised_subset(index, args.classes)
    for it in items:
        it["cache"] = str(args.target_cache / f"{Path(it['path']).stem}.npz")

    ds = SugarcaneLeafDataset(items, physaug=None, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    ck = torch.load(args.source_checkpoint, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model_name, num_classes=len(args.classes)).to(device)
    model.load_state_dict(ck["state_dict"], strict=False)
    preds = collect_predictions(model, loader, device=device, model_name=args.model_name)
    print(json.dumps(score_predictions(preds, label_names=args.classes), indent=2))


if __name__ == "__main__":
    main()
