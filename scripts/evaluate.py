#!/usr/bin/env python3
"""Evaluate one PigmentNet checkpoint on the specified split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml

from pigmentnet.data import SugarcaneLeafDataset
from pigmentnet.evaluate import collect_predictions, score_predictions
from pigmentnet.models import build_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    folds = json.loads(Path(cfg["dataset"]["index"]).read_text())
    paths = folds[str(args.fold)][args.split]
    items = [{"path": p, "cache": p.replace("/images/", "/cache/") + ".npz"} for p in paths]
    ds = SugarcaneLeafDataset(items, physaug=None, train=False)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                        shuffle=False, num_workers=4)

    ck = torch.load(args.checkpoint, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg["model"]["name"], num_classes=cfg["dataset"]["num_classes"]).to(device)
    model.load_state_dict(ck["state_dict"], strict=False)
    preds = collect_predictions(model, loader, device=device, model_name=cfg["model"]["name"])
    print(json.dumps(score_predictions(preds), indent=2))


if __name__ == "__main__":
    main()
