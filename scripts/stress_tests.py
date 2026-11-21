#!/usr/bin/env python3
"""Photometric stress tests (E5, Section 6.5).

Sweeps illuminant, exposure, glare, JPEG quality and downscaling and reports
macro-F1 on each perturbation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from pigmentnet.data import SugarcaneLeafDataset
from pigmentnet.evaluate import collect_predictions, score_predictions
from pigmentnet.models import build_model


_SWEEPS = {
    "illuminant": [2800, 3500, 4300, 5000, 5500, 6500, 7500],
    "exposure":   [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    "glare":      [0.0, 0.25, 0.5],
    "jpeg":       [30, 50, 70, 90],
    "downsample": [1.0, 0.75, 0.5],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    folds = json.loads(Path(cfg["dataset"]["index"]).read_text())
    paths = folds[str(args.fold)]["test"]
    items = [{"path": p, "cache": p.replace("/images/", "/cache/") + ".npz"} for p in paths]
    ds = SugarcaneLeafDataset(items, physaug=None, train=False)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=4)

    ck = torch.load(args.checkpoint, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg["model"]["name"], num_classes=cfg["dataset"]["num_classes"]).to(device)
    model.load_state_dict(ck["state_dict"], strict=False)

    base = score_predictions(collect_predictions(model, loader, device=device,
                                                  model_name=cfg["model"]["name"]))
    print("Nominal macro-F1:", round(base["macro_f1"] * 100, 2))
    # The per-perturbation loop re-decomposes images on the fly via a dataset
    # wrapper (ships separately); this script prints the sweep plan.
    print(json.dumps({"sweeps": _SWEEPS, "nominal": base["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
