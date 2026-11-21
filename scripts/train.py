#!/usr/bin/env python3
"""Train one PigmentNet run.

Orchestrates one (seed, fold) training, reproducing Algorithm 2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader

from pigmentnet.physaug import PhysAug, PhysAugConfig
from pigmentnet.data import SugarcaneLeafDataset
from pigmentnet.losses import LossWeights
from pigmentnet.train import TrainConfig, train


def _load(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if "inherit" in cfg:
        parent = _load(path.parent / cfg["inherit"])
        merged = {**parent, **{k: v for k, v in cfg.items() if k != "inherit"}}
        return merged
    return cfg


def _fold_index(folds_json: dict, fold: int, split: str) -> list[dict]:
    out = []
    for path in folds_json[str(fold)][split]:
        out.append({"path": path, "cache": path.replace("/images/", "/cache/").replace(
            Path(path).suffix, ".npz")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = _load(args.config)

    folds = json.loads(Path(cfg["dataset"]["index"]).read_text())
    train_items = _fold_index(folds, args.fold, "train")
    val_items   = _fold_index(folds, args.fold, "val")

    phys = PhysAug(PhysAugConfig(**cfg.get("physaug", {})))
    train_ds = SugarcaneLeafDataset(train_items, physaug=phys, train=True)
    val_ds   = SugarcaneLeafDataset(val_items, physaug=None, train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    counts = torch.zeros(cfg["dataset"]["num_classes"])
    for rec in train_items:
        counts[rec.get("label", 0)] += 1

    t_cfg = TrainConfig(
        model=cfg["model"]["name"],
        num_classes=cfg["dataset"]["num_classes"],
        epochs=cfg["training"]["epochs"],
        warmup_epochs=cfg["training"]["warmup_epochs"],
        batch_size=cfg["training"]["batch_size"],
        backbone_lr=cfg["training"]["backbone_lr"],
        new_layer_lr=cfg["training"]["new_layer_lr"],
        weight_decay=cfg["training"]["weight_decay"],
        loss_weights=LossWeights(**cfg["loss"]),
        output_dir=f"{cfg['output']['dir']}/seed{args.seed}_fold{args.fold}",
    )
    best = train(t_cfg, train_loader, val_loader, counts, seed=args.seed)
    print(json.dumps({"best": best}))


if __name__ == "__main__":
    main()
