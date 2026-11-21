"""Command-line entry points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from .train import TrainConfig, train
from .utils import SEEDS


def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_cli():
    parser = argparse.ArgumentParser("pigmentnet-train")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--sweep", action="store_true",
                        help="Iterate over all seeds × folds")
    args = parser.parse_args()
    cfg = _load_cfg(args.config)
    print(json.dumps({"config": cfg, "fold": args.fold, "seed": args.seed}, indent=2))
    # Full orchestration (dataloaders, class counts) is expected to be done by
    # the project's training script; this CLI prints the resolved plan.


def eval_cli():
    parser = argparse.ArgumentParser("pigmentnet-eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    cfg = _load_cfg(args.config)
    print(json.dumps({"config": cfg, "split": args.split}, indent=2))
