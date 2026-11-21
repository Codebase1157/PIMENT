"""Training loop (Algorithm 2)."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import PigmentNetLoss, LossWeights, class_balanced_weights
from .models import build_model
from .utils import seed_everything


@dataclass
class TrainConfig:
    model: str = "pigmentnet"
    num_classes: int = 11
    epochs: int = 100
    warmup_epochs: int = 5
    batch_size: int = 32
    backbone_lr: float = 1e-4
    new_layer_lr: float = 5e-4
    weight_decay: float = 0.05
    loss_weights: LossWeights = field(default_factory=LossWeights)
    mixed_precision: bool = True
    grad_clip: float = 1.0
    log_every: int = 50
    output_dir: str = "runs/default"


def _param_groups(model: nn.Module, backbone_lr: float, new_lr: float):
    bbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bbone if "texture" in name else head).append(p)
    return [
        {"params": bbone, "lr": backbone_lr, "name": "backbone"},
        {"params": head,  "lr": new_lr,      "name": "new"},
    ]


def _cosine_warmup(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    t = (epoch - warmup) / max(1, (total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * t))


def train(cfg: TrainConfig, train_loader: DataLoader, val_loader: DataLoader,
          class_counts: torch.Tensor, seed: int = 42):
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg.model, num_classes=cfg.num_classes).to(device)
    optimizer = torch.optim.AdamW(_param_groups(model, cfg.backbone_lr, cfg.new_layer_lr),
                                  weight_decay=cfg.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    class_weights = class_balanced_weights(class_counts).to(device)
    loss_fn = PigmentNetLoss(class_weights=class_weights, weights=cfg.loss_weights)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision and device == "cuda")

    best = {"val_f1": 0.0, "epoch": -1}
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        mul = _cosine_warmup(epoch, cfg.warmup_epochs, cfg.epochs)
        for g, lr0 in zip(optimizer.param_groups, base_lrs):
            g["lr"] = lr0 * mul

        model.train()
        for step, batch in enumerate(train_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision and device == "cuda"):
                if cfg.model == "pigmentnet":
                    out = model(batch["diffuse"], batch["C"], batch["delta_C"],
                                batch["mask"], batch["glare_mask"],
                                od=batch["od"], E=batch["E"])
                else:
                    logits = model(batch["diffuse"])
                    out = {"logits": logits, "concept_logits": logits,
                           "fidelity": torch.tensor(0.0, device=device),
                           "residual": torch.zeros(logits.size(0), 1, device=device)}
                losses = loss_fn(out, batch["label"])
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        val_f1 = _evaluate(model, val_loader, device, cfg)
        if val_f1 > best["val_f1"]:
            best = {"val_f1": val_f1, "epoch": epoch}
            torch.save({
                "state_dict": model.state_dict(),
                "cfg": cfg.__dict__,
                "seed": seed,
                "epoch": epoch,
                "val_f1": val_f1,
            }, Path(cfg.output_dir) / "best.pt")

    return best


@torch.no_grad()
def _evaluate(model, loader, device, cfg) -> float:
    from .utils.metrics import macro_f1
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        if cfg.model == "pigmentnet":
            out = model(batch["diffuse"], batch["C"], batch["delta_C"],
                        batch["mask"], batch["glare_mask"])
            logits = out["logits"]
        else:
            logits = model(batch["diffuse"])
        y_true.append(batch["label"].cpu().numpy())
        y_pred.append(logits.argmax(dim=-1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return macro_f1(y_true, y_pred)
