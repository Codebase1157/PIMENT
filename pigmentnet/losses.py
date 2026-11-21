"""Training objective (Section 4.7, Eq. 35):

    L = L_cls + α L_KL + β_p L_phys + ν ||A||_1 + η ||r||_2^2
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    alpha: float = 0.5
    beta_phys: float = 0.1
    nu: float = 1e-3
    eta: float = 1e-4


class PigmentNetLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None,
                 weights: LossWeights | None = None):
        super().__init__()
        self.weights = weights or LossWeights()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, outputs: dict, targets: torch.Tensor) -> dict:
        w = self.weights
        logits = outputs["logits"]
        concept_logits = outputs["concept_logits"]

        cls = F.cross_entropy(logits, targets, weight=self.class_weights)
        p_full = logits.softmax(dim=-1)
        q_only = concept_logits.softmax(dim=-1).clamp_min(1e-8)
        kl = F.kl_div(q_only.log(), p_full, reduction="batchmean")
        phys = outputs.get("phys_loss", torch.zeros((), device=logits.device))
        a_l1 = outputs.get("A_l1", torch.zeros((), device=logits.device))
        residual = outputs["residual"]
        r_l2 = residual.pow(2).sum(dim=-1).mean()

        total = cls + w.alpha * kl + w.beta_phys * phys + w.nu * a_l1 + w.eta * r_l2
        return {
            "loss": total,
            "cls": cls.detach(),
            "kl": kl.detach(),
            "phys": phys.detach(),
            "A_l1": a_l1.detach(),
            "r_l2": r_l2.detach(),
            "fidelity": outputs["fidelity"].detach(),
        }


def class_balanced_weights(counts: torch.Tensor) -> torch.Tensor:
    """w_y = N_s / (C · n_y) (Eq. 3)."""
    N = counts.sum()
    C = counts.numel()
    return (N / (C * counts.clamp_min(1.0))).float()
