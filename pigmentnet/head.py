"""Module D: concept-first head with capped texture residual (Eq. 31)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptFirstHead(nn.Module):
    """Logits are a sparse linear map of the first seven decision concepts
    plus a L2-capped texture residual:

        z = A · c_{1:7} + W · r,
        r = κ · h / max(κ, ||h||_2)

    The cap κ matters: without it the residual absorbs the decision and the
    concepts become decoration.
    """

    def __init__(self, num_classes: int, residual_dim: int = 16,
                 num_concepts: int = 7, cap: float = 1.0, l1: float = 1e-3):
        super().__init__()
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.cap = cap
        self.l1 = l1
        self.A = nn.Linear(num_concepts, num_classes, bias=True)
        self.W = nn.Linear(residual_dim, num_classes, bias=False)

    def forward(self, concepts: torch.Tensor, residual: torch.Tensor):
        """Return (full-logits, concept-only-logits, fidelity, |A|_1).

        ``concepts`` has shape (B, num_concepts) and uses c_{1:7}; c_8 (glare)
        is the reliability flag and is not routed through the head.
        """
        norm = residual.norm(dim=1, keepdim=True).clamp_min(1e-8)
        r = self.cap * residual / torch.maximum(torch.full_like(norm, self.cap), norm)
        logits_c = self.A(concepts)
        logits_r = self.W(r)
        full = logits_c + logits_r
        fidelity = self._fidelity(logits_c, logits_r)
        return full, logits_c, fidelity, self.A.weight.abs().sum()

    @staticmethod
    def _fidelity(logits_c: torch.Tensor, logits_r: torch.Tensor) -> torch.Tensor:
        """Eq. 32: share of logit magnitude carried by the concept term."""
        num = logits_c.abs().sum(dim=-1)
        den = num + logits_r.abs().sum(dim=-1) + 1e-8
        return (num / den).mean()
