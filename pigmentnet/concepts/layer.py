"""Symptom-concept layer (Section 4.5, Table 2)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .area import area_concepts
from .texture import heterogeneity, patchiness_moran, lesion_density, stripe_coherence


class SymptomConceptLayer(nn.Module):
    """Deterministic, differentiable layer that produces the 8-dim concept
    vector c from refined pigment maps Ĉ, pigment composition p, blade mask
    Ω and the glare mask G.

        c = (c1 chlorotic, c2 red, c3 necrotic, c4 heterogeneity,
             c5 patchiness, c6 lesion density, c7 stripe coherence,
             c8 glare)

    Only the thresholds and temperature are learnable; the formulas are
    fixed. c_8 is a reliability flag and is excluded from the decision.
    """

    def __init__(self, K: int = 4,
                 init_t_ylw: float = 0.35,
                 init_t_red: float = 0.30,
                 init_t_brn: float = 0.25,
                 init_t_log: float = 0.04,
                 init_T: float = 0.08,
                 lesion_sigma: float = 2.5):
        super().__init__()
        self.K = K
        self.lesion_sigma = lesion_sigma
        # thresholds for c1-c3 and c6
        self.t_ylw = nn.Parameter(torch.tensor(init_t_ylw))
        self.t_red = nn.Parameter(torch.tensor(init_t_red))
        self.t_brn = nn.Parameter(torch.tensor(init_t_brn))
        self.t_log = nn.Parameter(torch.tensor(init_t_log))
        self.T     = nn.Parameter(torch.tensor(init_T))

    def forward(self, Chat: torch.Tensor, mask: torch.Tensor,
                glare_mask: torch.Tensor) -> torch.Tensor:
        """Parameters
        ----------
        Chat : (B, K, H, W) refined pigment maps
        mask : (B, 1, H, W) blade mask Ω
        glare_mask : (B, 1, H, W) glare pixel indicator G

        Returns
        -------
        c : (B, 8) concept vector
        """
        eps = 1e-6
        p = Chat / (Chat.sum(dim=1, keepdim=True) + eps)
        p_chl = p[:, 0:1]
        p_brn = p[:, 3:4]

        c1, c2, c3 = area_concepts(p, mask,
                                   torch.stack([self.t_ylw, self.t_red, self.t_brn]), self.T)
        c4 = heterogeneity(p_chl, mask)
        c5 = patchiness_moran(p_chl, mask)
        c6 = lesion_density(p_brn, mask, self.t_log, self.T, sigma=self.lesion_sigma)
        c7 = stripe_coherence(p_chl, mask)
        denom = mask.sum(dim=(-2, -1, -3)).clamp_min(1.0)
        c8 = (glare_mask * mask).sum(dim=(-2, -1, -3)) / denom

        out = torch.stack([c1, c2, c3, c4, c5, c6, c7, c8], dim=-1)
        return out
