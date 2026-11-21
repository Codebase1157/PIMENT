"""PigmentNet — the full two-stream + concept + head model (Section 4)."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnext import ConvNeXtTextureStream
from .pigment_unet import PigmentRefiner
from .pga import PigmentGuidedAttention
from ..concepts.layer import SymptomConceptLayer
from ..head import ConceptFirstHead


@dataclass
class PigmentNetConfig:
    num_classes: int = 11
    K: int = 4                        # number of pigments
    residual_dim: int = 16
    residual_cap: float = 1.0
    attention_bias: float = 0.5
    drop_path: float = 0.1
    convnext_dims: tuple = (96, 192, 384, 768)
    convnext_depths: tuple = (3, 3, 9, 3)
    num_concepts: int = 7


class PigmentNet(nn.Module):
    def __init__(self, cfg: PigmentNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or PigmentNetConfig()
        c = self.cfg

        # Texture stream reads the corrected diffuse image
        self.texture = ConvNeXtTextureStream(
            in_chans=3, depths=c.convnext_depths, dims=c.convnext_dims,
            drop_path_rate=c.drop_path,
        )
        # Pigment refiner U-Net
        self.pigment = PigmentRefiner(K=c.K, widths=(32, 64))
        # Pigment-guided attention after stage 3 and stage 4
        self.pga3 = PigmentGuidedAttention(c.convnext_dims[2], K=c.K,
                                           bias_strength=c.attention_bias)
        self.pga4 = PigmentGuidedAttention(c.convnext_dims[3], K=c.K,
                                           bias_strength=c.attention_bias)
        # 16-D texture residual h
        self.residual_proj = nn.Linear(c.convnext_dims[-1], c.residual_dim)
        # Concept layer and head
        self.concepts = SymptomConceptLayer(K=c.K)
        self.head = ConceptFirstHead(
            num_classes=c.num_classes, residual_dim=c.residual_dim,
            num_concepts=c.num_concepts, cap=c.residual_cap,
        )

    # -----------------------------------------------------------------
    def forward(self, diffuse: torch.Tensor, C_init: torch.Tensor,
                delta_init: torch.Tensor, mask: torch.Tensor,
                glare_mask: torch.Tensor, od: torch.Tensor | None = None,
                E: torch.Tensor | None = None):
        """Return a dict with every output of Module D plus auxiliaries.

        Parameters
        ----------
        diffuse : (B, 3, H, W) illuminant-corrected sRGB (encoded or linear)
        C_init : (B, K, H, W) raw pigment maps from the factorisation
        delta_init : (B, K, H, W) leaf-relative deviation
        mask : (B, 1, H, W) blade Ω
        glare_mask : (B, 1, H, W) G
        od : optional (B, 3, H, W) optical density for L_phys
        E : optional (B, 3, K) absorbance matrix for L_phys
        """
        Chat = self.pigment(C_init, delta_init, mask)
        delta_hat = self._deviation(Chat, mask)

        feats = self.texture(diffuse)
        feats3 = self.pga3(feats[2], delta_hat)
        feats_down = self._downsample(feats3, target=feats[3])
        feats4 = self.pga4(feats_down, delta_hat)

        pooled = F.adaptive_avg_pool2d(feats4, 1).flatten(1)
        h = self.residual_proj(pooled)

        concept_vec = self.concepts(Chat, mask, glare_mask)
        c_decision = concept_vec[..., :self.cfg.num_concepts]
        full_logits, concept_logits, fidelity, l1 = self.head(c_decision, h)

        out = {
            "logits": full_logits,
            "concept_logits": concept_logits,
            "concepts": concept_vec,
            "fidelity": fidelity,
            "A_l1": l1,
            "Chat": Chat,
            "delta_Chat": delta_hat,
            "mask": mask,
            "residual": h,
        }
        if od is not None and E is not None:
            out["phys_loss"] = self._phys_loss(Chat, mask, od, E)
        return out

    # -----------------------------------------------------------------
    @staticmethod
    def _deviation(Chat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, K, H, W = Chat.shape
        out = torch.zeros_like(Chat)
        for b in range(B):
            m = mask[b, 0] > 0.5
            if m.sum() == 0:
                continue
            for k in range(K):
                px = Chat[b, k][m]
                med = px.median()
                out[b, k][m] = Chat[b, k][m] - med
        return out

    @staticmethod
    def _downsample(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != target.shape[-2:]:
            x = F.adaptive_avg_pool2d(x, target.shape[-2:])
        return x

    @staticmethod
    def _phys_loss(Chat: torch.Tensor, mask: torch.Tensor,
                   od: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """L_phys = ||E Ĉ − OD||_F^2 / |Ω| (Eq. 23).

        E : (B, 3, K), Chat : (B, K, H, W), OD : (B, 3, H, W).
        """
        B, K, H, W = Chat.shape
        c_flat = Chat.view(B, K, -1)
        pred = torch.bmm(E, c_flat).view(B, 3, H, W)
        diff = (pred - od) * mask
        area = mask.sum(dim=(-2, -1, -3)).clamp_min(1.0)
        return (diff.pow(2).sum(dim=(-3, -2, -1)) / area).mean()
