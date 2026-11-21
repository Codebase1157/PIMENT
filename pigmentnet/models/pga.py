"""Pigment-guided attention (Section 4.4.3, Eqs. 24–25).

A self-attention layer whose logits receive an additive bias proportional to
the pooled absolute pigment deviation on the token grid:

    A_l = softmax(QK^T / √d + λ_a B),    B_ij = b_j = w_b^T · Pool_s(|ΔĈ|)_j

Every query is pushed toward keys whose tissue departs from the leaf's own
pigment median.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PigmentGuidedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, K: int = 4, bias_strength: float = 0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.w_b = nn.Parameter(torch.ones(K) / K)
        self.lambda_a = nn.Parameter(torch.tensor(float(bias_strength)))
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat: torch.Tensor, delta_C: torch.Tensor) -> torch.Tensor:
        """Parameters
        ----------
        feat : (B, C, H, W) texture feature map
        delta_C : (B, K, H_d, W_d) pigment deviation at full resolution
        """
        B, C, H, W = feat.shape
        N = H * W
        tokens = feat.flatten(2).transpose(1, 2)        # (B, N, C)
        tokens = self.norm(tokens)
        qkv = self.qkv(tokens).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                # (3, B, H, N, Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Pool |ΔĈ| to the token grid
        d_abs = delta_C.abs()
        d_pool = F.adaptive_avg_pool2d(d_abs, (H, W))   # (B, K, H, W)
        b = (d_pool * self.w_b.view(1, -1, 1, 1)).sum(dim=1)   # (B, H, W)
        b = b.flatten(1)                                        # (B, N)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.lambda_a * b.unsqueeze(1).unsqueeze(2)
        attn = attn.softmax(dim=-1)
        out = attn @ v                                  # (B, H, N, Hd)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out).transpose(1, 2).reshape(B, C, H, W)
        return feat + out
