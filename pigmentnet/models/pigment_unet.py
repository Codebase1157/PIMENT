"""Pigment-refiner U-Net (Section 4.4.2).

A light U-Net with encoder widths 32 and 64 and a 32-channel decoder refines
the four pigment-proxy maps into Ĉ, removing the blockiness of the per-pixel
factorisation while staying close to the measured optical density through
the physics term L_phys = ||E Ĉ − OD||_F^2 / |Ω| (Eq. 23).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PigmentRefiner(nn.Module):
    """Light U-Net Ĉ = f(C, Δ, mask).

    Input channels = K (raw pigment maps) + K (Δ) + 1 (mask) = 2K + 1.
    Output channels = K non-negative refined maps.
    """

    def __init__(self, K: int = 4, widths=(32, 64)):
        super().__init__()
        self.K = K
        in_c = 2 * K + 1
        self.enc1 = ConvBlock(in_c, widths[0])
        self.down = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(widths[0], widths[1])
        self.bot  = ConvBlock(widths[1], widths[1])
        self.up   = nn.ConvTranspose2d(widths[1], widths[0], 2, stride=2)
        self.dec1 = ConvBlock(widths[0] + widths[0], widths[0])
        self.head = nn.Conv2d(widths[0], K, 1)

    def forward(self, C: torch.Tensor, delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([C, delta, mask], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        b  = self.bot(e2)
        u  = self.up(b)
        if u.shape[-2:] != e1.shape[-2:]:
            u = F.interpolate(u, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u, e1], dim=1))
        Chat = F.softplus(self.head(d1))      # non-negative pigment maps
        Chat = Chat * mask                    # zero outside Ω
        return Chat
