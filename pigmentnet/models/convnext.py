"""ConvNeXt-T texture stream (Section 4.4.1, Eq. 22).

Uses timm when available, otherwise a self-contained implementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], 1, 1, 1).bernoulli_(keep).div_(keep)
        return x * mask


class ConvNeXtBlock(nn.Module):
    """h_{l+1} = h_l + W2 · GELU(W1 · LN(DWConv7x7(h_l))) (Eq. 22)."""

    def __init__(self, dim: int, drop_path: float = 0.0, ls_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(ls_init * torch.ones(dim)) if ls_init > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma[None, :, None, None]
        return shortcut + self.drop_path(x)


class ConvNeXtTextureStream(nn.Module):
    """Four-stage ConvNeXt-T with stage depths (3, 3, 9, 3) and widths
    (96, 192, 384, 768). Returns stage-wise feature maps for attention."""

    def __init__(self, in_chans: int = 3, depths=(3, 3, 9, 3),
                 dims=(96, 192, 384, 768), drop_path_rate: float = 0.1):
        super().__init__()
        self.dims = dims
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        total = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total)]
        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        cur = 0
        for i, (d, w) in enumerate(zip(depths, dims)):
            if i > 0:
                self.downsamplers.append(nn.Sequential(
                    LayerNorm2d(dims[i - 1]),
                    nn.Conv2d(dims[i - 1], w, kernel_size=2, stride=2),
                ))
            blocks = nn.Sequential(*[ConvNeXtBlock(w, drop_path=dpr[cur + j]) for j in range(d)])
            self.stages.append(blocks)
            cur += d

    def forward(self, x: torch.Tensor):
        feats = []
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsamplers[i - 1](x)
            x = stage(x)
            feats.append(x)
        return feats
