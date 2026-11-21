"""Model registry for PigmentNet and the eleven comparison methods."""
from __future__ import annotations

import torch
import torch.nn as nn

from .pigmentnet import PigmentNet, PigmentNetConfig
from .convnext import ConvNeXtTextureStream


def build_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    """Build one of the models listed in Section 5.4.

    Names
    -----
    ``pigmentnet`` – the proposed model.
    ``convnext_t``, ``swin_t``, ``deit_s``, ``mobilevit_v2``, ``resnet50``,
    ``efficientnetv2_s``, ``vmamba_t`` – backbones. Backbones other than
    convnext_t require ``timm`` (optional dependency).
    """
    name = name.lower()
    if name == "pigmentnet":
        cfg = PigmentNetConfig(num_classes=num_classes, **kwargs)
        return PigmentNet(cfg)

    try:
        import timm
    except ImportError as exc:
        if name == "convnext_t":
            # fall through to self-contained backbone
            pass
        else:
            raise RuntimeError(
                f"Comparison model '{name}' needs the optional dependency `timm`."
            ) from exc

    if name == "convnext_t":
        try:
            import timm
            m = timm.create_model("convnext_tiny.fb_in1k", pretrained=False,
                                  num_classes=num_classes)
            return m
        except Exception:
            return _SelfContainedConvNeXt(num_classes=num_classes)
    if name == "swin_t":
        return timm.create_model("swin_tiny_patch4_window7_224", pretrained=False,
                                 num_classes=num_classes, img_size=384)
    if name == "deit_s":
        return timm.create_model("deit_small_patch16_224", pretrained=False,
                                 num_classes=num_classes, img_size=384)
    if name == "mobilevit_v2":
        return timm.create_model("mobilevitv2_100", pretrained=False,
                                 num_classes=num_classes)
    if name == "resnet50":
        return timm.create_model("resnet50", pretrained=False, num_classes=num_classes)
    if name == "efficientnetv2_s":
        return timm.create_model("tf_efficientnetv2_s", pretrained=False,
                                 num_classes=num_classes)
    if name == "vmamba_t":
        try:
            return timm.create_model("vmamba_tiny", pretrained=False,
                                     num_classes=num_classes)
        except Exception as exc:
            raise RuntimeError(
                "VMamba-T weights are not in timm by default. Install the "
                "`vmamba` package from the author release and re-run."
            ) from exc
    raise ValueError(f"Unknown model name: {name}")


class _SelfContainedConvNeXt(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = ConvNeXtTextureStream()
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        pooled = feats[-1].mean(dim=(-2, -1))
        return self.head(pooled)
