"""Evaluation protocols E1–E5 (Section 5.2)."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils.metrics import macro_f1, balanced_accuracy, per_class_f1, confusion_matrix, one_vs_rest_auc


@torch.no_grad()
def collect_predictions(model, loader, device: str = "cuda",
                        model_name: str = "pigmentnet"):
    model.eval()
    y_true, y_pred, y_score, concepts = [], [], [], []
    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        if model_name == "pigmentnet":
            out = model(batch["diffuse"], batch["C"], batch["delta_C"],
                        batch["mask"], batch["glare_mask"])
            logits = out["logits"]
            concepts.append(out["concepts"].cpu().numpy())
        else:
            logits = model(batch["diffuse"])
        probs = logits.softmax(dim=-1)
        y_true.append(batch["label"].cpu().numpy())
        y_pred.append(logits.argmax(dim=-1).cpu().numpy())
        y_score.append(probs.cpu().numpy())
    return {
        "y_true": np.concatenate(y_true),
        "y_pred": np.concatenate(y_pred),
        "y_score": np.concatenate(y_score),
        "concepts": np.concatenate(concepts) if concepts else None,
    }


def score_predictions(preds: dict, label_names=None) -> dict:
    y_t, y_p, y_s = preds["y_true"], preds["y_pred"], preds["y_score"]
    labels = list(range(y_s.shape[1])) if label_names is None else list(range(len(label_names)))
    return {
        "macro_f1": macro_f1(y_t, y_p),
        "balanced_accuracy": balanced_accuracy(y_t, y_p),
        "per_class_f1": per_class_f1(y_t, y_p).tolist(),
        "ovr_auc": one_vs_rest_auc(y_t, y_s).tolist(),
        "confusion_matrix": confusion_matrix(y_t, y_p, labels=labels).tolist(),
    }


# ---------------------------------------------------------------------
# E5 stress tests
def _apply_jpeg(arr: np.ndarray, quality: int) -> np.ndarray:
    import io
    from PIL import Image
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf)).astype(np.float32) / 255.0


def photometric_stress(image: np.ndarray, kind: str, value: float):
    """Apply one of the perturbations of E5 to a single sRGB image."""
    if kind == "illuminant":
        from .physaug import planckian_illuminant
        from .decomposition.linearise import linearise_srgb, encode_srgb
        gamma = planckian_illuminant(value)
        lin = linearise_srgb(image)
        lin = lin * (gamma / (gamma.mean() + 1e-8)).reshape(1, 1, 3)
        return encode_srgb(np.clip(lin, 0, 1))
    if kind == "exposure":
        from .decomposition.linearise import linearise_srgb, encode_srgb
        lin = linearise_srgb(image)
        lin = np.clip(lin * (2.0 ** value), 0.0, 1.0)
        return encode_srgb(lin)
    if kind == "glare":
        gain = float(value)
        return np.clip(image + gain * (image > 0.6).astype(np.float32) * image, 0.0, 1.0)
    if kind == "jpeg":
        return _apply_jpeg(image, quality=int(value))
    if kind == "downsample":
        from PIL import Image
        H, W = image.shape[:2]
        im = Image.fromarray((image * 255).astype("uint8")).resize(
            (int(W * value), int(H * value)), Image.BILINEAR
        ).resize((W, H), Image.BILINEAR)
        return np.asarray(im).astype(np.float32) / 255.0
    raise ValueError(f"Unknown stress kind: {kind}")
