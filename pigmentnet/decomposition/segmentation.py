"""Blade segmentation (Section 4.2.1).

The paper prompts a promptable segmenter at the centre of the frame and
refines its output with a small U-Net trained on 150 corrected masks. Here we
provide both a light classical fallback (Otsu on an ExG-like index, closed by
morphology) and the hook for a learned refiner when weights are available.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects


def _exg(img: np.ndarray) -> np.ndarray:
    """2G - R - B, a classical plant index."""
    img = np.clip(img.astype(np.float32), 0.0, 1.0)
    return 2.0 * img[..., 1] - img[..., 0] - img[..., 2]


def segment_blade(img: np.ndarray, refiner=None, center_prompt: bool = True) -> np.ndarray:
    """Return a binary blade mask Ω.

    Parameters
    ----------
    img : (H, W, 3) linear or sRGB in [0, 1]
    refiner : optional ``torch.nn.Module`` applied to the Otsu seed
    center_prompt : if True keep the connected component covering the image centre
    """
    e = _exg(img)
    try:
        thr = threshold_otsu(e)
    except ValueError:
        thr = 0.0
    mask = e > thr
    mask = binary_opening(mask, disk(3))
    mask = binary_closing(mask, disk(5))
    mask = remove_small_objects(mask, min_size=max(64, int(0.001 * mask.size)))

    if center_prompt:
        labels, n = ndi.label(mask)
        if n > 0:
            H, W = mask.shape
            cy, cx = H // 2, W // 2
            # pick the label closest to the centre weighted by area
            scores = []
            for lab in range(1, n + 1):
                ys, xs = np.where(labels == lab)
                if ys.size == 0:
                    scores.append(0.0)
                    continue
                dist = (ys.mean() - cy) ** 2 + (xs.mean() - cx) ** 2
                scores.append(ys.size / (1.0 + dist / (H * W)))
            best = int(np.argmax(scores)) + 1
            mask = labels == best

    if refiner is not None:
        try:
            import torch
            with torch.no_grad():
                t = torch.from_numpy(img.transpose(2, 0, 1))[None].float()
                seed = torch.from_numpy(mask.astype(np.float32))[None, None]
                prob = refiner(t, seed).squeeze().cpu().numpy()
                mask = prob > 0.5
        except Exception:
            # refiner is optional – classical mask is the fallback
            pass

    return mask.astype(bool)
