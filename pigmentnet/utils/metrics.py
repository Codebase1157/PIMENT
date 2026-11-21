"""Classification metrics (Section 5.5, Eqs. 38–39)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score as _sk_f1,
    confusion_matrix as _sk_cm,
    balanced_accuracy_score as _sk_ba,
    roc_auc_score as _sk_auc,
)


def macro_f1(y_true, y_pred) -> float:
    return float(_sk_f1(y_true, y_pred, average="macro", zero_division=0))


def per_class_f1(y_true, y_pred) -> np.ndarray:
    return _sk_f1(y_true, y_pred, average=None, zero_division=0)


def balanced_accuracy(y_true, y_pred) -> float:
    return float(_sk_ba(y_true, y_pred))


def confusion_matrix(y_true, y_pred, labels=None, normalize: str | None = "true") -> np.ndarray:
    return _sk_cm(y_true, y_pred, labels=labels, normalize=normalize)


def one_vs_rest_auc(y_true, y_scores) -> np.ndarray:
    """Per-class one-vs-rest AUC. ``y_scores`` is (N, C) softmax probabilities."""
    C = y_scores.shape[1]
    aucs = np.zeros(C, dtype=np.float32)
    for k in range(C):
        yk = (y_true == k).astype(np.int64)
        try:
            aucs[k] = _sk_auc(yk, y_scores[:, k])
        except ValueError:
            aucs[k] = float("nan")
    return aucs


def pointing_game(evidence_maps, boxes_per_image) -> float:
    """Share of leaves whose peak evidence falls inside a lesion box.

    ``evidence_maps`` : (N, H, W) float
    ``boxes_per_image`` : list of lists [(y0, x0, y1, x1)] in map coords.
    """
    N = len(evidence_maps)
    hits = 0
    for i in range(N):
        emap = evidence_maps[i]
        boxes = boxes_per_image[i]
        if emap.max() <= 0 or not boxes:
            continue
        y, x = np.unravel_index(int(np.argmax(emap)), emap.shape)
        for y0, x0, y1, x1 in boxes:
            if y0 <= y < y1 and x0 <= x < x1:
                hits += 1
                break
    return hits / max(1, N)


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / union) if union > 0 else 0.0
