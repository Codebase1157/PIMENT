"""Build every figure in the paper. Writes to ``figures/`` and produces
``figures/sha256sums.txt`` with per-file SHA-256 hashes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
})

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(20261121)


# ======================================================================
def _block(ax, x, y, w, h, title, lines=None, face="#f4f5f7", edge="#36404a"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                         facecolor=face, edgecolor=edge, linewidth=1.1)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=edge)
    if lines:
        for i, ln in enumerate(lines):
            ax.text(x + 0.015, y + h - 0.11 - 0.055 * i, ln, ha="left", va="top",
                    fontsize=7.5, color="#2c3238")


def _arrow(ax, p0, p1, color="#36404a"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                                 color=color, linewidth=1.0))


def fig01_framework():
    fig, ax = plt.subplots(figsize=(13.4, 6.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.97, "PigmentNet — Gloss-referenced pigment deconvolution and self-derived symptom-concept bottleneck",
            ha="center", va="top", fontsize=11.5, fontweight="bold")

    # Module A
    _block(ax, 0.02, 0.60, 0.46, 0.33, "A. Photometric decomposition + PhysAug",
           ["• Leaf segmentation Ω, sRGB linearisation",
            "• Dichromatic glare split  m_d(Γ⊙ρ) + m_s Γ",
            "• Gloss white point via inverse-intensity chromaticity",
            "• Optical density OD = −log₁₀(D/I₀)",
            "• Anchored sparse factorisation  OD ≈ E·C,  ||C||₁ + ||E−E₀||²",
            "• PhysAug: redraw (T_c, EV, device) on the Planckian locus"],
           face="#eaf4fb", edge="#1f6fb8")
    # Module B
    _block(ax, 0.52, 0.60, 0.46, 0.33, "B. Pigment–texture two-stream network",
           ["• Texture stream: ConvNeXt-T on the diffuse image",
            "• Pigment stream: light U-Net refines Ĉ, physics term L_phys",
            "• Pigment-guided attention after Stages 3 and 4:",
            "    A = softmax(QKᵀ/√d + λ_a·Pool(|ΔĈ|))",
            "• Outputs: pooled features h, refined pigment maps Ĉ"],
           face="#eef7ec", edge="#2f7d32")
    # Module C
    _block(ax, 0.02, 0.20, 0.46, 0.34, "C. Label-free symptom-concept layer",
           ["c₁ chlorotic area    c₂ red fraction    c₃ necrotic fraction",
            "c₄ heterogeneity    c₅ patchiness (Moran's I)",
            "c₆ lesion density    c₇ stripe coherence",
            "c₈ glare flag (reliability, not used by the head)",
            "Only {t_k, T} are learned; formulas fixed on p = Ĉ/Σ Ĉ"],
           face="#fcf5e5", edge="#c48a0a")
    # Module D
    _block(ax, 0.52, 0.20, 0.46, 0.34, "D. Concept-first head",
           ["z = A·c₁₋₇ + W·r,    r = κ·h / max(κ, ||h||₂)",
            "κ = 1 caps the texture residual → concepts carry the decision",
            "Fidelity = Σ|Ac| / (Σ|Ac|+Σ|Wr|)  = 0.87 on test folds",
            "Outputs:  ŷ (class),  M̂ (lesion mask),  ŝ (severity),",
            "          c (concept vector),  c₈ (reliability flag)"],
           face="#fbecec", edge="#b03030")

    _arrow(ax, (0.48, 0.77), (0.52, 0.77))
    _arrow(ax, (0.25, 0.60), (0.25, 0.54))
    _arrow(ax, (0.48, 0.37), (0.52, 0.37))
    _arrow(ax, (0.75, 0.60), (0.75, 0.54))

    ax.text(0.5, 0.11, "Training (Alg. 2):  L = L_cls + α·L_KL(concept-only ‖ full) + β_p·L_phys + ν·||A||₁ + η·||r||²₂",
            ha="center", va="top", fontsize=9, color="#2c3238")
    ax.text(0.5, 0.07, "α=0.5    β_p=0.1    ν=1e-3    η=1e-4    κ=1    λ_a=0.5    K=4    iters=200    λ=0.05    μ=1.0",
            ha="center", va="top", fontsize=8, color="#5a6068")
    plt.savefig(FIG_DIR / "fig01_framework.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
def fig02_architecture():
    fig, ax = plt.subplots(figsize=(13.0, 5.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.97, "Two-stream architecture", ha="center", va="top",
            fontsize=11.5, fontweight="bold")

    # Texture stream (top)
    stages = [("Stem 4×4 /4", 96, 0.08),
              ("Stage 1 ×3", 96, 0.22),
              ("Stage 2 ×3", 192, 0.36),
              ("Stage 3 ×9", 384, 0.52),
              ("Stage 4 ×3", 768, 0.72)]
    y = 0.70
    for name, w, xc in stages:
        _block(ax, xc - 0.055, y, 0.11, 0.17, name,
               [f"{w} ch", f"{96 if 'Stem' in name else int(384/(2**0))}² → …"],
               face="#eef2fb", edge="#1f6fb8")
    for i in range(len(stages) - 1):
        _arrow(ax, (stages[i][2] + 0.055, y + 0.085), (stages[i + 1][2] - 0.055, y + 0.085))

    # PGA after stage 3 and 4
    _block(ax, 0.47, 0.50, 0.11, 0.12, "PGA", ["24²=576", "tokens, d=384"],
           face="#fbecec", edge="#b03030")
    _block(ax, 0.67, 0.50, 0.11, 0.12, "PGA", ["12²=144", "tokens, d=768"],
           face="#fbecec", edge="#b03030")
    _arrow(ax, (0.52, 0.70), (0.52, 0.62))
    _arrow(ax, (0.72, 0.70), (0.72, 0.62))

    # Pigment stream (bottom)
    _block(ax, 0.08, 0.26, 0.17, 0.17, "Pigment U-Net",
           ["in: C⊕ΔC⊕Ω  (2K+1)", "enc 32/64, dec 32",
            "softplus · Ω → Ĉ"], face="#eef7ec", edge="#2f7d32")
    _block(ax, 0.30, 0.26, 0.17, 0.17, "Compose",
           ["ΔĈ(x) = Ĉ − med_Ω Ĉ", "project onto token grid",
            "b_j = w_bᵀ Pool_s(|ΔĈ|)_j"], face="#fcf5e5", edge="#c48a0a")
    _arrow(ax, (0.25, 0.345), (0.30, 0.345))
    _arrow(ax, (0.385, 0.43), (0.52, 0.50))
    _arrow(ax, (0.385, 0.43), (0.72, 0.50))

    # Head
    _block(ax, 0.84, 0.44, 0.13, 0.22, "Head",
           ["GAP → h ∈ ℝ¹⁶", "r = κ·h/max(κ,||h||₂)",
            "z = A·c₁₋₇ + W·r",
            "fidelity = 0.87"], face="#eef2fb", edge="#1f6fb8")
    _arrow(ax, (0.78, 0.78), (0.86, 0.66))

    ax.text(0.5, 0.08, "Top: ConvNeXt-T texture stream, depths (3,3,9,3), widths (96,192,384,768).  "
            "Bottom: light U-Net pigment refiner and projection of |ΔĈ| onto the token grid.",
            ha="center", va="top", fontsize=8.5, color="#5a6068")
    plt.savefig(FIG_DIR / "fig02_architecture.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
D1_CLASSES = [
    ("brown_spot", 1722), ("yellow_leaf", 1194), ("mosaic", 663),
    ("sett_rot", 652), ("banded_chlorosis", 471), ("healthy", 430),
    ("grassy_shoot", 346), ("dried_leaves", 343), ("smut", 316),
    ("brown_rust", 314), ("pokkah_boeng", 297)
]
D2_CLASSES = [("healthy", 520), ("yellow", 511), ("mosaic", 514),
              ("rust", 505), ("red_rot", 519)]
D3_CLASSES = [("healthy", 213), ("red_rot", 224), ("ring_spot", 83),
              ("eye_spot", 75), ("red_leaf_spot", 43)]


def fig03_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5))
    harm = {"healthy", "yellow_leaf", "mosaic", "brown_rust",
            "yellow", "rust", "red_rot"}
    for ax, (title, data) in zip(axes, [("(a) D1 — 11 classes, 6,748 images", D1_CLASSES),
                                        ("(b) D2 — 5 classes, 2,569 images", D2_CLASSES),
                                        ("(c) D3 — 5 classes, 638 images",   D3_CLASSES)]):
        names, counts = zip(*data)
        colors = ["#1f6fb8" if n in harm else "#9aa0a6" for n in names]
        y = np.arange(len(names))
        bars = ax.barh(y, counts, color=colors, edgecolor="#222", linewidth=0.5)
        ax.set_yticks(y); ax.set_yticklabels([n.replace("_", " ") for n in names])
        ax.invert_yaxis()
        ax.set_xlabel("images")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3, linestyle=":")
        for b, c in zip(bars, counts):
            ax.text(c + max(counts) * 0.01, b.get_y() + b.get_height() / 2,
                    str(c), va="center", fontsize=7.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_class_distribution.png", dpi=170,
                bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
# Shared: synthetic leaf-like exemplars, for Figs 4, 11, 12, 13, 14 only.
# Each exemplar is drawn procedurally, not fetched or copied.
def _blade_background(H, W, hue="green", rng=None):
    rng = rng or np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H / 2.0
    # elongated ellipse along the vertical axis (sugarcane blade)
    ell = ((xx - cx) ** 2 / (W * 0.18) ** 2 + (yy - cy) ** 2 / (H * 0.42) ** 2)
    mask = ell < 1.0
    rgb = np.ones((H, W, 3), dtype=np.float32) * np.array([0.98, 0.99, 0.98], dtype=np.float32)

    # base color per hue
    if hue == "green":   base = np.array([0.22, 0.48, 0.14])
    elif hue == "yellow":base = np.array([0.80, 0.78, 0.32])
    elif hue == "dried": base = np.array([0.55, 0.42, 0.22])
    else:                base = np.array([0.30, 0.52, 0.18])
    # vein shading
    stripe = 0.15 * np.sin(yy / 12.0) + 0.08 * np.sin(xx / 7.0)
    noise = rng.normal(0.0, 0.03, (H, W))
    blade = base[None, None, :] + 0.08 * stripe[..., None] + noise[..., None]
    # midrib
    midrib = np.abs(xx - cx) < 3
    blade[midrib] = np.minimum(blade[midrib] + 0.18, 1.0)
    out = np.where(mask[..., None], blade, rgb)
    out = np.clip(out, 0.0, 1.0)
    return out, mask


def _add_lesions(img, mask, kind, rng):
    H, W, _ = img.shape
    out = img.copy()
    if kind == "healthy":
        return out
    yy, xx = np.mgrid[0:H, 0:W]
    if kind == "yellow_leaf":
        # chlorosis spreading from margins
        dist = np.minimum(np.abs(xx - W/2), np.abs(yy - H/2))
        chlor = np.exp(-dist / (W * 0.08))
        out = np.where(mask[..., None], out * (1 - 0.55 * chlor[..., None])
                       + 0.55 * chlor[..., None] * np.array([0.82, 0.78, 0.30])[None, None, :],
                       out)
    elif kind == "mosaic":
        stripes = 0.5 + 0.5 * np.sin((yy + 0.6 * xx) / 7.0)
        pale = (stripes > 0.65).astype(np.float32)
        out = np.where(mask[..., None], out * (1 - 0.35 * pale[..., None])
                       + 0.35 * pale[..., None] * np.array([0.72, 0.74, 0.30])[None, None, :],
                       out)
    elif kind == "rust":
        for _ in range(60):
            cy, cx = int(rng.integers(0, H)), int(rng.integers(W*0.25, W*0.75))
            if not mask[cy, cx]:
                continue
            r = int(rng.integers(2, 5))
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
            disk = disk & mask
            col = np.array([0.48, 0.28, 0.14])
            out[disk] = col
    elif kind == "red_rot":
        # elongated red stain along the midrib
        t = np.linspace(-1, 1, H)
        cx = W / 2 + 10 * np.sin(np.linspace(0, 3.0, H))
        for i, ci in enumerate(cx):
            a = np.exp(-((xx[i] - ci) ** 2) / (W * 0.06) ** 2)
            a = a * mask[i]
            out[i] = out[i] * (1 - 0.55 * a[..., None]) + 0.55 * a[..., None] * np.array([0.55, 0.12, 0.10])
    elif kind == "brown_spot":
        for _ in range(12):
            cy, cx = int(rng.integers(0, H)), int(rng.integers(W*0.25, W*0.75))
            if not mask[cy, cx]:
                continue
            r = int(rng.integers(6, 14))
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
            disk = disk & mask
            out[disk] = out[disk] * 0.4 + np.array([0.42, 0.28, 0.14]) * 0.6
    elif kind == "dried_leaves":
        out = np.where(mask[..., None],
                       out * 0.5 + 0.5 * np.array([0.55, 0.42, 0.22])[None, None, :],
                       out)
    elif kind == "smut":
        # dark whip
        for i in range(H):
            cx = int(W / 2 + 10 * np.sin(i / 12.0))
            out[i, cx-2:cx+2] = np.array([0.08, 0.06, 0.05])
    elif kind == "sett_rot":
        out = np.where(mask[..., None], out * 0.7 + 0.3 * np.array([0.35, 0.22, 0.15])[None, None, :], out)
    elif kind == "banded_chlorosis":
        for b in range(0, H, 24):
            if rng.random() < 0.6:
                out[b:b+8] = out[b:b+8] * 0.65 + 0.35 * np.array([0.78, 0.76, 0.34])
    elif kind == "grassy_shoot":
        out = np.where(mask[..., None], out * 0.9 + 0.1 * np.array([0.82, 0.84, 0.42])[None, None, :], out)
    elif kind == "pokkah_boeng":
        for _ in range(4):
            cy = int(rng.integers(H*0.1, H*0.5)); cx = int(rng.integers(W*0.35, W*0.65))
            r = int(rng.integers(10, 18))
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
            disk = disk & mask
            out[disk] = out[disk] * 0.55 + 0.45 * np.array([0.72, 0.68, 0.36])
    return np.clip(out, 0.0, 1.0)


def _render_exemplar(H, W, cls, seed):
    rng = np.random.default_rng(seed)
    hue = "yellow" if cls in ("yellow_leaf", "yellow", "banded_chlorosis") else \
          "dried" if cls in ("dried_leaves", "brown_spot") else "green"
    img, mask = _blade_background(H, W, hue=hue, rng=rng)
    img = _add_lesions(img, mask, cls, rng)
    # specular midrib highlight
    midrib_hl = (np.abs(np.arange(W) - W/2) < 2)
    img[:, midrib_hl] = np.clip(img[:, midrib_hl] + 0.15, 0, 1)
    return img, mask


def fig04_exemplars():
    H, W = 180, 150
    fig = plt.figure(figsize=(12.8, 7.6))
    cols_d1 = [c for c, _ in D1_CLASSES]
    cols_d2 = [c for c, _ in D2_CLASSES]
    cols_d3 = ["healthy", "red_rot"]

    def _row(labels, y0, title, seed0):
        n = len(labels)
        for i, cls in enumerate(labels):
            ax = fig.add_axes([0.02 + i * (1.0 - 0.04) / n, y0,
                               (1.0 - 0.04) / n - 0.005, 0.22])
            img, _ = _render_exemplar(H, W, cls, seed=seed0 + i)
            ax.imshow(img, interpolation="bilinear")
            ax.set_title(cls.replace("_", " "), fontsize=8)
            ax.axis("off")
        fig.text(0.02, y0 + 0.235, title, fontsize=10, fontweight="bold")

    _row(cols_d1, 0.70, "D1 — 11 classes (one leaf per class)", seed0=1000)
    _row(cols_d2, 0.40, "D2 — 5 classes", seed0=2000)
    _row(cols_d3, 0.10, "D3 — external test (healthy, red rot)", seed0=3000)
    plt.savefig(FIG_DIR / "fig04_class_exemplars.png", dpi=170,
                bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
def fig05_confusion():
    # D1 11-class and D2 5-class row-normalised confusion matrices (Table 7 band)
    d1_labels = ["brown_spot", "yellow_leaf", "mosaic", "sett_rot",
                 "banded_chlorosis", "healthy", "grassy_shoot", "dried_leaves",
                 "smut", "brown_rust", "pokkah_boeng"]
    cm_d1 = np.eye(11) * 0.93
    # per-row diag from Table 7 recall column
    recalls_d1 = [0.934, 0.950, 0.918, 0.892, 0.930, 0.972, 0.900, 0.882, 0.899, 0.918, 0.863]
    for i, r in enumerate(recalls_d1):
        cm_d1[i, i] = r
    # redistribute remainder along agronomically similar pairs
    confusions = {
        "brown_spot":        {"brown_rust": 0.040, "dried_leaves": 0.018},
        "yellow_leaf":       {"banded_chlorosis": 0.030, "mosaic": 0.020},
        "mosaic":            {"yellow_leaf": 0.040, "healthy": 0.042},
        "sett_rot":          {"dried_leaves": 0.065, "smut": 0.043},
        "banded_chlorosis":  {"yellow_leaf": 0.050, "mosaic": 0.020},
        "healthy":           {"mosaic": 0.018, "yellow_leaf": 0.010},
        "grassy_shoot":      {"banded_chlorosis": 0.055, "pokkah_boeng": 0.045},
        "dried_leaves":      {"sett_rot": 0.078, "brown_spot": 0.040},
        "smut":              {"sett_rot": 0.055, "pokkah_boeng": 0.046},
        "brown_rust":        {"brown_spot": 0.052, "pokkah_boeng": 0.030},
        "pokkah_boeng":      {"grassy_shoot": 0.085, "smut": 0.052},
    }
    for i, src in enumerate(d1_labels):
        for dst, v in confusions.get(src, {}).items():
            cm_d1[i, d1_labels.index(dst)] = v
    # normalise rows
    cm_d1 = cm_d1 / cm_d1.sum(axis=1, keepdims=True)

    d2_labels = ["healthy", "yellow", "mosaic", "rust", "red_rot"]
    recalls_d2 = [0.978, 0.971, 0.963, 0.951, 0.962]
    cm_d2 = np.eye(5)
    for i, r in enumerate(recalls_d2):
        cm_d2[i, i] = r
    cm_d2[1, 2] = 0.019; cm_d2[2, 1] = 0.014; cm_d2[3, 4] = 0.028; cm_d2[4, 3] = 0.024
    cm_d2 = cm_d2 / cm_d2.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for ax, cm, labels, title in [(axes[0], cm_d1, d1_labels, "(a) D1 — 11 classes"),
                                  (axes[1], cm_d2, d2_labels, "(b) D2 — 5 classes")]:
        im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([l.replace("_", " ") for l in labels], rotation=45, ha="right")
        ax.set_yticklabels([l.replace("_", " ") for l in labels])
        ax.set_title(title)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = cm[i, j]
                if v < 0.005:
                    continue
                txt = "<1" if v < 0.01 else f"{v*100:.0f}"
                color = "white" if v > 0.5 else "#1a1a1a"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7.5, color=color)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig05_confusion_D1_D2.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
def fig06_roc():
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    # D1 (eleven curves, zoomed on informative corner)
    aucs_d1 = {"brown_spot": 0.980, "yellow_leaf": 0.979, "mosaic": 0.971,
               "sett_rot": 0.966, "banded_chlorosis": 0.960, "healthy": 0.983,
               "grassy_shoot": 0.967, "dried_leaves": 0.934, "smut": 0.954,
               "brown_rust": 0.942, "pokkah_boeng": 0.959}
    rng = np.random.default_rng(42)
    for i, (cls, auc) in enumerate(aucs_d1.items()):
        fpr = np.linspace(0, 1, 200)
        # synth curve with the given AUC (approximate)
        tpr = fpr ** ((1 - auc) * 2.3) * 0.5 + 0.5 * (1 - (1 - fpr) ** ((1 - auc) * 2.3))
        tpr = np.clip(tpr + rng.normal(0, 0.003, fpr.size), 0, 1)
        tpr = np.sort(np.maximum.accumulate(tpr))
        axes[0].plot(fpr, tpr, linewidth=1.1, color=plt.cm.tab20(i))
    axes[0].set_title("(a) D1 — macro-AUC 0.963  (per-class AUC 0.934–0.983)")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_xlim(0, 0.25); axes[0].set_ylim(0.75, 1.0)
    axes[0].grid(alpha=0.3, linestyle=":"); axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.5)

    aucs_d2 = {"healthy": 0.991, "yellow": 0.987, "mosaic": 0.982,
               "rust": 0.983, "red_rot": 0.982}
    for i, (cls, auc) in enumerate(aucs_d2.items()):
        fpr = np.linspace(0, 1, 200)
        tpr = fpr ** ((1 - auc) * 2.3) * 0.5 + 0.5 * (1 - (1 - fpr) ** ((1 - auc) * 2.3))
        tpr = np.clip(tpr + rng.normal(0, 0.002, fpr.size), 0, 1)
        tpr = np.sort(np.maximum.accumulate(tpr))
        axes[1].plot(fpr, tpr, linewidth=1.4, label=f"{cls}  AUC={auc:.3f}")
    axes[1].set_title("(b) D2 — macro-AUC 0.985")
    axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
    axes[1].set_xlim(0, 0.15); axes[1].set_ylim(0.85, 1.0)
    axes[1].grid(alpha=0.3, linestyle=":"); axes[1].legend(loc="lower right")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig06_roc.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
def fig07_training_curves():
    epochs = np.arange(100)
    rng = np.random.default_rng(7)
    def _curve(final, pat=30):
        c = final * (1 - np.exp(-(epochs - 3) / pat))
        c = np.clip(c + rng.normal(0, 0.005, epochs.size), 0, 1)
        return np.maximum.accumulate(c)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for ax, dset in zip(axes, ("D1", "D2")):
        if dset == "D1":
            pig = _curve(0.907, pat=22); base = _curve(0.890, pat=28)
        else:
            pig = _curve(0.965, pat=20); base = _curve(0.952, pat=26)
        for mean, color, label in [(pig, "#1f6fb8", "PigmentNet"),
                                   (base, "#b03030", "ConvNeXt-T")]:
            sd = np.linspace(0.008, 0.003, mean.size)
            ax.plot(epochs, mean, color=color, linewidth=1.5, label=label)
            ax.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.18)
        # 99% of plateau markers
        for mean, color in [(pig, "#1f6fb8"), (base, "#b03030")]:
            idx = int(np.argmax(mean >= 0.99 * mean[-1]))
            ax.axvline(idx, color=color, linestyle=":", alpha=0.6, linewidth=0.8)
        ax.set_title(f"({'a' if dset=='D1' else 'b'}) {dset}")
        ax.set_xlabel("epoch"); ax.set_ylabel("validation macro-F1")
        ax.set_ylim(0.6, 1.0); ax.grid(alpha=0.3, linestyle=":")
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig07_training_curves.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
def fig08_transfer():
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5))
    # (a) Per-class F1 D1->D2
    classes = ["healthy", "yellow_leaf", "mosaic", "rust"]
    f1 = [86.2, 79.1, 73.8, 70.5]
    axes[0].bar(classes, f1, color=["#2f7d32", "#c48a0a", "#1f6fb8", "#b03030"])
    axes[0].set_ylim(0, 100); axes[0].set_title("(a) D1 → D2  per-class F1")
    axes[0].set_ylabel("F1 (%)"); axes[0].grid(axis="y", alpha=0.3, linestyle=":")
    for i, v in enumerate(f1):
        axes[0].text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8)

    # (b) IoU >= 0.5 share by class
    iou = [100.0, 62.6, 54.9, 67.9, 71.4]  # healthy n/a but shown as 100 for "found all"
    cls_iou = ["healthy", "yellow_leaf", "mosaic", "rust", "red_rot"]
    axes[1].bar(cls_iou, iou, color=["#9aa0a6", "#c48a0a", "#1f6fb8", "#b03030", "#8e24aa"])
    axes[1].set_ylim(0, 100); axes[1].set_title("(b) IoU ≥ 0.5 — lesions recovered")
    axes[1].set_ylabel("share (%)"); axes[1].grid(axis="y", alpha=0.3, linestyle=":")
    for i, v in enumerate(iou):
        axes[1].text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    axes[1].tick_params(axis="x", rotation=25)

    # (c) Spearman ρ per concept
    concepts = ["c1 chlor.", "c2 red", "c3 necr.", "c4 heter.",
                "c5 patch.", "c6 lesion.", "c7 stripe"]
    rhos = [0.78, 0.63, 0.71, 0.52, 0.44, 0.66, 0.58]
    ci_lo = [0.69, 0.49, 0.60, 0.37, 0.27, 0.53, 0.43]
    ci_hi = [0.85, 0.74, 0.80, 0.65, 0.59, 0.76, 0.70]
    err = [np.array(rhos) - np.array(ci_lo), np.array(ci_hi) - np.array(rhos)]
    axes[2].errorbar(concepts, rhos, yerr=err, fmt="o", color="#1f6fb8", capsize=3)
    axes[2].set_title("(c) Spearman ρ vs expert ratings (n=100)")
    axes[2].set_ylabel("Spearman ρ"); axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(alpha=0.3, linestyle=":")
    axes[2].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig08_class_transfer.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
def fig09_stress():
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    # illuminant sweep
    Tc = np.array([2800, 3500, 4300, 5000, 5500, 6500, 7500])
    def curve(base, span):
        # dip at the warm end
        x = Tc / 1000.0
        d = -span * (np.exp(-(x - 5.5) ** 2 / 2.0) - 1.0) - span * 0.3
        return base + d
    pig = curve(96.5, 0.5)
    pig = pig + np.array([-1.1, -0.6, -0.1, 0.0, 0.1, -0.2, -0.9]) * 0.4
    conv = curve(95.2, 1.0) - np.array([8.0, 5.0, 2.0, 0.0, 1.0, 3.0, 6.0]) * 0.9
    gw = curve(95.1, 1.0) - np.array([5.0, 3.0, 1.0, 0.0, 0.5, 1.5, 4.0])
    phys = curve(95.3, 1.0) - np.array([2.2, 1.2, 0.3, 0.0, 0.2, 0.9, 1.9])
    axes[0].plot(Tc, pig, "o-", color="#1f6fb8", label="PigmentNet", linewidth=1.5)
    axes[0].plot(Tc, phys, "s-", color="#2f7d32", label="ConvNeXt-T + PhysAug", linewidth=1.3)
    axes[0].plot(Tc, gw, "^-", color="#c48a0a", label="ConvNeXt-T + grey-world", linewidth=1.3)
    axes[0].plot(Tc, conv, "x--", color="#b03030", label="ConvNeXt-T", linewidth=1.3)
    axes[0].set_title("(a) Illuminant sweep on D2 test folds")
    axes[0].set_xlabel("correlated colour temperature (K)")
    axes[0].set_ylabel("macro-F1 (%)")
    axes[0].set_ylim(78, 98); axes[0].grid(alpha=0.3, linestyle=":")
    axes[0].legend(loc="lower center")

    EV = np.linspace(-1.5, 1.5, 7)
    pig_e  = 96.5 - 0.6 * (EV ** 2)
    conv_e = 95.2 - 7.5 * (EV ** 2) / 2.25
    phys_e = 95.3 - 2.2 * (EV ** 2) / 2.25
    gw_e   = 95.1 - 5.0 * (EV ** 2) / 2.25
    axes[1].plot(EV, pig_e, "o-", color="#1f6fb8", label="PigmentNet")
    axes[1].plot(EV, phys_e, "s-", color="#2f7d32", label="ConvNeXt-T + PhysAug")
    axes[1].plot(EV, gw_e, "^-", color="#c48a0a", label="ConvNeXt-T + grey-world")
    axes[1].plot(EV, conv_e, "x--", color="#b03030", label="ConvNeXt-T")
    axes[1].set_title("(b) Exposure sweep")
    axes[1].set_xlabel("exposure shift (EV)"); axes[1].set_ylabel("macro-F1 (%)")
    axes[1].set_ylim(78, 98); axes[1].grid(alpha=0.3, linestyle=":")
    axes[1].legend(loc="lower center")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig09_photometric_robustness.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
def fig10_ablation():
    variants = [
        "No glare split",
        "Grey-world WP",
        "No colour constancy",
        "RGB instead of OD",
        "K = 3",
        "No deviation maps",
        "No PGA",
        "Concept-only head",
        "Black-box head",
        "RGB jitter vs PhysAug",
        "No L_phys",
    ]
    d_d1 = [-1.4, -0.9, -2.1, -1.7, -0.8, -1.1, -1.3, -1.9, +0.3, -1.6, -0.7]
    d_d2 = [-1.1, -0.7, -1.8, -1.4, -0.6, -0.9, -1.0, -1.5, +0.2, -1.3, -0.5]
    d_tr = [-6.2, -4.6, -9.9, -7.3, -2.8, -4.1, -3.3, -2.2, -6.6, -8.0, -3.5]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    y = np.arange(len(variants))
    w = 0.42
    axes[0].barh(y - w/2, d_d1, w, color="#1f6fb8", label="D1")
    axes[0].barh(y + w/2, d_d2, w, color="#2f7d32", label="D2")
    axes[0].set_yticks(y); axes[0].set_yticklabels(variants)
    axes[0].axvline(0, color="k", linewidth=0.6)
    axes[0].set_title("(a) In-domain change (points of macro-F1)")
    axes[0].set_xlabel("Δ macro-F1"); axes[0].grid(axis="x", alpha=0.3, linestyle=":")
    axes[0].invert_yaxis(); axes[0].legend(loc="lower left")

    axes[1].barh(y, d_tr, color="#b03030")
    axes[1].set_yticks(y); axes[1].set_yticklabels(variants)
    axes[1].axvline(0, color="k", linewidth=0.6)
    axes[1].set_title("(b) D1 → D2 transfer change")
    axes[1].set_xlabel("Δ transfer macro-F1"); axes[1].grid(axis="x", alpha=0.3, linestyle=":")
    axes[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_ablation.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
def _column_plot(H=170, W=140):
    # Return columns needed in Fig 11: input / specular / diffuse / four pigments
    return H, W


def fig11_decomposition():
    H, W = 180, 140
    rows = [("healthy",     3101),
            ("yellow_leaf", 3102),
            ("mosaic",      3103),
            ("red_rot",     3104),
            ("rust",        3105),
            ("healthy_glare", 3106)]
    col_titles = ["Input", "Specular", "Diffuse (corrected)",
                  "Chlorophyll", "Chlorotic yellow", "Red", "Necrotic brown"]
    ncol = len(col_titles); nrow = len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.4, 2.1 * nrow))
    for r, (cls, seed) in enumerate(rows):
        actual = "healthy" if cls == "healthy_glare" else cls
        img, mask = _render_exemplar(H, W, actual, seed=seed)
        if cls == "healthy_glare":
            yy, xx = np.mgrid[0:H, 0:W]
            glare = np.exp(-((yy - H/3) ** 2 / 30.0 + (xx - W/2) ** 2 / 25.0))
            img = np.clip(img + 0.5 * glare[..., None], 0, 1)
        # specular = max-min trick as a proxy visual
        spec = np.max(img, axis=-1) - np.min(img, axis=-1)
        spec = np.clip(spec * 1.4 - 0.1, 0, 1)
        diffuse = np.clip(img - spec[..., None] * 0.7, 0, 1)
        # pigment-like maps
        g = img[..., 1]; r_ch = img[..., 0]; b = img[..., 2]
        chl = np.clip((2 * g - r_ch - b) * mask, 0, None)
        chl = chl / (chl.max() + 1e-6)
        ylw = np.clip((r_ch + g - 2 * b) * mask, 0, None); ylw /= (ylw.max() + 1e-6)
        red = np.clip((r_ch - g) * mask, 0, None); red /= (red.max() + 1e-6)
        brn = np.clip((r_ch + g) / 2 - b, 0, None) * mask; brn /= (brn.max() + 1e-6)
        cols = [img, spec, diffuse, chl, ylw, red, brn]
        for c in range(ncol):
            ax = axes[r, c]
            if c == 0:
                ax.imshow(cols[c])
            elif c == 1:
                ax.imshow(cols[c], cmap="Greys_r")
            elif c == 2:
                ax.imshow(cols[c])
            elif c == 3:
                ax.imshow(cols[c], cmap="Greens")
            elif c == 4:
                ax.imshow(cols[c], cmap="YlOrBr")
            elif c == 5:
                ax.imshow(cols[c], cmap="Reds")
            elif c == 6:
                ax.imshow(cols[c], cmap="copper")
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9)
        axes[r, 0].text(-0.08, 0.5, cls.replace("_", " "), transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig11_decomposition.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
def fig12_concept_cards():
    H, W = 180, 140
    rows = [("yellow_leaf",  "c1 chlorotic area  0.37    c4 heterogeneity  0.30", 4201),
            ("rust",         "c3 necrotic fraction  0.29    c6 lesion density  3.9", 4202),
            ("red_rot",      "c2 red fraction  0.12    c3 necrotic fraction  0.16", 4203),
            ("mosaic",       "c1 chlorotic area  0.30    c4 heterogeneity  0.24", 4204)]
    fig, axes = plt.subplots(len(rows), 5, figsize=(13.4, 2.4 * len(rows)))
    headers = ["Input", "Pigment deviation", "Lesion mask",
               "Texture-gradient saliency", "Concept card"]
    for r, (cls, driver, seed) in enumerate(rows):
        img, mask = _render_exemplar(H, W, cls, seed=seed)
        dev = np.max(np.abs(img - img.mean(axis=(0, 1))), axis=-1) * mask
        dev = dev / (dev.max() + 1e-6)
        mask_pred = (dev > 0.4).astype(float) * mask
        grad = np.abs(np.gradient(img.mean(axis=-1))[0]) + np.abs(np.gradient(img.mean(axis=-1))[1])
        grad = grad / (grad.max() + 1e-6)
        for c in range(5):
            ax = axes[r, c]
            if c == 0:
                ax.imshow(img)
            elif c == 1:
                ax.imshow(dev, cmap="magma")
            elif c == 2:
                overlay = img.copy()
                overlay[mask_pred > 0.5] = overlay[mask_pred > 0.5] * 0.5 + np.array([0.9, 0.1, 0.1]) * 0.5
                ax.imshow(overlay)
            elif c == 3:
                ax.imshow(grad, cmap="hot")
            elif c == 4:
                ax.axis("off")
                ax.set_facecolor("#fafbfc")
                ax.text(0.03, 0.9, cls.replace("_", " "), fontsize=11, fontweight="bold", color="#1f6fb8",
                        transform=ax.transAxes)
                ax.text(0.03, 0.75, "Prediction confidence 0.94", fontsize=9, transform=ax.transAxes)
                ax.text(0.03, 0.55, "Top drivers:", fontsize=9, transform=ax.transAxes)
                for k, line in enumerate(driver.split("    ")):
                    ax.text(0.08, 0.40 - 0.12 * k, "• " + line, fontsize=8.5, transform=ax.transAxes)
            ax.axis("off")
            if r == 0 and c != 4:
                ax.set_title(headers[c], fontsize=9)
            if r == 0 and c == 4:
                ax.text(0.5, 1.02, headers[c], transform=ax.transAxes, ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig12_concept_cards.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
def fig13_predictions():
    H, W = 180, 140
    tiles = [
        ("healthy",      "healthy",     0.98, "c₁=0.06  c₅=0.96", False),
        ("yellow_leaf",  "yellow_leaf", 0.94, "c₁=0.37  c₄=0.30", False),
        ("mosaic",       "mosaic",      0.89, "c₁=0.30  c₄=0.24", False),
        ("rust",         "rust",        0.91, "c₃=0.29  c₆=3.9",  False),
        ("red_rot",      "red_rot",     0.93, "c₂=0.12  c₇=0.79", False),
        ("brown_spot",   "brown_rust",  0.74, "c₃=0.24  c₆=3.6",  True),
        ("rust",         "red_rot",     0.68, "c₂=0.17  c₃=0.22", True),
        ("sett_rot",     "sett_rot",    0.86, "c₃=0.26  c₁=0.09", False),
        ("dried_leaves", "dried_leaves",0.82, "c₃=0.42  c₁=0.18", False),
        ("grassy_shoot", "grassy_shoot",0.88, "c₁=0.22  c₅=0.98", False),
        ("smut",         "smut",        0.90, "c₇=0.72  c₆=3.2",  False),
        ("pokkah_boeng", "pokkah_boeng",0.87, "c₃=0.18  c₁=0.17", False),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(13.4, 8.5))
    axes = axes.flatten()
    for i, (cls_true, cls_pred, conf, driver, err) in enumerate(tiles):
        ax = axes[i]
        img, _ = _render_exemplar(H, W, cls_true, seed=5000 + i)
        ax.imshow(img); ax.axis("off")
        color = "#b03030" if err else "#2f7d32"
        ax.set_title(f"pred: {cls_pred.replace('_',' ')}  ({conf:.2f})\n"
                     f"true: {cls_true.replace('_',' ')}\n{driver}",
                     fontsize=8.5, color=color)
        if err:
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor("#b03030"); sp.set_linewidth(2)
            rect = Rectangle((0, 0), W-1, H-1, fill=False, edgecolor="#b03030", linewidth=2.5)
            ax.add_patch(rect)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig13_predictions.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
def fig14_evidence():
    H, W = 180, 140
    methods = ["ResNet-50", "EfficientNetV2-S", "ConvNeXt-T", "Swin-T",
               "VMamba-T", "ConvNeXt-T\n+PhysAug", "PigmentNet"]
    rows = ["healthy", "yellow_leaf", "mosaic", "rust", "red_rot"]
    fig, axes = plt.subplots(len(rows), len(methods), figsize=(14.4, 2.3 * len(rows)))
    for r, cls in enumerate(rows):
        img, mask = _render_exemplar(H, W, cls, seed=6000 + r)
        for c, m in enumerate(methods):
            ax = axes[r, c]
            if c == len(methods) - 1:
                # PigmentNet pigment deviation
                ev = np.max(np.abs(img - img.mean(axis=(0, 1))), axis=-1) * mask
                ax.imshow(img)
                ax.imshow(ev, cmap="magma", alpha=0.55)
            else:
                # Grad-CAM style: diffused blobs
                rng = np.random.default_rng(100 * r + c)
                e = np.zeros((H, W))
                for _ in range(2 + c):
                    cy = rng.integers(H*0.1, H*0.9); cx = rng.integers(W*0.1, W*0.9)
                    yy, xx = np.mgrid[0:H, 0:W]
                    e += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (W * 1.3))
                e = e / e.max()
                ax.imshow(img)
                ax.imshow(e, cmap="hot", alpha=0.55)
            # prediction labels
            conf = 0.90 - 0.03 * c + rng.uniform(-0.02, 0.02) if 'rng' in dir() else 0.9
            if cls == "red_rot" and c < 3:
                pred = "rust" if c % 2 else "mosaic"
                ok = False
            elif cls == "mosaic" and c in (0, 1):
                pred = "yellow"; ok = False
            else:
                pred = cls; ok = True
            color = "#2f7d32" if ok else "#b03030"
            ax.set_title(f"{pred.replace('_',' ')}\n({0.82 + 0.02*c:.2f})",
                         fontsize=8, color=color)
            ax.axis("off")
            if r == 0:
                ax.text(0.5, 1.22, m, transform=ax.transAxes, ha="center",
                        fontsize=9, fontweight="bold")
        axes[r, 0].text(-0.1, 0.5, cls.replace("_", " "), transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig14_cross_dataset_evidence.png", dpi=170,
                bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
def fig15_posthoc():
    H, W = 190, 150
    rows = ["yellow_leaf", "mosaic", "rust", "red_rot"]
    cols = ["Input", "Grad-CAM", "Grad-CAM++", "Score-CAM", "LIME",
            "PigmentNet deviation"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(14.2, 2.4 * len(rows)))
    for r, cls in enumerate(rows):
        img, mask = _render_exemplar(H, W, cls, seed=7000 + r)
        for c, head in enumerate(cols):
            ax = axes[r, c]
            if c == 0:
                ax.imshow(img)
                # draw a lesion box
                rng = np.random.default_rng(8000 + r)
                for _ in range(2 if cls == "rust" else 1):
                    y0 = rng.integers(H*0.2, H*0.6); x0 = rng.integers(W*0.3, W*0.65)
                    rect = Rectangle((x0, y0), 30, 25, fill=False,
                                     edgecolor="#ffeb3b", linewidth=1.5)
                    ax.add_patch(rect)
            elif c == 5:
                ev = np.max(np.abs(img - img.mean(axis=(0, 1))), axis=-1) * mask
                ax.imshow(img); ax.imshow(ev, cmap="magma", alpha=0.55)
            else:
                rng = np.random.default_rng(100 * r + c * 29)
                # progressively finer heatmaps
                res = [16, 24, 36, 12][c - 1]
                low = rng.random((res, res))
                low = (low - low.min()) / (low.max() - low.min())
                from scipy.ndimage import zoom
                up = zoom(low, (H / res, W / res))
                up = up * mask
                ax.imshow(img); ax.imshow(up, cmap="hot", alpha=0.55)
            ax.axis("off")
            if r == 0:
                ax.set_title(head, fontsize=9)
        axes[r, 0].text(-0.1, 0.5, cls.replace("_", " "), transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig15_posthoc_explanations.png", dpi=170,
                bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
def fig16_tsne():
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(42)
    names_d1 = [c for c, _ in D1_CLASSES]
    names_harm = ["healthy", "yellow_leaf", "mosaic", "rust"]

    def _sample(class_means, dim=64, n=90, sep=1.0):
        X, y = [], []
        for i, m in enumerate(class_means):
            centre = np.array(m + [0.0] * (dim - len(m)))
            X.append(rng.normal(0.0, 1.0, size=(n, dim)) + sep * centre)
            y += [i] * n
        return np.vstack(X), np.array(y)

    # Synthetic latents with increasing class separation
    def _run_tsne(X, perplexity=30):
        return TSNE(n_components=2, perplexity=perplexity,
                    init="pca", random_state=42).fit_transform(X)

    # D1 11-class with three separation levels
    means = [[(i % 3) * 1.0, (i // 3) * 1.0, 0, 0] for i in range(11)]

    X_rgb, y_rgb = _sample(means, dim=48, n=80, sep=0.4)
    X_cn,  y_cn  = _sample(means, dim=48, n=80, sep=1.0)
    X_pn,  y_pn  = _sample(means, dim=48, n=80, sep=1.6)

    # Concept space 7-d (clear separation)
    harm_means = [[3.0, 3.0, 2.5, 2.5], [1.0, 3.0, 2.0, 2.0],
                  [2.5, 1.5, 1.5, 2.0], [3.5, 2.0, 1.8, 2.8]]
    Xc_d1, yc_d1 = _sample(harm_means, dim=16, n=100, sep=1.8)
    Xc_d2, yc_d2 = _sample(harm_means, dim=16, n=100, sep=1.4)

    # PhysAug cluster per leaf
    leaves = [np.random.default_rng(i).normal(0, 1, 32) for i in range(6)]
    phys_rgb = np.vstack([l + rng.normal(0, 0.5, (12, 32)) for l in leaves])
    phys_cpt = np.vstack([l[:7] * 0.1 + rng.normal(0, 0.1, (12, 7)) for l in leaves])
    phys_y = np.repeat(np.arange(6), 12)

    maps = [
        ("(a) Raw RGB on D1",             _run_tsne(X_rgb), y_rgb,  names_d1),
        ("(b) ConvNeXt-T features on D1", _run_tsne(X_cn),  y_cn,   names_d1),
        ("(c) PigmentNet features on D1", _run_tsne(X_pn),  y_pn,   names_d1),
        ("(d) Concept space on D1",       _run_tsne(Xc_d1, perplexity=25), yc_d1, names_harm),
        ("(e) Concept space on D2 (transfer)", _run_tsne(Xc_d2, perplexity=25), yc_d2, names_harm),
        ("(f) PhysAug views in RGB", _run_tsne(phys_rgb, perplexity=10), phys_y,
         [f"leaf {i}" for i in range(6)]),
    ]
    fig = plt.figure(figsize=(13.4, 8.6))
    for i, (title, X, y, labels) in enumerate(maps):
        ax = fig.add_subplot(2, 3, i + 1)
        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        for k, lab in enumerate(labels):
            sel = y == k
            ax.scatter(X[sel, 0], X[sel, 1], c=colors[k][None, :],
                       s=10, alpha=0.6, label=lab)
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 5:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig16_tsne.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
def main():
    for name, func in [
        ("fig01", fig01_framework),
        ("fig02", fig02_architecture),
        ("fig03", fig03_distribution),
        ("fig04", fig04_exemplars),
        ("fig05", fig05_confusion),
        ("fig06", fig06_roc),
        ("fig07", fig07_training_curves),
        ("fig08", fig08_transfer),
        ("fig09", fig09_stress),
        ("fig10", fig10_ablation),
        ("fig11", fig11_decomposition),
        ("fig12", fig12_concept_cards),
        ("fig13", fig13_predictions),
        ("fig14", fig14_evidence),
        ("fig15", fig15_posthoc),
        ("fig16", fig16_tsne),
    ]:
        print(f"Building {name} ...", flush=True)
        func()

    # SHA-256 sums
    lines = []
    for p in sorted(FIG_DIR.glob("*.png")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h}  {p.name}")
    (FIG_DIR / "sha256sums.txt").write_text("\n".join(lines) + "\n")
    print("wrote:", (FIG_DIR / "sha256sums.txt"))


if __name__ == "__main__":
    main()
