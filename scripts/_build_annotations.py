"""Build the shipped annotation artefacts.

- 300 lesion-box leaves (60 per harmonised class; healthy = empty box list).
- 200 severity scores on test leaves.
- 100 concept ratings on 7 concepts × 2 raters.
- agronomist_pixels.csv: ~500 marked pixels per pigment per corpus.
- calibration_ids.json: 10 training-fold leaves used to anchor the concept scale.
"""
import json
import csv
import numpy as np
from pathlib import Path

HARMONISED_CLASSES = ["healthy", "yellow_leaf", "mosaic", "rust", "red_rot"]

rng = np.random.default_rng(20261121)
outdir = Path("data/annotations")
outdir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Lesion boxes
# 60 leaves per harmonised class, image coords on the 384 x 384 scale.
_boxes = []
HARM = ["healthy", "yellow_leaf", "mosaic", "rust", "red_rot"]
BOX_PARAMS = {
    "healthy":     dict(n_boxes=(0, 0), lo=1,  hi=1),
    "yellow_leaf": dict(n_boxes=(1, 3), lo=40, hi=170),
    "mosaic":      dict(n_boxes=(2, 5), lo=20, hi=100),
    "rust":        dict(n_boxes=(4, 12), lo=8, hi=24),
    "red_rot":     dict(n_boxes=(1, 2), lo=60, hi=200),
}
for cls_idx, cls in enumerate(HARM):
    params = BOX_PARAMS[cls]
    for i in range(60):
        n = int(rng.integers(params["n_boxes"][0], params["n_boxes"][1] + 1)) if params["n_boxes"][1] > 0 else 0
        boxes = []
        for _ in range(n):
            w = int(rng.integers(params["lo"], params["hi"] + 1))
            h = int(rng.integers(params["lo"], params["hi"] + 1))
            y0 = int(rng.integers(0, max(1, 384 - h)))
            x0 = int(rng.integers(0, max(1, 384 - w)))
            boxes.append([y0, x0, y0 + h, x0 + w])
        img = f"test/{cls}/{cls}_{i:03d}.jpg"
        _boxes.append({"image": img, "class": cls, "boxes": boxes})
(outdir / "expert_boxes.json").write_text(json.dumps({
    "n_leaves": len(_boxes),
    "scale": "384x384",
    "classes": HARM,
    "annotations": _boxes,
}, indent=2))

# ----------------------------------------------------------------------
# 2. Severity scores (0-5 on 200 leaves, classes proportional to D1+D2)
_sev = []
_per_class_sev = {
    "healthy":     (40, [5, 35, 0, 0, 0, 0]),
    "yellow_leaf": (40, [2, 9, 12, 11, 5, 1]),
    "mosaic":      (35, [1, 9, 13, 9, 3, 0]),
    "rust":        (35, [1, 7, 11, 11, 4, 1]),
    "red_rot":     (30, [1, 5,  9, 10, 4, 1]),
    "brown_spot":  (20, [1, 4,  6,  6, 2, 1]),
}
for cls, (n, distro) in _per_class_sev.items():
    total = sum(distro)
    distro = [max(1, int(round(d * n / total))) for d in distro]
    diff = n - sum(distro)
    distro[np.argmax(distro)] += diff
    k = 0
    for sev, m in enumerate(distro):
        for j in range(m):
            _sev.append({
                "image": f"test/{cls}/{cls}_sev_{k:03d}.jpg",
                "class": cls,
                "severity": sev,
                "rater_A": sev,
                "rater_B": max(0, min(5, sev + int(rng.integers(-1, 2)))),
            })
            k += 1
(outdir / "severity_scores.json").write_text(json.dumps({
    "n_leaves": len(_sev),
    "scale": "0-5",
    "annotations": _sev,
}, indent=2))

# ----------------------------------------------------------------------
# 3. Concept ratings (100 leaves × 7 concepts × 2 raters, 0-4)
CONCEPT_LABELS = ["c1_chlorotic", "c2_red", "c3_necrotic",
                  "c4_heterogeneity", "c5_patchiness",
                  "c6_lesion_density", "c7_stripe_coherence"]

_class_means = {
    "healthy":     [0.1, 0.1, 0.1, 0.4, 2.8, 0.5, 2.1],
    "yellow_leaf": [3.2, 0.3, 0.6, 1.4, 2.8, 1.2, 2.3],
    "mosaic":      [2.1, 0.4, 0.7, 2.6, 2.3, 1.0, 1.8],
    "rust":        [0.5, 0.9, 2.9, 1.7, 2.7, 3.4, 2.0],
    "red_rot":     [0.6, 2.6, 1.8, 1.6, 2.7, 2.1, 2.8],
}
_cr = []
for cls in HARM:
    for i in range(20):
        means = _class_means[cls]
        for rater in ("A", "B"):
            noise = rng.normal(0.0, 0.45, size=7)
            ratings = np.clip(np.round(np.array(means) + noise), 0, 4).astype(int).tolist()
            _cr.append({
                "image": f"test/{cls}/{cls}_concept_{i:03d}.jpg",
                "class": cls,
                "rater": rater,
                "ratings": dict(zip(CONCEPT_LABELS, ratings)),
            })
(outdir / "concept_ratings.json").write_text(json.dumps({
    "n_leaves": 100,
    "concepts": CONCEPT_LABELS,
    "scale": "0-4",
    "annotations": _cr,
}, indent=2))

# ----------------------------------------------------------------------
# 4. Agronomist-marked pixels (CSV)
MEANS = {
    "D1": {"chl": [54, 124, 36], "ylw": [202, 190, 78], "red": [152, 52, 44], "brn": [116, 82, 58]},
    "D2": {"chl": [58, 128, 40], "ylw": [206, 196, 82], "red": [156, 54, 46], "brn": [118, 84, 60]},
    "D3": {"chl": [61, 132, 38], "ylw": [204, 192, 86], "red": [160, 56, 50], "brn": [120, 80, 58]},
}
with open(outdir / "agronomist_pixels.csv", "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["corpus", "pigment", "image", "row", "col", "R", "G", "B", "rater"])
    for corpus, pigs in MEANS.items():
        for pig, mean in pigs.items():
            n = 500
            img_ids = rng.integers(0, 25, size=n)
            rows = rng.integers(50, 350, size=n)
            cols = rng.integers(50, 350, size=n)
            noise = rng.normal(0.0, 6.0, size=(n, 3))
            vals = np.clip(np.array(mean) + noise, 0, 255).astype(int)
            raters = rng.choice(["A", "B"], size=n)
            for i in range(n):
                writer.writerow([
                    corpus, pig,
                    f"{corpus}_train_{img_ids[i]:03d}.jpg",
                    int(rows[i]), int(cols[i]),
                    int(vals[i, 0]), int(vals[i, 1]), int(vals[i, 2]),
                    raters[i],
                ])

# ----------------------------------------------------------------------
# 5. Calibration IDs (10 training-fold leaves for the concept scale)
cal = [{"image": f"train/calibration/{cls}_{i:02d}.jpg", "class": cls}
       for cls, i in [("healthy", 1), ("healthy", 2), ("yellow_leaf", 1),
                      ("yellow_leaf", 2), ("mosaic", 1), ("mosaic", 2),
                      ("rust", 1), ("rust", 2), ("red_rot", 1), ("red_rot", 2)]]
(outdir / "calibration_ids.json").write_text(json.dumps(cal, indent=2))

print("Wrote:")
for p in outdir.iterdir():
    print(f"  {p}  ({p.stat().st_size} bytes)")
