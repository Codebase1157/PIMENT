# Expert annotation guidelines

This document is the protocol followed by the two agronomists who annotated
sugarcane leaves for this paper. It covers four tasks:

1. **Agronomist-marked pixels** used to estimate the prior absorbance matrix
   E₀ (Section 4.2.5 and Eq. 15).
2. **Lesion boxes** on 300 test leaves (60 per harmonised class) used for the
   pointing-game and IoU scoring (Section 6.4).
3. **Severity scores** on 200 test leaves on a 0–5 ordinal scale
   (Section 6.4).
4. **Concept ratings** on 100 test leaves on a 0–4 ordinal scale per concept
   (Section 6.7).

All annotation was carried out before any model output was computed for the
same images. Raters did not see the predicted class, the predicted mask or
the concept vector, and they worked independently.

---

## Shared conventions

- **Viewing**: images are shown at full resolution on a sRGB-calibrated LCD
  with ambient light at roughly 300 lx and the display set to ≈100 cd/m².
- **Order**: images are shown in a fixed random order (seed 42), balanced
  across classes.
- **Time**: at most 45 seconds per leaf; raters take a five-minute break
  every twenty leaves.
- **Flags**: raters may mark an image *unrateable* for reasons outside the
  measurement (whole-blade blur, out-of-focus, more than half the blade
  occluded); such images are excluded from aggregation.

---

## Task 1 — Pigment pixel marking for E₀

**Goal.** Supply ~500 pixels per pigment per corpus from which the prior
absorbance matrix E₀ is estimated. One pixel is one (image, row, column)
triple with its linear-sRGB value.

**Pigments to mark**
- `chl` — healthy chlorophyll-green tissue, well-lit and non-specular.
- `ylw` — chlorotic yellowing (yellow leaf, mosaic pale bands, early
  symptoms).
- `red` — red rot red lesion interior and red stripe mid-lesion.
- `brn` — necrotic brown (rust pustule centre, brown-spot necrosis).

**Procedure.**

1. Open the leaf in the annotation tool. Zoom to 200% and pan across the
   blade.
2. Switch to the single-pixel tool. Avoid veins unless the symptom is on
   the vein.
3. Avoid specular pixels (anywhere the channel maximum is above 0.95 in
   sRGB), pixels within three pixels of a shadow boundary and pixels at the
   blade margin.
4. Mark 15–25 pixels per present pigment on the image. Mark **only** the
   pigments that are visible on the leaf; do not supply speculative pixels
   for a pigment that is not there.
5. Save; the tool records (`corpus`, `image`, `row`, `col`, `R`, `G`, `B`,
   `pigment`, `rater`).

**Minimum per pigment per corpus:** 400 pixels from at least 20 images.

**Quality control.** For each (corpus, pigment) the tool rejects any pixel
whose linear-sRGB is more than three robust standard deviations from the
median of the pixels marked for that pigment, flagging the image for
re-inspection.

Output CSV columns: `corpus,pigment,image,row,col,R,G,B,rater`.

---

## Task 2 — Lesion boxes

**Goal.** Axis-aligned rectangles tight around every visible lesion on 300
test leaves (60 per harmonised class: healthy, yellow leaf, mosaic, rust,
red rot). Healthy leaves produce zero boxes.

**What counts as a lesion?**

- **Yellow leaf** — the chlorotic patch of the blade, bounded by its visible
  edge. One box per contiguous chlorotic region.
- **Mosaic** — one box per visible pale stripe or mottled region.
- **Rust** — one box per discrete necrotic pustule or coalesced group whose
  members overlap.
- **Red rot** — one box per red stain, following the red colour, not the
  surrounding necrosis.

**Procedure.**

1. Zoom to fit, then adjust to see at least a full blade width.
2. Click–drag to place a box; adjust edges until the box is tight on the
   lesion without clipping any visible stained pixel.
3. Mark very small lesions only when their longest axis is ≥ 6 px at the
   nominal 384 × 384 input scale.
4. Do not merge spatially distinct lesions. Two lesions overlapping by less
   than 20% stay as two boxes.

**Scoring.** Pointing-game (share of leaves whose peak model evidence falls
in a box) and IoU ≥ 0.5 are computed against the union of boxes per image.

Output JSON per leaf:
```json
{"image": "D2/rust/img_0123.jpg", "boxes": [[y0, x0, y1, x1], ...]}
```

---

## Task 3 — Severity scoring

**Goal.** A 0–5 ordinal score of overall disease severity on 200 test leaves,
sampled proportional to class frequency.

| Score | Description |
|-------|-------------|
| 0 | Healthy. No symptom visible. |
| 1 | Trace: one small lesion or <5% of blade affected. |
| 2 | Mild: 5–15% of blade affected; discrete lesions only. |
| 3 | Moderate: 15–40% affected, or chlorosis of the blade margin. |
| 4 | Severe: 40–75% affected, or chlorosis of the whole blade. |
| 5 | Dying: >75% affected; blade losing structural integrity. |

**Procedure.**

1. Estimate the share of blade area affected by any symptom.
2. Pick the row whose description best matches; when between rows, pick the
   higher score for red rot (which spreads) and the lower for yellow leaf
   (which progresses more slowly).

Output: `{"image": "...", "severity": int}`.

Agreement is reported as Spearman ρ between the mean of the two raters and
the model's severity estimate (Eq. 34).

---

## Task 4 — Concept ratings

**Goal.** Rate each of the seven decision concepts on 0–4 for 100 test
leaves (60 D2 + 40 D1 shared classes). The glare flag c₈ is **not** rated;
it is a reliability flag, not a symptom.

Raters receive a one-page card listing the seven concepts with their
verbal descriptions:

| Concept | Rate from 0 (absent) to 4 (dominant) |
|---------|--------------------------------------|
| **c₁ chlorotic area** | Share of blade visibly yellowed. 0 = none, 2 ≈ 25%, 4 ≈ >60%. |
| **c₂ red** | Share of blade visibly red or stained red. |
| **c₃ necrotic** | Share of blade visibly brown and dead. |
| **c₄ heterogeneity** | Mottling: is the chlorophyll salt-and-pepper (4) or uniform (0)? |
| **c₅ patchiness** | Coherent patches (4) vs scattered points (0). |
| **c₆ lesion density** | Number of discrete lesions per unit area. |
| **c₇ stripe coherence** | Linear stripes along the blade (4) vs isotropic lesions (0). |

**Procedure.**

1. Calibration: the ten calibration leaves (held out of the training folds)
   are shown first with their expected ratings; raters discuss them with
   the senior rater before the main batch starts.
2. For each leaf, rate every concept visible, independently and without
   access to the other rater's sheet.
3. If a concept is undefined on a given leaf (e.g. stripe coherence on a
   healthy leaf), leave it blank.

**Agreement.** Spearman ρ between each concept and the model value is
computed on the mean of the two raters' scores, with a 95% percentile
bootstrap interval; Krippendorff's α is reported for the "top-3 concepts
judged correct" check (Table 14b last row).

Output: `{"image": "...", "ratings": {"c1": 2, ..., "c7": 1}, "rater": "A"}`.

---

## Labelling tool

A thin Streamlit wrapper around these forms lives in
`tools/annotation_app/`. It writes CSV and JSON in the shapes above, keeps a
per-rater log, and refuses to accept pixels or boxes outside the blade mask
provided by Module A.

## Release

- Lesion boxes: `data/annotations/expert_boxes.json`
- Severity scores: `data/annotations/severity_scores.json`
- Concept ratings: `data/annotations/concept_ratings.json`
- Agronomist-marked pixels: `data/annotations/agronomist_pixels.csv`

The 100-leaf calibration set used to anchor the concept scale is listed
separately in `data/annotations/calibration_ids.json` and is drawn from
training folds only.
