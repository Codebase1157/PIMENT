# Experimental protocols E1–E5

This note is a self-contained reference to the five protocols defined in
Section 5.2 of the paper. All five share the leakage-controlled splitting and
annotation partitioning described below.

## Leakage control

Before any split is drawn:

1. Compute 16×16 perceptual hashes for every image (`scripts/compute_phash.py`).
2. Union–find group images within Hamming distance ≤ 6.
3. Expand each pHash group with any shared capture-session identifier (`exif`
   DateTimeOriginal within 2 seconds on the same device) to form the final
   groups.
4. Build a stratified group k-fold assignment (`scripts/build_folds.py`)
   with k = 5 and a 15% within-fold validation split.

The fold JSONs in `data/folds/` are the artefact of this procedure for D1,
D2, and the external D3 index. Under a random split (no grouping), the same
architectures reach 98.9%/99.2% on D1/D2 and PigmentNet 99.2%/99.5%; under
the grouped split they land at 90.7%/96.5%. Every number reported in the
paper is on the grouped split.

## Annotation partitioning

| Annotation                   | Count | Drawn from       | Used for          |
|------------------------------|-------|------------------|-------------------|
| Leaf masks                   | 150   | training folds   | mask refiner      |
| Agronomist-marked pixels     | ~25 images × 4 pigments × corpus | training folds | E₀ |
| Lesion boxes                 | 300 (60 × 5) | test folds | pointing game, IoU |
| Severity scores (0–5)        | 200   | test folds       | Spearman ρ        |
| Concept ratings (0–4 × 7)    | 100   | test folds       | Table 14a         |

Model selection is on validation macro-F1 alone. The target corpus in any
transfer protocol is touched once, at scoring time.

## E1 — In-domain D1

- 11 classes, 6,748 images.
- 5-fold stratified group CV × 5 seeds (`configs/seeds.yaml`).
- Primary metric: macro-F1. Secondary: balanced accuracy, per-class F1,
  one-vs-rest AUC, confusion matrix.

## E2 — In-domain D2

- 5 classes, 2,569 images.
- Same CV × seeds grid.
- Primary metric: macro-F1.

## E3 — Cross-dataset transfer

- Four harmonised classes (`healthy, yellow_leaf, mosaic, rust`).
- `D1 → D2` and `D2 → D1`. No target fine-tuning, no target-side
  normalisation, no adaptation.
- The checkpoint used is the one selected by validation macro-F1 on the
  source corpus.

## E4 — External Bangladesh test

- `D1 + D2` for training; `D3` healthy versus red rot only at test time.
- The D3 images are natively 224 × 224 and are upsampled to 384 × 384
  bilinearly. The limitation is reported explicitly in Section 6.4.

## E5 — Photometric stress tests

Each test fold of D2 is re-synthesised under the following perturbations
(`scripts/stress_tests.py`):

- **Illuminant sweep.** Planckian illuminants at 2,800, 3,500, 4,300, 5,000,
  5,500, 6,500 and 7,500 K.
- **Exposure sweep.** −1.5 to +1.5 EV in 0.5-EV steps.
- **Added glare.** A specular term scaled by 0, 0.25 and 0.5.
- **JPEG compression.** Quality 30, 50, 70, 90.
- **Downscaling.** 1.0×, 0.75×, 0.5× nearest, then bilinear upsampled to the
  nominal input.

Reports: macro-F1 at each point (Fig. 9) and macro-F1 under the extreme in
each dimension (Table 11).

## Statistical analysis

Primary test: Nadeau–Bengio corrected paired t-test with Holm step-down
over the family of eleven competitors (see `scripts/nadeau_bengio.py`).
Secondary test: Wilcoxon signed-rank over pooled fold-level differences.
Confidence intervals: 10,000-resample percentile bootstrap over the same
fold-level unit.
