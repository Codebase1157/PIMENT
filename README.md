# PigmentNet

Gloss-referenced pigment deconvolution and a self-derived symptom-concept bottleneck
for interpretable sugarcane leaf disease classification.

> J. Rani and P. Garg. *Gloss-Referenced Pigment Deconvolution and a Self-Derived
> Symptom-Concept Bottleneck for Interpretable Sugarcane Leaf Disease Classification*.

PigmentNet separates a leaf photograph physically before any learning takes place:
dichromatic glare removal, gloss-referenced colour constancy, optical density, and
an anchored sparse factorisation yield four pigment-proxy maps (chlorophyll,
chlorotic yellow, red, necrotic brown). A pigment–texture two-stream network coupled
by pigment-guided attention computes eight agronomically named symptom concepts, and
a concept-first head carries the decision through those named quantities. A
physics-consistent augmentation (PhysAug) redraws illuminant, exposure and device
response from the pigment maps, so the label stays valid where colour jitter would
break it.

## Highlights

- **Training-free photometric decomposition** (`pigmentnet.decomposition`): sRGB
  linearisation, dichromatic glare split, inverse-intensity-chromaticity white point,
  optical density, and anchored sparse factorisation with a prior absorbance matrix
  measured on agronomist-marked pixels.
- **PhysAug** (`pigmentnet.physaug`): re-synthesises training views under new
  illuminants (Planckian locus, 2,800–7,500 K), exposures (±1.5 EV) and perturbed
  device responses without disturbing pigment content.
- **Two-stream network with pigment-guided attention** (`pigmentnet.models`):
  ConvNeXt-T texture stream, U-Net pigment refiner, and attention whose logits
  receive an additive bias proportional to pigment deviation.
- **Label-free symptom-concept layer** (`pigmentnet.concepts`): eight concepts
  (chlorotic, red, necrotic, heterogeneity, patchiness, lesion density, stripe
  coherence, glare flag) computed as deterministic, differentiable functions of the
  refined pigment maps.
- **Concept-first head with capped texture residual** (`pigmentnet.head`).
- **Leakage-controlled evaluation**: perceptual-hash de-duplication, group-aware
  stratified five-fold CV, Nadeau–Bengio corrected paired test with Holm correction,
  cross-dataset and external transfer protocols.

## Reported results (reproduces Tables 6–16 of the paper)

| Protocol                                  | Metric        | PigmentNet |
|-------------------------------------------|---------------|------------|
| In-domain D1 (five-fold, leakage-aware)   | macro-F1 (%)  | 90.7       |
| In-domain D2 (five-fold, leakage-aware)   | macro-F1 (%)  | 96.5       |
| Cross-dataset D1 → D2                     | macro-F1 (%)  | 77.4       |
| Cross-dataset D2 → D1                     | macro-F1 (%)  | 71.8       |
| External D3 (healthy vs red rot)          | F1 (%)        | 88.3       |
| Lesion localisation                       | IoU ≥ 0.5 (%) | 64.2       |
| Severity vs agronomist                    | Spearman ρ    | 0.79       |

## Repository layout

```
pigmentnet/
├── pigmentnet/              # package source
│   ├── decomposition/       # Module A: photometric decomposition
│   ├── models/              # Module B: two-stream + PGA
│   ├── concepts/            # Module C: symptom-concept layer
│   ├── head.py              # Module D: concept-first head
│   ├── physaug.py           # physics-consistent augmentation
│   ├── losses.py            # training objective
│   ├── data/                # datasets, phash, preprocessing
│   └── utils/               # seeds, logging, metrics
├── configs/                 # hyper-parameter YAMLs (reproduces Table 5)
├── data/
│   ├── priors/              # measured absorbance matrices E0
│   ├── folds/               # leakage-controlled fold assignments
│   ├── hash_groups/         # perceptual-hash duplicate groups
│   └── annotations/         # expert boxes, severity, agronomist pixels
├── checkpoints/             # trained model weights
├── figures/                 # paper figures with SHA-256 checksums
├── scripts/                 # preprocessing, training, evaluation, stats
├── docs/                    # annotation guidelines and protocols
└── tests/                   # unit tests for the decomposition
```

## Installation

```bash
git clone https://github.com/Codebase1157/pigmentnet.git
cd pigmentnet
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Core dependencies: PyTorch 2.3, CUDA 12.1, timm 1.0, scikit-image 0.24, scikit-learn
1.5, imagehash 4.3, scipy 1.18.1, PyYAML, Pillow, NumPy, matplotlib.

## Datasets

The three corpora are public and must be downloaded separately:

- **D1** – [Sugarcane Leaf Dataset, Thite et al. 2024, CC BY 4.0](https://doi.org/10.17632/9twjtv923v.1). 6,748 images, 11 classes.
- **D2** – [Sugarcane Leaf Disease Dataset, Daphal & Koli 2022, CC BY 4.0](https://doi.org/10.17632/9424skmnrk.1). 2,569 images, 5 classes.
- **D3** – [SugarcaneLD-BD, CC BY 4.0](https://doi.org/10.17632/n8mpzb7p4k.1). 638 images, used only as an external test.

Arrange them as `raw/D1`, `raw/D2`, `raw/D3` under the project root, then run:

```bash
python scripts/preprocess_datasets.py --root raw --out processed
python scripts/compute_phash.py        --root processed
python scripts/build_folds.py          --root processed --seeds 5 --folds 5
python scripts/estimate_E0.py          --root processed --anchors data/annotations/agronomist_pixels.csv
```

Fold assignments and hash groups shipped in `data/folds/` and `data/hash_groups/`
match those produced by the scripts on the canonical dataset checksums listed in
`data/CHECKSUMS.md`.

## Training

```bash
# in-domain D1
python scripts/train.py --config configs/pigmentnet_D1.yaml --fold 0 --seed 42

# full 5 × 5 grid (seeds × folds)
python scripts/train.py --config configs/pigmentnet_D1.yaml --sweep
```

Hyper-parameters follow Table 5 of the paper: AdamW, backbone LR 1e-4, new-layer LR
5e-4, weight decay 0.05, 5-epoch warm-up + cosine decay over 100 epochs, batch 32,
mixed precision, 384 × 384 input. Random seeds are listed in `configs/seeds.yaml`.

## Evaluation and statistical tests

```bash
# E1/E2 in-domain
python scripts/evaluate.py --config configs/pigmentnet_D1.yaml --split test
python scripts/evaluate.py --config configs/pigmentnet_D2.yaml --split test

# E3 cross-dataset transfer
python scripts/cross_dataset.py --source D1 --target D2
python scripts/cross_dataset.py --source D2 --target D1

# E4 external
python scripts/cross_dataset.py --source D1+D2 --target D3 --binary healthy,red_rot

# E5 photometric stress tests
python scripts/stress_tests.py --config configs/pigmentnet_D2.yaml

# Nadeau–Bengio corrected paired test with Holm step-down
python scripts/nadeau_bengio.py \
    --primary runs/pigmentnet \
    --competitors runs/convnext_t runs/vmamba_t runs/swin_t runs/deit_s \
                  runs/effnetv2_s runs/resnet50 runs/mobilevit_v2 \
                  runs/convnext_clahe runs/convnext_hsv runs/convnext_greyworld \
                  runs/convnext_jitter
```

## Model checkpoints

`checkpoints/` contains one file per (dataset × fold × seed) PigmentNet run, plus
the matched controls used for Tables 10 and 12. File names follow
`{method}_{dataset}_fold{f}_seed{s}.pt`. SHA-256 checksums are in
`checkpoints/sha256sums.txt`.

## Figures

Every figure in the paper is in `figures/` with a matching SHA-256 line in
`figures/sha256sums.txt`. The generating notebooks are in `notebooks/`.

## Expert annotation guidelines

The expert concept-rating sheet, the lesion-box drawing protocol, the severity 0–5
rubric, and the agronomist-pixel sampling procedure used to build E0 are in
`docs/annotation_guidelines.md`.

## Citation

```bibtex
@article{rani2026pigmentnet,
  title   = {Gloss-Referenced Pigment Deconvolution and a Self-Derived
             Symptom-Concept Bottleneck for Interpretable Sugarcane Leaf
             Disease Classification},
  author  = {Rani, Jaya and Garg, Prachi},
  year    = {2026}
}
```

## License

Code is released under the MIT license (see `LICENSE`). The three sugarcane corpora
retain their original CC BY 4.0 licenses and are not redistributed here.
