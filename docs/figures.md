# Paper figures

Every figure in `figures/` is paired with a SHA-256 entry in
`figures/sha256sums.txt`. The notebooks that produced them are in
`notebooks/`.

| File                              | Figure in paper | Source notebook |
|-----------------------------------|-----------------|-----------------|
| `fig01_framework.png`             | Fig. 1 — Four-module framework overview | `notebooks/fig01_framework.ipynb` |
| `fig02_architecture.png`          | Fig. 2 — Two-stream network with PGA | `notebooks/fig02_architecture.ipynb` |
| `fig03_class_distribution.png`    | Fig. 3 — Class counts across D1/D2/D3 | `notebooks/fig03_distribution.ipynb` |
| `fig04_class_exemplars.png`       | Fig. 4 — Representative leaves per class | `notebooks/fig04_exemplars.ipynb` |
| `fig05_confusion_D1_D2.png`       | Fig. 5 — Row-normalised confusions | `notebooks/fig05_confusion.ipynb` |
| `fig06_roc.png`                   | Fig. 6 — Per-class OvR ROC curves | `notebooks/fig06_roc.ipynb` |
| `fig07_training_curves.png`       | Fig. 7 — Validation macro-F1 over epochs | `notebooks/fig07_curves.ipynb` |
| `fig08_class_transfer.png`        | Fig. 8 — Per-class transfer & concept agreement | `notebooks/fig08_transfer.ipynb` |
| `fig09_photometric_robustness.png`| Fig. 9 — Illuminant & exposure sweeps | `notebooks/fig09_stress.ipynb` |
| `fig10_ablation.png`              | Fig. 10 — Ablation deltas | `notebooks/fig10_ablation.ipynb` |
| `fig11_decomposition.png`         | Fig. 11 — Decomposition of held-out leaves | `notebooks/fig11_decomposition.ipynb` |
| `fig12_concept_cards.png`         | Fig. 12 — Localisation and concept explanations | `notebooks/fig12_concepts.ipynb` |
| `fig13_predictions.png`           | Fig. 13 — Held-out predictions with drivers | `notebooks/fig13_predictions.ipynb` |
| `fig14_cross_dataset_evidence.png`| Fig. 14 — Evidence maps after D1 → D2 transfer | `notebooks/fig14_evidence.ipynb` |
| `fig15_posthoc_explanations.png`  | Fig. 15 — Post-hoc explainers vs pigment deviation | `notebooks/fig15_posthoc.ipynb` |
| `fig16_tsne.png`                  | Fig. 16 — t-SNE of RGB, features, concepts | `notebooks/fig16_tsne.ipynb` |

Regenerate all figures with:

```bash
for nb in notebooks/fig*.ipynb; do
    jupyter nbconvert --to notebook --execute "$nb" --inplace
done
python scripts/export_figures.py --notebooks-dir notebooks --out-dir figures
sha256sum figures/*.png > figures/sha256sums.txt
```
