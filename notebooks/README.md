# Notebooks

Each `fig*.ipynb` regenerates one figure in `../figures/`. They read the
fold-level metrics from `../results/fold_level_metrics.json`, the priors
from `../data/priors/` and the shipped annotations from
`../data/annotations/`. A one-shot rebuild of every figure is just:

```bash
python ../scripts/_build_figures.py
sha256sum ../figures/*.png > ../figures/sha256sums.txt
```

which bypasses the notebooks and writes the figures directly. Use the
notebooks when you want to tune a particular panel; use the script when
you want byte-reproducible outputs.
