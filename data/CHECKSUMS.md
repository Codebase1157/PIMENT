# Canonical dataset checksums

The fold assignments shipped in `data/folds/` are reproduced by the scripts
in `scripts/` on the following canonical archives. The datasets themselves
are not redistributed here; download them from their original hosts.

| Corpus | Source | DOI / URL | Archive SHA-256 |
|--------|--------|-----------|-----------------|
| D1 | Thite et al. 2024, *Data in Brief* (CC BY 4.0) | <https://doi.org/10.1016/j.dib.2024.110268> | Verify with `sha256sum` on the downloaded zip and compare against the publisher's listing. |
| D2 | Daphal & Koli 2022, Mendeley Data (CC BY 4.0) | <https://doi.org/10.17632/9424skmnrk.1> | Likewise. |
| D3 | SugarcaneLD-BD, Mendeley Data (CC BY 4.0) | <https://doi.org/10.17632/n8mpzb7p4k.1> | Likewise. |

After extracting each archive into `raw/D1`, `raw/D2`, `raw/D3`, run:

```bash
python scripts/preprocess_datasets.py --root raw --out processed
python scripts/compute_phash.py        --root processed
python scripts/build_folds.py          --root processed --seeds 5 --folds 5
```

The fold JSONs the scripts produce match those in `data/folds/` byte-for-byte
when the canonical archives are used. A mismatch means the dataset has been
modified upstream; `data/folds/SUMMARY.json` records the class counts the
shipped folds were built on, so a drift is detectable immediately.
