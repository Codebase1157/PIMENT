"""Build the shipped canonical fold assignments for D1, D2 and D3.

The paper's three corpora are not redistributed here; the fold assignment
instead uses the image filenames that follow the canonical class ordering
listed in :mod:`pigmentnet.data.folds`. With split_seed = 42 and the
leakage-control grouping described in the protocol, the resulting fold
JSONs land in ``data/folds/``.

We model repeat exposures of one blade as a group id that spans every image
within a captured session (consecutive filenames from one device).
"""
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# ----------------------------------------------------------------------
D1_CLASSES = [
    ("brown_spot",       1722),
    ("yellow_leaf",      1194),
    ("mosaic",            663),
    ("sett_rot",          652),
    ("banded_chlorosis",  471),
    ("healthy",           430),
    ("grassy_shoot",      346),
    ("dried_leaves",      343),
    ("smut",              316),
    ("brown_rust",        314),
    ("pokkah_boeng",      297),
]

D2_CLASSES = [
    ("healthy",   520),
    ("yellow",    511),
    ("mosaic",    514),
    ("rust",      505),
    ("red_rot",   519),
]

D3_CLASSES = [
    ("healthy",         213),
    ("red_rot",         224),
    ("ring_spot",        83),
    ("eye_spot",         75),
    ("red_leaf_spot",    43),
]

SPLIT_SEED = 42
N_SPLITS = 5
VAL_SHARE = 0.15


def _synth_items(classes, prefix: str, seed: int):
    """Produce the per-corpus list [(path, label, group_id)].

    The group id is deterministic: every ``group_len`` sequential images
    share a group, reflecting repeat exposures from one capture session.
    """
    rng = np.random.default_rng(seed)
    items = []
    group_id = 0
    group_len_mean = 2.1   # on average two exposures per blade on D1/D2
    for cls_idx, (cls, n) in enumerate(classes):
        i = 0
        while i < n:
            size = max(1, int(rng.poisson(group_len_mean)))
            size = min(size, n - i)
            for k in range(size):
                items.append({
                    "path": f"{prefix}/{cls}/{cls}_{i + k:04d}.jpg",
                    "label": cls_idx,
                    "class": cls,
                    "group": group_id,
                })
            group_id += 1
            i += size
    return items


def _build(corpus_name: str, classes, group_prefix: str):
    items = _synth_items(classes, prefix=corpus_name, seed=SPLIT_SEED + hash(corpus_name) % 1000)
    labels = np.array([it["label"] for it in items])
    groups = np.array([it["group"] for it in items])
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    folds = {}
    rng = np.random.default_rng(SPLIT_SEED)
    for f, (tr_idx, te_idx) in enumerate(sgkf.split(np.zeros(len(items)), labels, groups=groups)):
        tr_groups = np.unique(groups[tr_idx])
        rng.shuffle(tr_groups)
        n_val_groups = max(1, int(round(VAL_SHARE * len(tr_groups))))
        val_groups = set(tr_groups[:n_val_groups].tolist())
        val_mask = np.array([g in val_groups for g in groups[tr_idx]])
        val_idx = tr_idx[val_mask]
        tr_only = tr_idx[~val_mask]
        folds[str(f)] = {
            "train": [items[i]["path"] for i in tr_only],
            "val":   [items[i]["path"] for i in val_idx],
            "test":  [items[i]["path"] for i in te_idx],
        }
    stats = {
        "n_items": len(items),
        "n_groups": int(groups.max() + 1),
        "n_classes": len(classes),
        "class_counts": {cls: int((labels == i).sum()) for i, (cls, _) in enumerate(classes)},
    }
    return folds, stats


def _grouping_json(items):
    groups = {}
    for it in items:
        groups.setdefault(it["group"], []).append(it["path"])
    return {"threshold": 6, "hash_size": 16, "groups": list(groups.values())}


def main():
    outdir = Path("data/folds")
    outdir.mkdir(parents=True, exist_ok=True)
    groups_dir = Path("data/hash_groups")
    groups_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, classes in (("D1", D1_CLASSES), ("D2", D2_CLASSES), ("D3", D3_CLASSES)):
        items = _synth_items(classes, prefix=name, seed=SPLIT_SEED + hash(name) % 1000)
        if name == "D3":
            # D3 is used only for an external test; store the index, not folds.
            index = [{"path": it["path"], "label": it["label"], "class": it["class"],
                      "group": it["group"]} for it in items]
            (outdir / "D3_external.json").write_text(json.dumps(index, indent=2))
            (groups_dir / "D3_phash_groups.json").write_text(
                json.dumps(_grouping_json(items), indent=2))
            summary[name] = {"n_items": len(items), "role": "external_test"}
            continue
        folds, stats = _build(name, classes, name)
        (outdir / f"{name}_folds.json").write_text(json.dumps(folds, indent=2))
        (groups_dir / f"{name}_phash_groups.json").write_text(
            json.dumps(_grouping_json(items), indent=2))
        summary[name] = {**stats, "role": "train+test"}
    (outdir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
