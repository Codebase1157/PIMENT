"""PyTorch dataset over cached decompositions."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from ..physaug import PhysAug
from ..decomposition.pipeline import Decomposition
from ..decomposition.factorisation import leaf_relative_deviation, pigment_composition


class SugarcaneLeafDataset(Dataset):
    """A dataset of pre-decomposed sugarcane leaves.

    Expects ``index.json`` with records ``{path, cache, label, group}`` where
    ``cache`` points to the ``.npz`` written by
    :func:`pigmentnet.data.preprocess.decompose_and_cache`.
    """

    def __init__(self, index_path: str | Path, physaug: PhysAug | None = None,
                 geometric_transform=None, train: bool = True):
        self.index = json.loads(Path(index_path).read_text())
        self.physaug = physaug
        self.geometric_transform = geometric_transform
        self.train = train

    def __len__(self):
        return len(self.index)

    # -----------------------------------------------------------------
    def _load(self, rec):
        d = np.load(rec["cache"])
        diffuse = d["diffuse"].astype(np.float32) / 255.0
        specular = d["specular"].astype(np.float32) / 255.0
        glare = d["glare"].astype(np.float32)
        mask = d["mask"].astype(bool)
        gamma = d["gamma"]
        C = d["C"].astype(np.float32)
        E = d["E"].astype(np.float32)
        od = np.where(diffuse > 1e-3, -np.log10(np.clip(diffuse, 1e-3, None)), 3.0)
        return {
            "diffuse": diffuse, "specular": specular, "glare": glare,
            "mask": mask, "gamma": gamma, "C": C, "E": E, "od": od.astype(np.float32),
        }

    def __getitem__(self, idx):
        rec = self.index[idx]
        d = self._load(rec)

        if self.train and self.physaug is not None:
            dec = Decomposition(
                diffuse=d["diffuse"], specular=d["specular"],
                glare_mask=d["glare"], glare_fraction=float(d["glare"].mean()),
                mask=d["mask"], gamma=d["gamma"], illum_fallback=False,
                C=d["C"], E=d["E"],
                delta_C=leaf_relative_deviation(d["C"], d["mask"]),
                p=pigment_composition(d["C"]),
                od=d["od"],
            )
            img_tilde, _ = self.physaug.apply(dec)
            d["diffuse"] = img_tilde

        delta = leaf_relative_deviation(d["C"], d["mask"]).astype(np.float32)
        p = pigment_composition(d["C"]).astype(np.float32)

        def to_t(x, c=True):
            t = torch.from_numpy(np.ascontiguousarray(x))
            if c and t.dim() == 3 and t.shape[-1] in (1, 3, 4):
                t = t.permute(2, 0, 1)
            return t.float()

        sample = {
            "diffuse": to_t(d["diffuse"]),
            "C": torch.from_numpy(d["C"]).float(),
            "delta_C": torch.from_numpy(delta).float(),
            "p": torch.from_numpy(p).float(),
            "od": to_t(d["od"]),
            "mask": torch.from_numpy(d["mask"]).float()[None],
            "glare_mask": torch.from_numpy(d["glare"]).float()[None],
            "E": torch.from_numpy(d["E"]).float(),
            "label": int(rec["label"]),
            "path": rec["path"],
        }
        return sample
