#!/usr/bin/env python3
"""Estimate the prior absorbance matrix E0 from agronomist-marked pixels
(Section 4.2.5, Eq. 15).

Expects a CSV with columns ``corpus,pigment,image,row,col,R,G,B`` where
RGB values are in [0, 255] and pigment is one of {chl, ylw, red, brn}.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from pigmentnet.decomposition.linearise import linearise_srgb
from pigmentnet.decomposition.factorisation import E0_from_pixels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors", type=Path, required=True,
                    help="CSV of agronomist-marked pixels (see docstring).")
    ap.add_argument("--out", type=Path, default=Path("data/priors"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.anchors)
    rgb = df[["R", "G", "B"]].to_numpy().astype(np.float32) / 255.0
    rgb_lin = linearise_srgb(rgb)
    df["Rl"], df["Gl"], df["Bl"] = rgb_lin[:, 0], rgb_lin[:, 1], rgb_lin[:, 2]

    for corpus in sorted(df["corpus"].unique()):
        sub = df[df["corpus"] == corpus]
        anchors = {}
        for pig in ("chl", "ylw", "red", "brn"):
            px = sub[sub["pigment"] == pig][["Rl", "Gl", "Bl"]].to_numpy()
            if px.size == 0:
                raise RuntimeError(f"No pixels for pigment '{pig}' in corpus '{corpus}'")
            anchors[pig] = px
        E0 = E0_from_pixels(anchors)
        out_path = args.out / f"E0_{corpus}.npy"
        np.save(out_path, E0)
        print(f"{corpus}: wrote {out_path} shape={E0.shape}")
        print(E0)


if __name__ == "__main__":
    main()
