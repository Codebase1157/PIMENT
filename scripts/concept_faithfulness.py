#!/usr/bin/env python3
"""Concept-faithfulness tests (Section 6.7, Table 14b).

For each diseased prediction, reset the dominant concept (largest ratio to
its healthy median) to the healthy median and re-evaluate the head. Reports
the share of predictions flipped to healthy, together with the three
baselines of Table 14b.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from pigmentnet.models import build_model


@torch.no_grad()
def faithfulness(model, concepts, residuals, healthy_median,
                 real_healthy_vectors, dominant_only: bool = True):
    device = next(model.parameters()).device
    c = torch.as_tensor(concepts, dtype=torch.float32, device=device)
    r = torch.as_tensor(residuals, dtype=torch.float32, device=device)
    full, _, _, _ = model.head(c, r)
    pred = full.argmax(dim=-1)
    diseased = pred != 0  # healthy == class 0 by convention

    results = {"n": int(diseased.sum()), "flip_median": 0, "flip_real_healthy": 0,
               "flip_random_nondom": 0, "flip_residual_zero": 0,
               "inside_2sd_ellipse": 0}
    if results["n"] == 0:
        return results

    med = torch.as_tensor(healthy_median, dtype=torch.float32, device=device)
    real = torch.as_tensor(real_healthy_vectors, dtype=torch.float32, device=device)

    for mode in ("median", "real_healthy", "random_nondom", "residual_zero"):
        c2 = c.clone()
        r2 = r.clone()
        if mode == "median":
            ratios = c2 / (med + 1e-6)
            dom = ratios.argmax(dim=-1)
            c2[diseased, dom[diseased]] = med[dom[diseased]]
        elif mode == "real_healthy":
            ratios = c2 / (med + 1e-6)
            dom = ratios.argmax(dim=-1)
            samp = real[torch.randint(0, real.shape[0], (c2.shape[0],))]
            c2[diseased, dom[diseased]] = samp[diseased, dom[diseased]]
        elif mode == "random_nondom":
            ratios = c2 / (med + 1e-6)
            dom = ratios.argmax(dim=-1)
            choices = torch.randint(0, c2.shape[-1], (c2.shape[0],))
            choices[choices == dom] = (choices[choices == dom] + 1) % c2.shape[-1]
            c2[diseased, choices[diseased]] = med[choices[diseased]]
        elif mode == "residual_zero":
            r2 = torch.zeros_like(r2)

        full2, _, _, _ = model.head(c2, r2)
        pred2 = full2.argmax(dim=-1)
        flipped = int(((pred == pred2) == False)[diseased].logical_and_(pred2[diseased] == 0).sum())
        results[f"flip_{'median' if mode=='median' else mode}"] = flipped
        if mode == "median":
            # check reset vector falls within the 2-SD healthy ellipse
            sd = real.std(dim=0) + 1e-6
            z = (c2[diseased] - real.mean(dim=0)) / sd
            inside = (z.pow(2).sum(dim=-1) <= 4 * c2.shape[-1]).sum()
            results["inside_2sd_ellipse"] = int(inside)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--concepts", required=True, type=Path,
                    help="NumPy .npz with keys ``concepts``, ``residuals``, "
                         "``healthy_median``, ``real_healthy``.")
    ap.add_argument("--num-classes", type=int, default=5)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    model = build_model("pigmentnet", num_classes=args.num_classes)
    model.load_state_dict(ck["state_dict"], strict=False)
    model.eval()

    data = np.load(args.concepts)
    results = faithfulness(
        model,
        data["concepts"], data["residuals"],
        data["healthy_median"], data["real_healthy"],
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
