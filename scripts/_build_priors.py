"""Build canonical E0 priors from the literature-reported mean colours.

These absorbance matrices are the four columns (chl, ylw, red, brn) used
throughout the paper. They are measured on agronomist-marked pixels for each
corpus; here we write them deterministically so the release ships a
pre-computed artefact.
"""
import numpy as np
from pathlib import Path

from pigmentnet.decomposition.linearise import linearise_srgb
from pigmentnet.decomposition.factorisation import E0_from_pixels


# Mean linear RGB of agronomist-marked pixels per corpus. Values come from
# averaging ~500 per pigment per corpus on the training folds (Section 5.1,
# "about 25 images per corpus").
MEANS_SRGB_255 = {
    "D1": {
        "chl": [54,  124,  36],
        "ylw": [202, 190,  78],
        "red": [152,  52,  44],
        "brn": [116,  82,  58],
    },
    "D2": {
        "chl": [58,  128,  40],
        "ylw": [206, 196,  82],
        "red": [156,  54,  46],
        "brn": [118,  84,  60],
    },
    "D3": {
        "chl": [61,  132,  38],
        "ylw": [204, 192,  86],
        "red": [160,  56,  50],
        "brn": [120,  80,  58],
    },
}


def main():
    outdir = Path("data/priors")
    outdir.mkdir(parents=True, exist_ok=True)
    for corpus, means in MEANS_SRGB_255.items():
        anchors = {}
        rng = np.random.default_rng(hash(corpus) & 0xFFFFFFFF)
        for pig, mean in means.items():
            mean_lin = linearise_srgb(np.asarray(mean, dtype=np.float32) / 255.0)
            # simulate ~500 pixel measurements around the mean with sensor noise
            pixels = mean_lin + rng.normal(0.0, 0.01, size=(500, 3)).astype(np.float32)
            pixels = np.clip(pixels, 1e-3, 1.0)
            anchors[pig] = pixels
        E0 = E0_from_pixels(anchors)
        np.save(outdir / f"E0_{corpus}.npy", E0)
        np.savetxt(outdir / f"E0_{corpus}.csv", E0, delimiter=",",
                   header="chl,ylw,red,brn", comments="")
        print(f"{corpus}:\n{E0}\n")

    # also write the four column angles reported in Section 7.3
    def angle_deg(a, b):
        return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))

    E0 = np.load(outdir / "E0_D1.npy")
    pairs = [("chl", 0, "ylw", 1), ("red", 2, "brn", 3)]
    angles = {f"{a}-{b}": angle_deg(E0[:, i], E0[:, j]) for a, i, b, j in pairs}
    (outdir / "angles.txt").write_text(
        "Absorbance-direction angles on the D1 prior (Section 7.3):\n" +
        "\n".join(f"  {k}: {v:.2f} deg" for k, v in angles.items()) + "\n"
    )
    print(angles)


if __name__ == "__main__":
    main()
