"""Stain color methods for H&E patches: HED augmentation and Macenko normalization.

Both work in optical density (OD): a dye absorbs light and, by the Beer-Lambert
law, OD = -log(I / I0) is proportional to the amount of dye. The OD of a pixel
is a mix of a hematoxylin (H) and an eosin (E) color vector.

- HEDJitter, training only (Tellez et al., 2019): random and independent change
  of the amount of each stain.
- MacenkoNormalizer, training and inference (Macenko et al., 2009): estimate the
  two stain vectors of the patch, then re-express it with reference vectors
  computed on the training set only.

Usage (from the repository root):
    python -m src.color reference   # fit the Macenko reference on train patches
    python -m src.color gallery     # visual check on train and val patches, fallback rates
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.color import hed_from_rgb, rgb_from_hed

EPS = 1e-6
# Tissue pixel: total OD (Euclidean norm) >= 0.15. Common Macenko implementations require
# OD >= 0.15 in every channel instead, which drops pure-eosin pixels (eosin absorbs little
# red light: a pink stroma pixel has a red OD near 0.1) and biases the eosin vector.
OD_THRESHOLD = 0.15
REFERENCE_PATH = Path("results/tables/macenko_reference.json")


def rgb_to_od(rgb: np.ndarray) -> np.ndarray:
    """RGB in [0, 1] -> optical density (natural log; 0 for white)."""
    return -np.log(np.clip(rgb, EPS, 1.0))


def od_to_rgb(od: np.ndarray) -> np.ndarray:
    return np.clip(np.exp(-od), 0.0, 1.0)


def to_image(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((rgb * 255).round().astype(np.uint8))


class HEDJitter:
    """Random scale and shift of the H, E and D concentrations (Tellez et al., 2019).

    For each channel c: C_c -> alpha_c * C_c + beta_c, alpha ~ U(1 - sigma, 1 + sigma),
    beta ~ U(-sigma, sigma). Concentrations are in OD units, where sigma = 0.05
    ("HED-light") changes intensities by a few percent. scikit-image's rgb2hed divides
    OD by -log(1e-6) = 13.8: the same sigma there would be about 14 times stronger.
    Random numbers come from torch, which the DataLoader seeds in every worker.
    """

    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        concentrations = rgb_to_od(rgb) @ hed_from_rgb
        alpha = 1 + (2 * torch.rand(3).numpy() - 1) * self.sigma
        beta = (2 * torch.rand(3).numpy() - 1) * self.sigma
        return to_image(od_to_rgb((concentrations * alpha + beta) @ rgb_from_hed))


def estimate_stains(od_tissue: np.ndarray, angle_percentile: float = 1.0) -> np.ndarray:
    """The two stain vectors (rows: hematoxylin, eosin) of tissue OD pixels (Macenko et al., 2009).

    1. Plane of the two main directions of the OD cloud (principal components).
    2. Angle of every pixel in that plane; the two extreme angles (robust 1st and 99th
       percentiles) are the two pure stains.
    """
    _, eigenvectors = np.linalg.eigh(np.cov(od_tissue.T))  # eigenvalues in ascending order
    plane = eigenvectors[:, [2, 1]]
    # Eigenvector signs are arbitrary: make the main one point towards positive OD,
    # so that pixel angles stay in (-pi/2, pi/2) and never wrap around.
    plane[:, 0] *= np.sign(plane[:, 0].sum()) or 1.0
    projected = od_tissue @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    low, high = np.percentile(angles, [angle_percentile, 100 - angle_percentile])
    first = plane @ np.array([np.cos(low), np.sin(low)])
    second = plane @ np.array([np.cos(high), np.sin(high)])
    # Hematoxylin (blue-purple) absorbs more red light than eosin (pink): it comes first.
    stains = np.array([first, second]) if first[0] > second[0] else np.array([second, first])
    return stains / np.linalg.norm(stains, axis=1, keepdims=True)


def stain_concentrations(od: np.ndarray, stains: np.ndarray) -> np.ndarray:
    """Least-squares amounts of each stain for every pixel: od ~= concentrations @ stains."""
    return np.linalg.lstsq(stains.T, od.T, rcond=None)[0].T


def tissue_od(image: Image.Image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    od = rgb_to_od(rgb).reshape(-1, 3)
    return rgb, od, np.linalg.norm(od, axis=1) >= OD_THRESHOLD


class MacenkoNormalizer:
    """Per-patch Macenko normalization towards reference stains fitted on train patches.

    Falls back to the original image when the patch has too little tissue
    (< min_tissue_fraction of pixels) or when the estimation fails.
    """

    def __init__(self, stains: np.ndarray, max_concentrations: np.ndarray, min_tissue_fraction: float = 0.05):
        self.stains = np.asarray(stains, dtype=np.float64)
        self.max_concentrations = np.asarray(max_concentrations, dtype=np.float64)
        self.min_tissue_fraction = min_tissue_fraction

    @classmethod
    def from_file(cls, path: Path = REFERENCE_PATH) -> "MacenkoNormalizer":
        reference = json.loads(Path(path).read_text())
        return cls(reference["stains"], reference["max_concentrations"], reference["min_tissue_fraction"])

    def normalize(self, image: Image.Image) -> tuple[Image.Image, bool]:
        """Normalized image, and whether normalization succeeded (False: original returned)."""
        rgb, od, tissue = tissue_od(image)
        if tissue.mean() < self.min_tissue_fraction:
            return image, False
        try:
            stains = estimate_stains(od[tissue])
            concentrations = stain_concentrations(od, stains)
        except np.linalg.LinAlgError:
            return image, False
        max_concentrations = np.percentile(concentrations[tissue], 99, axis=0)
        if not np.all(np.isfinite(max_concentrations)) or np.any(max_concentrations <= 0):
            return image, False
        concentrations *= self.max_concentrations / max_concentrations
        return to_image(od_to_rgb(concentrations @ self.stains).reshape(rgb.shape)), True

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.normalize(image)[0]


def fit_reference(images, min_tissue_fraction: float = 0.05) -> dict:
    """Median stain vectors and 99th-percentile concentrations over training patches only."""
    stains, maxima = [], []
    for image in images:
        _, od, tissue = tissue_od(image)
        if tissue.mean() < min_tissue_fraction:
            continue
        try:
            patch_stains = estimate_stains(od[tissue])
        except np.linalg.LinAlgError:
            continue
        stains.append(patch_stains)
        maxima.append(np.percentile(stain_concentrations(od[tissue], patch_stains), 99, axis=0))
    median_stains = np.median(np.array(stains), axis=0)
    median_stains /= np.linalg.norm(median_stains, axis=1, keepdims=True)
    return {"stains": median_stains.tolist(), "max_concentrations": np.median(np.array(maxima), axis=0).tolist(),
            "min_tissue_fraction": min_tissue_fraction, "n_patches_used": len(stains)}


def sample_indices(dataset, split: str, n: int, seed: int, hospitals=None) -> list[int]:
    """Random patches of a split, balanced across its hospitals."""
    split_mask = dataset.split_array == dataset.split_dict[split]
    hospital_array = dataset.metadata_array[:, 0].numpy()
    hospitals = hospitals or sorted(set(hospital_array[split_mask]))
    rng = np.random.default_rng(seed)
    per_hospital = n // len(hospitals)
    return [int(i) for h in hospitals
            for i in rng.choice(np.flatnonzero(split_mask & (hospital_array == h)), per_hospital, replace=False)]


def make_reference(args) -> None:
    from src.data import load_dataset
    from src.utils import save_json

    dataset = load_dataset(args.data_root)
    indices = sample_indices(dataset, "train", args.n_patches, args.seed)
    reference = fit_reference(dataset.get_input(i) for i in indices)
    reference.update({"fitted_on": f"{len(indices)} random train patches, balanced across hospitals 0, 3, 4",
                      "seed": args.seed})
    save_json(reference, REFERENCE_PATH)
    print(f"Reference from {reference['n_patches_used']} patches -> {REFERENCE_PATH}")
    print("stains (rows H, E; columns R, G, B OD):", np.round(reference["stains"], 3).tolist())


def make_gallery(args) -> None:
    from src.data import load_dataset
    from src.plotting import INK, apply_style, plt
    from src.utils import save_json

    dataset = load_dataset(args.data_root)
    normalizer = MacenkoNormalizer.from_file()
    apply_style()
    hospital_array = dataset.metadata_array[:, 0].numpy()

    # 1. HED-light: one train patch per row, original then five random draws.
    torch.manual_seed(args.seed)
    jitter = HEDJitter(sigma=args.sigma)
    rows = sample_indices(dataset, "train", 6, args.seed)
    fig, axes = plt.subplots(len(rows), 6, figsize=(9, 9.3), gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    for r, index in enumerate(rows):
        image = dataset.get_input(index)
        for c in range(6):
            axes[r, c].imshow(image if c == 0 else jitter(image))
        axes[r, 0].set_ylabel(f"H{hospital_array[index]} (train)", rotation=0, ha="right", va="center", color=INK)
    axes[0, 0].set_title("original")
    axes[0, 1].set_title(f"HED jitter, sigma = {args.sigma} (five random draws)", loc="left")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.savefig("results/figures/hed_light_examples.png")
    plt.close(fig)

    # 2. Macenko: original / normalized pairs, three patches per train or val hospital (never test).
    hospitals = [(0, "train"), (3, "train"), (4, "train"), (1, "val")]
    fig, axes = plt.subplots(len(hospitals), 6, figsize=(9, 6.3), gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    for r, (hospital, split) in enumerate(hospitals):
        for k, index in enumerate(sample_indices(dataset, split, 3, args.seed + 1, hospitals=[hospital])):
            image = dataset.get_input(index)
            normalized, ok = normalizer.normalize(image)
            axes[r, 2 * k].imshow(image)
            axes[r, 2 * k + 1].imshow(normalized)
            if not ok:
                axes[r, 2 * k + 1].set_xlabel("fallback", color=INK, fontsize=8)
        axes[r, 0].set_ylabel(f"H{hospital} ({split})", rotation=0, ha="right", va="center", color=INK)
    for k in range(3):
        axes[0, 2 * k].set_title("original")
        axes[0, 2 * k + 1].set_title("Macenko")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.savefig("results/figures/macenko_examples.png")
    plt.close(fig)

    # 3. How often per-patch Macenko falls back to the original image (train and val only).
    rates = {}
    for split in ("train", "val"):
        indices = sample_indices(dataset, split, args.n_patches, args.seed + 2)
        fallbacks = sum(not normalizer.normalize(dataset.get_input(i))[1] for i in indices)
        rates[split] = {"n": len(indices), "fallbacks": fallbacks, "rate": fallbacks / len(indices)}
        print(f"Macenko fallback on {split}: {fallbacks}/{len(indices)} patches ({fallbacks / len(indices):.1%})")
    save_json(rates, "results/tables/macenko_fallback.json")
    print("-> results/figures/hed_light_examples.png, results/figures/macenko_examples.png, "
          "results/tables/macenko_fallback.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit the Macenko reference, or draw the visual checks.")
    parser.add_argument("command", choices=["reference", "gallery"])
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--n-patches", type=int, default=999, help="Patches sampled (a multiple of 3 hospitals).")
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    make_reference(args) if args.command == "reference" else make_gallery(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
