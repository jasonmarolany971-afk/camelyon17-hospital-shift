"""Camelyon17-WILDS loading: official splits, image transforms and data loaders."""

import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.wilds_dataset import WILDSSubset

from src.utils import default_data_root, seed_worker

# Harmless warning raised inside wilds with pandas >= 3 (read-only array, never written).
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

# ImageNet statistics, as in the WILDS reference code ("image_base" transform).
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
COLOR_METHODS = ["none", "hed", "macenko"]


def load_dataset(data_root=None) -> Camelyon17Dataset:
    return Camelyon17Dataset(root_dir=str(data_root or default_data_root()), download=False)


def build_transform(config: dict, train: bool):
    """Color step (if any), then the WILDS "image_base" transform: tensor in [0, 1], ImageNet normalization.

    - "none": no color step (WILDS reference protocol; WILDS also resizes to 96x96, a no-op here).
    - "hed": random HED jitter on training patches only. Evaluation patches are never
      augmented, otherwise the val accuracy used to select the epoch would be random.
    - "macenko": Macenko normalization on every patch, training and evaluation alike,
      towards the reference fitted on train patches (results/tables/macenko_reference.json).
    """
    from src.color import HEDJitter, MacenkoNormalizer

    color = config.get("color", "none")
    if color not in COLOR_METHODS:
        raise ValueError(f"Unknown color method {color!r}, expected one of {COLOR_METHODS}")
    steps = []
    if color == "hed" and train:
        steps.append(HEDJitter(sigma=config["hed_sigma"]))
    elif color == "macenko":
        steps.append(MacenkoNormalizer.from_file(config["macenko_reference"]))
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def get_subset(dataset, split: str, transform, max_size: int | None = None, seed: int = 0) -> WILDSSubset:
    """Patches of one official split, optionally a fixed random sample (quick tests)."""
    indices = np.flatnonzero(dataset.split_array == dataset.split_dict[split])
    if max_size is not None and max_size < len(indices):
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_size, replace=False))
    return WILDSSubset(dataset, indices, transform)


def make_loader(subset, batch_size: int, shuffle: bool, num_workers: int, generator=None) -> DataLoader:
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=seed_worker,
        pin_memory=torch.cuda.is_available(),
    )
