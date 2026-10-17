"""Small helpers shared by the training, evaluation and benchmark scripts."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def default_data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", "~/datasets")).expanduser()


def set_seed(seed: int) -> None:
    """Seed every random number generator used by the project."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU, CUDA and MPS


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own NumPy and `random` seed.

    PyTorch seeds its own generator in every worker, but not NumPy's: without
    this, workers could draw identical "random" augmentations.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work to finish (required for correct timings)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(data, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
