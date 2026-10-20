"""Checks of the demo app before deployment, written to results/tables/app_checks.json.

Usage (from the repository root, after python -m src.export_app):
    python -m src.check_app

1. Same model: the app's tumor probability (before temperature) on random val patches must match
   the chosen run's saved predictions. Small gaps are expected: the features were extracted in
   float16 on the GPU, the app computes in float32 on the CPU.
2. Speed on a CPU limited to 2 threads, like a free Hugging Face Space.
3. Heat-map sanity check (Adebayo et al., 2018): with the last transformer block re-initialized at
   random, the map should change. If it did not, it would not reflect what the model learned.
"""

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from src.data import load_dataset
from src.utils import save_json


def load_app(path: Path):
    spec = importlib.util.spec_from_file_location("demo_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def randomized_copy(app, seed: int):
    """A fresh copy of the classifier with the last transformer block re-initialized at random."""
    from transformers import ViTModel

    backbone = app.HEAD["backbone"]
    vit = ViTModel.from_pretrained(backbone["hf_id"], revision=backbone["revision"], add_pooling_layer=False)
    layers = vit.layers if hasattr(vit, "layers") else vit.encoder.layer
    torch.manual_seed(seed)
    for module in layers[-1].modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            torch.nn.init.zeros_(module.bias)
    return app.PhikonClassifier(vit, app.HEAD)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the demo app: same model, speed, heat-map sanity.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--app", type=Path, default=Path("app/app.py"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--n-patches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    app = load_app(args.app)
    card = app.CARD
    run_seed = card["selection"]["chosen_seed"]
    saved = pd.read_parquet(args.artifacts_dir / card["config"] / f"seed{run_seed}" / "predictions" / "val.parquet")
    sample = saved.sample(n=args.n_patches, random_state=args.seed)
    dataset = load_dataset(args.data_root)
    torch.set_num_threads(2)
    classifier = app.load_model()

    probs, maps, seconds = [], [], []
    for index in sample["index"]:
        patch, _ = app.to_patch(dataset.get_input(int(index)))
        start = time.perf_counter()
        z, cam = classifier(app.preprocess(patch))
        seconds.append(time.perf_counter() - start)
        probs.append(float(torch.sigmoid(z[0])))
        maps.append(cam[0].numpy())
    gap = np.abs(np.array(probs) - sample["prob"].to_numpy())
    same_decision = float(np.mean((np.array(probs) > 0.5) == (sample["prob"].to_numpy() > 0.5)))

    random_classifier = randomized_copy(app, args.seed)
    correlations = []
    for index, cam in list(zip(sample["index"], maps))[:50]:
        patch, _ = app.to_patch(dataset.get_input(int(index)))
        _, random_cam = random_classifier(app.preprocess(patch))
        if cam.std() > 0 and random_cam[0].std() > 0:
            correlations.append(spearmanr(cam.ravel(), random_cam[0].numpy().ravel()).statistic)

    checks = {
        "run": f"{card['config']}/seed{run_seed}",
        "same_model": {"n_val_patches": args.n_patches, "max_abs_probability_gap": float(gap.max()),
                       "mean_abs_probability_gap": float(gap.mean()), "same_decision_share": same_decision},
        "speed_cpu_2_threads": {"machine": f"{platform.system()} {platform.machine()}",
                                "seconds_per_image_mean": float(np.mean(seconds[1:])),  # first call warms up
                                "seconds_per_image_max": float(np.max(seconds[1:]))},
        "heat_map_randomization": {"n_patches": len(correlations),
                                   "spearman_original_vs_random_last_block_mean": float(np.mean(correlations)),
                                   "note": "close to 0: the map depends on the learned weights"},
    }
    save_json(checks, Path("results/tables/app_checks.json"))
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
