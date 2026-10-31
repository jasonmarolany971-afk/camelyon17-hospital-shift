"""Checks of the in-browser demo before deployment, written to results/tables/web_demo_checks.json.

Usage (from the repository root, after python -m src.export_web_demo):
    python -m src.check_web_demo

1. Same model: ONNX Runtime on random val patches must reproduce the chosen run's saved predictions.
2. The 12 examples: decision z and referral for each, to compare with what the web page shows.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import pandas as pd
from PIL import Image

from src.data import load_dataset
from src.utils import save_json


def preprocess(image: Image.Image, settings: dict) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    array = (array - np.array(settings["input"]["mean"], dtype=np.float32)) / np.array(settings["input"]["std"],
                                                                                       dtype=np.float32)
    return array.transpose(2, 0, 1)[None]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the in-browser demo: same model, examples.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--web-dir", type=Path, default=Path("web"))
    parser.add_argument("--n-patches", type=int, default=200)
    args = parser.parse_args()

    settings = json.loads((args.web_dir / "settings.json").read_text())
    session = onnxruntime.InferenceSession(str(args.web_dir / "model.onnx"))
    model = settings["model"]
    saved = pd.read_parquet(Path("results/runs") / model["config"] / f"seed{model['chosen_seed']}" / "predictions"
                            / "val.parquet").sample(n=args.n_patches, random_state=0)
    dataset = load_dataset(args.data_root)
    z = np.array([session.run(None, {"image": preprocess(dataset.get_input(int(i)), settings)})[0][0]
                  for i in saved["index"]])
    prob = 1 / (1 + np.exp(-z))
    gap = np.abs(prob - saved["prob"].to_numpy())

    examples = []
    for example in settings["examples"]:
        image = Image.open(args.web_dir / "examples" / example["file"])
        ez, cam = session.run(None, {"image": preprocess(image, settings)})
        examples.append({"file": example["file"], "label": example["label"], "z": float(ez[0]),
                         "referred": bool(abs(ez[0]) < settings["referral_abs_logit"]),
                         "predicted": "tumor" if ez[0] > 0 else "normal", "cam_max": float(cam.max())})
    checks = {"run": f"{model['config']}/seed{model['chosen_seed']}",
              "same_model": {"n_val_patches": args.n_patches, "max_abs_probability_gap": float(gap.max()),
                             "same_decision_share": float(np.mean((prob > 0.5) == (saved["prob"].to_numpy() > 0.5)))},
              "examples": examples}
    save_json(checks, Path("results/tables/web_demo_checks.json"))
    print(json.dumps(checks["same_model"]))
    for e in examples:
        print(f"{e['file']:<24} label {e['label']:<6} -> {e['predicted']:<6} z {e['z']:+.3f}"
              f"{'  referred' if e['referred'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
