"""Export the in-browser demo (web/): CNN + HED as ONNX with its Grad-CAM, settings, examples.

Usage (from the repository root):
    python -m src.export_web_demo

The free Hugging Face Spaces only serve static pages, so the model runs in the visitor's browser
(ONNX Runtime Web). Model: CNN + HED stain augmentation, the best CNN on val; its seed is chosen by
the same rule as the local Phikon app (highest val accuracy). The test hospital is only read
afterwards, to report this run's results and draw one error example.

Grad-CAM needs no autograd here: DenseNet ends with ReLU, global average pooling and a linear layer,
so the gradient of the decision z = logit(tumor) - logit(normal) with respect to the post-ReLU
activations A is (w_tumor - w_normal) / (h * w) at every cell. Grad-CAM then equals CAM (Zhou et al.,
2016): relu(sum_k (w_tumor - w_normal)_k A_k) / (h * w). The exported graph computes it and returns
(z, 3 x 3 map); tests/test_web_demo.py checks it against the autograd Grad-CAM of src/explain.py.

Writes:
    web/model.onnx       input 1 x 3 x 96 x 96 (ImageNet-normalized), outputs z and the 3 x 3 map
    web/settings.json    temperature, referral cutoff, this run's results, examples (all from results/)
    web/examples/*.png   one tumor and one normal patch per hospital, plus one patch the model refers
                         and one error on the test hospital, all drawn at random (seed 0)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.abstention import apply_cutoff, cutoff_for_coverage
from src.calibration import to_logit
from src.data import load_dataset
from src.export_app import COVERAGE, HOSPITAL_ROLES, draw_examples
from src.models import build_model
from src.train import code_commit
from src.utils import save_json

CONFIG = "cnn_hed"
PATCH = 96


class DecisionWithCam(nn.Module):
    """DenseNet-121 returning the decision z and its exact Grad-CAM on the last block."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.features = model.features
        self.classifier = model.classifier

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.features(x)                         # b x 1024 x 3 x 3, before the final ReLU
        activations = torch.relu(feature_map)
        logits = self.classifier(activations.mean(dim=(2, 3)))
        z = logits[:, 1] - logits[:, 0]
        # dz / dA for the post-ReLU activations A (the tensor src/explain.py hooks, since torchvision
        # applies that ReLU in place): the same at every cell, so it is also the Grad-CAM weight.
        cells = feature_map.shape[2] * feature_map.shape[3]
        weights = (self.classifier.weight[1] - self.classifier.weight[0])[None, :, None, None] / cells
        cam = torch.relu((weights * activations).sum(dim=1))
        return z, cam


def select_seed(results_runs: Path) -> tuple[int, dict]:
    accuracy = {int(p.parent.name.removeprefix("seed")): json.loads(p.read_text())["splits"]["val"]["official_accuracy"]
                for p in sorted((results_runs / CONFIG).glob("seed*/metrics.json"))}
    return max(accuracy, key=accuracy.get), accuracy


def load_wrapped(artifacts_dir: Path, seed: int) -> DecisionWithCam:
    checkpoint = torch.load(artifacts_dir / CONFIG / f"seed{seed}" / "best.pt", map_location="cpu")
    model = build_model(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"])
    return DecisionWithCam(model).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the in-browser demo into web/.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/runs"), help="Weights (best.pt)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--web-dir", type=Path, default=Path("web"))
    args = parser.parse_args()

    runs = args.results_dir / "runs"
    seed, val_accuracy = select_seed(runs)
    run = runs / CONFIG / f"seed{seed}"
    wrapped = load_wrapped(args.artifacts_dir, seed)
    torch.onnx.export(wrapped, (torch.zeros(1, 3, PATCH, PATCH),), args.web_dir / "model.onnx",
                      input_names=["image"], output_names=["z", "cam"], opset_version=18, dynamo=True,
                      external_data=False)

    metrics = json.loads((run / "metrics.json").read_text())
    calibration = json.loads((run / "calibration.json").read_text())
    val = pd.read_parquet(run / "predictions" / "val.parquet")
    test = pd.read_parquet(run / "predictions" / "test.parquet")
    confidence_cutoff = cutoff_for_coverage(val.prob, COVERAGE)
    referral = {}
    for split, frame in (("val", val), ("test", test)):
        answered, error = apply_cutoff(frame.prob, frame.y, confidence_cutoff)
        referral[split] = {"answered": answered, "error_when_answered": error,
                           "error_without_referral": float(((frame.prob > 0.5) != (frame.y == 1)).mean())}
    bootstrap = pd.read_csv(args.results_dir / "tables" / "bootstrap.csv")
    ci = bootstrap[(bootstrap.config == CONFIG) & (bootstrap.split == "test")
                   & (bootstrap.metric == "accuracy")].iloc[0]

    # Examples: the 10 random patches of the Phikon app, plus one referred patch (val) and one error (test).
    dataset = load_dataset(args.data_root)
    examples = draw_examples(dataset)
    rng = np.random.default_rng(0)
    abs_logit_cutoff = float(to_logit(0.5 + confidence_cutoff)[()])
    referred = val[np.abs(to_logit(val.prob)) < abs_logit_cutoff]
    errors = test[(test.prob > 0.5) != (test.y == 1)]
    for kind, pool, split, hospital in (("referred", referred, "val", 1), ("error", errors, "test", 2)):
        row = pool.iloc[int(rng.integers(len(pool)))]
        examples.append({"index": int(row["index"]), "hospital": hospital, "split": split,
                         "role": HOSPITAL_ROLES[hospital],
                         "label": "tumor" if row.y == 1 else "normal", "slide": int(row.slide), "kind": kind,
                         "file": f"hospital{hospital}_{kind}.png"})
    (args.web_dir / "examples").mkdir(parents=True, exist_ok=True)
    for example in examples:
        dataset.get_input(example["index"]).save(args.web_dir / "examples" / example["file"])

    settings = {
        "model": {"config": CONFIG, "description": "DenseNet-121 trained from scratch with HED stain augmentation",
                  "chosen_seed": seed, "val_accuracy_by_seed": val_accuracy,
                  "selected_epoch": metrics["selected_epoch"],
                  "rule": "cnn_hed: best CNN on val accuracy (mean of 4 seeds); seed: highest val accuracy. "
                          "Test never used to choose."},
        "input": {"size": PATCH, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "temperature": calibration["temperature"],
        "referral_abs_logit": abs_logit_cutoff,
        "referral": referral,
        "this_run": {split: {k: metrics["splits"][split][k] for k in ("accuracy", "auc")}
                     for split in ("id_val", "val", "test")},
        "method_test_accuracy_mean_over_seeds": ci.estimate,
        "method_test_accuracy_ci95_over_patients": [ci.ci_low, ci.ci_high],
        "ece_test_before_after": [calibration["splits"]["test"]["all"]["ece_raw"],
                                  calibration["splits"]["test"]["all"]["ece_scaled"]],
        "examples": examples,
        "code_commit": code_commit(),
    }
    save_json(settings, args.web_dir / "settings.json")
    shutil.copy(Path("LICENSE"), args.web_dir / "LICENSE")
    print(f"Seed {seed} chosen on val ({', '.join(f'seed{s} {a:.4f}' for s, a in val_accuracy.items())})")
    print(f"-> {args.web_dir}/model.onnx, settings.json, {len(examples)} examples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
