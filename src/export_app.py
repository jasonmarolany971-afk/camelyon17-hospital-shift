"""Export everything the demo app needs, from files produced by the earlier phases.

Usage (from the repository root):
    python -m src.export_app

Model: frozen Phikon + linear head (probe_phikon), the best method on val (hospital 1). The seed
is chosen by a rule fixed in advance and applied on val only: highest val accuracy. The test
set is only read afterwards, to report the chosen run's results.

Writes:
    app/model/head.json     linear head, train feature mean and std, temperature, referral cutoff,
                            pinned backbone revision
    app/model/card.json     selection rule and its result, and the numbers the app displays, read
                            from results/ (none typed by hand)
    app/examples/*.png      one tumor and one normal patch per hospital, drawn at random (seed 0)
                            rather than picked where the model succeeds
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.abstention import apply_cutoff, cutoff_for_coverage
from src.calibration import to_logit
from src.data import load_dataset
from src.features import BACKBONES
from src.train import code_commit
from src.utils import save_json

CONFIG = "probe_phikon"
BACKBONE = "phikon"
COVERAGE = 0.8      # refer the 20% least confident patches (cutoff set on val), chosen in Phase 6
HOSPITAL_SPLITS = {0: "train", 1: "val", 2: "test", 3: "train", 4: "train"}
HOSPITAL_ROLES = {0: "training", 1: "validation (unseen in training)", 2: "test (unseen)",
                  3: "training", 4: "training"}


def select_seed(results_runs: Path) -> tuple[int, dict]:
    """Seed with the highest official val accuracy; returns it and every seed's val accuracy."""
    val_accuracy = {}
    for path in sorted((results_runs / CONFIG).glob("seed*/metrics.json")):
        metrics = json.loads(path.read_text())
        val_accuracy[int(path.parent.name.removeprefix("seed"))] = metrics["splits"]["val"]["official_accuracy"]
    if not val_accuracy:
        raise SystemExit(f"No metrics in {results_runs / CONFIG}")
    return max(val_accuracy, key=val_accuracy.get), val_accuracy


def referral_cutoff(val_prob) -> float:
    """|logit| below which a patch is referred: the Phase 5 cutoff (80% coverage on val), in logit units
    so that it does not depend on the temperature."""
    confidence = cutoff_for_coverage(val_prob, COVERAGE)
    return float(to_logit(0.5 + confidence)[()])


def draw_examples(dataset, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    split_array = dataset.split_array
    hospital = dataset.metadata_array[:, 0].numpy()
    y = dataset.y_array.numpy()
    examples = []
    for h, split in HOSPITAL_SPLITS.items():
        for label in (1, 0):
            pool = np.flatnonzero((split_array == dataset.split_dict[split]) & (hospital == h) & (y == label))
            index = int(rng.choice(pool))
            examples.append({"index": index, "hospital": h, "split": split, "role": HOSPITAL_ROLES[h],
                             "label": "tumor" if label else "normal",
                             "slide": int(dataset.metadata_array[index, 1]),
                             "file": f"hospital{h}_{'tumor' if label else 'normal'}.png"})
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the demo app's model and examples.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/runs"), help="Weights (best.pt)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--app-dir", type=Path, default=Path("app"))
    args = parser.parse_args()

    runs = args.results_dir / "runs"
    seed, val_accuracy = select_seed(runs)
    run_dir = args.artifacts_dir / CONFIG / f"seed{seed}"
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    calibration = json.loads((runs / CONFIG / f"seed{seed}" / "calibration.json").read_text())
    metrics = json.loads((runs / CONFIG / f"seed{seed}" / "metrics.json").read_text())
    val = pd.read_parquet(runs / CONFIG / f"seed{seed}" / "predictions" / "val.parquet")
    test = pd.read_parquet(runs / CONFIG / f"seed{seed}" / "predictions" / "test.parquet")
    cutoff = referral_cutoff(val.prob)
    confidence_cutoff = cutoff_for_coverage(val.prob, COVERAGE)  # same cutoff, in probability units

    head = {
        "backbone": {"hf_id": BACKBONES[BACKBONE]["hf_id"], "revision": BACKBONES[BACKBONE]["revision"],
                     "license": BACKBONES[BACKBONE]["license"]},
        "preprocessing": {"input_size": 96, "model_size": 224, "resize": "bilinear",
                          "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "feature_mean": checkpoint["feature_mean"].tolist(),
        "feature_std": checkpoint["feature_std"].tolist(),
        "weight": checkpoint["model"]["weight"].tolist(),   # rows: normal, tumor
        "bias": checkpoint["model"]["bias"].tolist(),
        "temperature": calibration["temperature"],
        "referral_abs_logit": cutoff,
    }
    bootstrap = pd.read_csv(args.results_dir / "tables" / "bootstrap.csv")
    test_ci = bootstrap[(bootstrap.config == CONFIG) & (bootstrap.split == "test") & (bootstrap.metric == "accuracy")].iloc[0]
    val_coverage, val_risk = apply_cutoff(val.prob, val.y, confidence_cutoff)
    test_coverage, test_risk = apply_cutoff(test.prob, test.y, confidence_cutoff)
    card = {
        "config": CONFIG,
        "selection": {"rule": "probe_phikon: best method on val accuracy (mean of 4 seeds); seed: highest val "
                              "accuracy. Test never used to choose.",
                      "val_accuracy_by_seed": val_accuracy, "chosen_seed": seed, "selected_epoch": metrics["selected_epoch"]},
        "this_run": {split: {k: metrics["splits"][split][k] for k in ("accuracy", "auc")} for split in ("id_val", "val", "test")},
        "method_test_accuracy_ci95_over_patients": [test_ci.ci_low, test_ci.ci_high],
        "method_test_accuracy_mean_over_seeds": test_ci.estimate,
        "temperature": {"value": calibration["temperature"], "fitted_on": "val (hospital 1)",
                        "ece_test_before_after": [calibration["splits"]["test"]["all"]["ece_raw"],
                                                  calibration["splits"]["test"]["all"]["ece_scaled"]]},
        "referral": {"rule": f"refer the patches whose |logit| is below the cutoff answering {COVERAGE:.0%} of val patches",
                     "val": {"answered": val_coverage, "error_when_answered": val_risk,
                             "error_without_referral": float(((val.prob > 0.5) != (val.y == 1)).mean())},
                     "test": {"answered": test_coverage, "error_when_answered": test_risk,
                              "error_without_referral": float(((test.prob > 0.5) != (test.y == 1)).mean())}},
        "code_commit": code_commit(),
    }

    dataset = load_dataset(args.data_root)
    examples = draw_examples(dataset)
    (args.app_dir / "examples").mkdir(parents=True, exist_ok=True)
    for example in examples:
        dataset.get_input(example["index"]).save(args.app_dir / "examples" / example["file"])
    card["examples"] = examples

    save_json(head, args.app_dir / "model" / "head.json")
    save_json(card, args.app_dir / "model" / "card.json")
    print(f"Seed {seed} chosen on val ({', '.join(f'seed{s} {a:.4f}' for s, a in val_accuracy.items())})")
    print(f"Temperature {head['temperature']:.3f}; referral below |logit| {cutoff:.3f}: val answered "
          f"{val_coverage:.1%} (error {val_risk:.2%}), test answered {test_coverage:.1%} (error {test_risk:.2%})")
    print(f"-> {args.app_dir / 'model'}/head.json, card.json; {len(examples)} examples in {args.app_dir / 'examples'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
