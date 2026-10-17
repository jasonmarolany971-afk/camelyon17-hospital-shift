"""Evaluate a trained run on id_val, val and test, globally and per hospital.

Usage (from the repository root):
    python -m src.evaluate --run-dir artifacts/runs/cnn_none/seed0

Metrics: official WILDS accuracy, AUC, and sensitivity at a fixed specificity.
The threshold for the latter is chosen on the OOD validation hospital (`val`)
and applied unchanged to the other splits: the test hospital is only
measured, never used to choose anything.

Outputs:
    <run-dir>/predictions/<split>.parquet   one row per patch (probability, label, hospital, slide)
    results/runs/<name>/seed<seed>/metrics.json and history.csv
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from src.data import build_transform, get_subset, load_dataset, make_loader
from src.models import build_model
from src.utils import get_device, save_json
from src.wilds_compat import ensure_torch_scatter

EVAL_SPLITS = ["id_val", "val", "test"]


@torch.no_grad()
def predict(model, loader, device):
    """Tumor probability, label and metadata (hospital, slide, y) for every patch."""
    model.eval()
    probs, labels, metadata = [], [], []
    for x, y, meta in tqdm(loader, desc="predict", leave=False):
        logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        labels.append(y)
        metadata.append(meta)
    return torch.cat(probs), torch.cat(labels), torch.cat(metadata)


def official_accuracy(dataset, probs, y, metadata) -> float:
    """Accuracy computed by the WILDS evaluator.

    `probs > 0.5` is the same decision as WILDS' argmax over the two logits.
    """
    ensure_torch_scatter()
    results, _ = dataset.eval((probs > 0.5).long(), y, metadata)
    return results["acc_avg"]


def threshold_at_specificity(y_true, scores, specificity: float) -> float:
    """Threshold with the highest sensitivity among those reaching `specificity`.

    A patch is predicted tumor if its score is >= threshold.
    """
    # drop_intermediate=False keeps every threshold, so the best one cannot be dropped.
    fpr, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=False)
    # Compare specificities with a tolerance: in floating point, 1 - 0.9 != 0.1.
    allowed = np.flatnonzero(1 - fpr >= specificity - 1e-9)
    return float(thresholds[allowed[np.argmax(tpr[allowed])]])


def binary_metrics(y_true, scores, threshold: float) -> dict:
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    positives, negatives = y_true == 1, y_true == 0
    predicted = scores >= threshold
    both_classes = positives.any() and negatives.any()
    return {
        "n": int(len(y_true)),
        "accuracy": float(((scores > 0.5) == positives).mean()),
        "auc": float(roc_auc_score(y_true, scores)) if both_classes else None,
        "sensitivity_at_threshold": float(predicted[positives].mean()) if positives.any() else None,
        "specificity_at_threshold": float((~predicted[negatives]).mean()) if negatives.any() else None,
    }


def evaluate_run(run_dir: Path, data_root=None, device_name=None, batch_size=256, num_workers=4,
                 subset=None, target_specificity=0.95, results_root=Path("results/runs")) -> dict:
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    config, seed = checkpoint["config"], checkpoint["seed"]
    device = get_device(device_name)
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    dataset = load_dataset(data_root)
    transform = build_transform(config["color"])
    (run_dir / "predictions").mkdir(exist_ok=True)
    frames, official = {}, {}
    for split in EVAL_SPLITS:
        split_subset = get_subset(dataset, split, transform, subset, seed=seed)
        loader = make_loader(split_subset, batch_size, shuffle=False, num_workers=num_workers)
        probs, y, metadata = predict(model, loader, device)
        official[split] = official_accuracy(dataset, probs, y, metadata)
        frames[split] = pd.DataFrame({
            "index": split_subset.indices,
            "y": y.numpy(),
            "prob": probs.numpy(),
            "hospital": metadata[:, 0].numpy(),
            "slide": metadata[:, 1].numpy(),
        })
        frames[split].to_parquet(run_dir / "predictions" / f"{split}.parquet", index=False)

    # Chosen on val only, then applied as is to every split.
    threshold = threshold_at_specificity(frames["val"].y, frames["val"].prob, target_specificity)
    metrics = {
        "run": f"{config['name']}/seed{seed}",
        "selected_epoch": checkpoint["epoch"],
        "threshold": {"value": threshold, "chosen_on": "val", "target_specificity": target_specificity},
        "splits": {},
    }
    for split, frame in frames.items():
        entry = {"official_accuracy": official[split], **binary_metrics(frame.y, frame.prob, threshold)}
        # Sanity check: our accuracy and the WILDS evaluator must agree.
        if abs(entry["accuracy"] - entry["official_accuracy"]) > 1e-6:
            raise RuntimeError(f"{split}: accuracy {entry['accuracy']} != official {entry['official_accuracy']}")
        entry["by_hospital"] = {
            int(hospital): binary_metrics(part.y, part.prob, threshold) for hospital, part in frame.groupby("hospital")
        }
        metrics["splits"][split] = entry

    # Quick tests (--subset) stay next to the run; only full evaluations go to results/.
    out_dir = run_dir if subset else results_root / config["name"] / f"seed{seed}"
    save_json(metrics, out_dir / "metrics.json")
    if out_dir != run_dir:
        shutil.copy(run_dir / "history.csv", out_dir / "history.csv")
    print(f"Metrics written to {out_dir / 'metrics.json'}")
    return metrics


def print_summary(metrics: dict) -> None:
    print(f"\n{metrics['run']} (epoch {metrics['selected_epoch']} selected on val)")
    print(f"Threshold for {metrics['threshold']['target_specificity']:.0%} specificity on val: "
          f"{metrics['threshold']['value']:.4f}")
    print(f"{'split':<8} {'n':>7} {'accuracy':>9} {'AUC':>7} {'sens@thr':>9} {'spec@thr':>9}")
    for split, entry in metrics["splits"].items():
        auc = f"{entry['auc']:.4f}" if entry["auc"] is not None else "-"
        print(f"{split:<8} {entry['n']:>7} {entry['accuracy']:>9.4f} {auc:>7} "
              f"{entry['sensitivity_at_threshold']:>9.4f} {entry['specificity_at_threshold']:>9.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained run on id_val, val and test.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subset", type=int, default=None, help="Use N random patches per split (quick tests).")
    parser.add_argument("--target-specificity", type=float, default=0.95)
    args = parser.parse_args()

    if not (args.run_dir / "best.pt").exists():
        print(f"No best.pt in {args.run_dir}: train the model first.")
        return 1
    metrics = evaluate_run(args.run_dir, args.data_root, args.device, args.batch_size, args.num_workers,
                           args.subset, args.target_specificity)
    print_summary(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
