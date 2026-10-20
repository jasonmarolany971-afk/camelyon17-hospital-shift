"""Train a linear classifier on frozen features, with the same protocol as the CNN.

Usage (from the repository root):
    python -m src.probe --config configs/probe_phikon.yaml --seed 0

Same protocol as src/train.py: SGD, 10 epochs, batch 32, epoch selected on the
official WILDS accuracy of `val`. The seed sets the initialization of the
linear layer and the order of the batches. Runs on the CPU on purpose: the
layer is tiny, and CPU training is reproducible bit for bit.

Features are standardized with the mean and standard deviation of the train
split only. The test split is loaded only after training, to evaluate the
selected epoch once, with the threshold chosen on `val`.

Outputs, in the same layout as the CNN (so src/aggregate.py works unchanged):
    artifacts/runs/<name>/seed<seed>/   best.pt, history.csv, train_summary.json
    results/runs/<name>/seed<seed>/     predictions/<split>.parquet, metrics.json, history.csv, train_summary.json
"""

import argparse
import copy
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data import load_dataset
from src.evaluate import EVAL_SPLITS, compute_metrics, official_accuracy
from src.train import code_commit, run_dir_for
from src.utils import load_config, save_json, set_seed

MONITORED_SPLITS = ["val", "id_val"]


def load_split(features_dir: Path, split: str, dataset) -> dict:
    """Features of one split, aligned with labels and metadata through the saved dataset indices."""
    index = np.load(features_dir / f"{split}_index.npy")
    return {
        "index": index,
        "x": torch.from_numpy(np.load(features_dir / f"{split}.npy").astype(np.float32)),
        "y": dataset.y_array[index],
        "metadata": dataset.metadata_array[index],
    }


def fit_standardizer(train_x: torch.Tensor):
    """Mean and standard deviation of the train split only (never val or test)."""
    return train_x.mean(dim=0), train_x.std(dim=0).clamp_min(1e-6)


def batches(n: int, batch_size: int, generator: torch.Generator):
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def train_epoch(model, optimizer, x, y, batch_size, generator) -> dict:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct = 0.0, 0
    for batch in batches(len(y), batch_size, generator):
        logits = model(x[batch])
        loss = loss_fn(logits, y[batch])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)
        correct += (logits.argmax(dim=1) == y[batch]).sum().item()
    return {"train_loss": total_loss / len(y), "train_acc": correct / len(y)}


@torch.no_grad()
def tumor_probability(model, x) -> torch.Tensor:
    model.eval()
    return torch.softmax(model(x), dim=1)[:, 1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a linear probe on frozen features.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--features-dir", type=Path, default=None,
                        help="Default: artifacts/features/<features from the config>.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of epochs (quick tests).")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    features_dir = args.features_dir or Path("artifacts/features") / config["features"]
    run_dir = run_dir_for(args.output_dir, config["name"], args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.set_num_threads(1)  # tiny model: one thread is fast and keeps the run deterministic

    dataset = load_dataset(args.data_root)
    train = load_split(features_dir, "train", dataset)
    monitored = {split: load_split(features_dir, split, dataset) for split in MONITORED_SPLITS}
    mean, std = fit_standardizer(train["x"])
    for part in (train, *monitored.values()):
        part["x"] = (part["x"] - mean) / std

    model = nn.Linear(train["x"].shape[1], 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"])
    generator = torch.Generator()
    print(f"Training {config['name']} seed {args.seed} on {len(train['y']):,} feature vectors "
          f"({features_dir}), {config['epochs']} epochs")

    history, best_val_acc, best_state, start = [], -1.0, None, time.time()
    for epoch in range(config["epochs"]):
        generator.manual_seed(args.seed * 1000 + epoch)  # same rule as the CNN
        row = {"epoch": epoch, **train_epoch(model, optimizer, train["x"], train["y"], config["batch_size"], generator)}
        for split, part in monitored.items():
            row[f"{split}_acc"] = official_accuracy(dataset, tumor_probability(model, part["x"]), part["y"],
                                                    part["metadata"])
        history.append(row)
        is_best = row["val_acc"] > best_val_acc
        if is_best:
            best_val_acc, best_epoch, best_state = row["val_acc"], epoch, copy.deepcopy(model.state_dict())
        print(f"epoch {epoch}: loss {row['train_loss']:.4f} | train acc {row['train_acc']:.4f} | "
              f"id_val acc {row['id_val_acc']:.4f} | val acc {row['val_acc']:.4f}{'  <- best so far' if is_best else ''}")
    train_minutes = (time.time() - start) / 60

    checkpoint = {"model": best_state, "epoch": best_epoch, "val_acc": best_val_acc, "config": config,
                  "seed": args.seed, "feature_mean": mean, "feature_std": std}
    torch.save(checkpoint, run_dir / "best.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    feature_meta = json.loads((features_dir / "meta.json").read_text()) if (features_dir / "meta.json").exists() else {}
    save_json({
        "run_dir": str(run_dir), "best_epoch": best_epoch, "best_val_acc": best_val_acc, "epochs": config["epochs"],
        "train_minutes": train_minutes, "device": "cpu", "features": str(features_dir),
        "feature_revision": feature_meta.get("revision"), "feature_code_commit": feature_meta.get("code_commit"),
        "code_commit": code_commit(), "torch": torch.__version__, "python": platform.python_version(),
    }, run_dir / "train_summary.json")

    # Training is over: only now is the test split loaded, to evaluate the selected epoch once.
    model.load_state_dict(best_state)
    out_dir = args.results_dir / config["name"] / f"seed{args.seed}"
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    frames = {}
    for split in EVAL_SPLITS:
        part = monitored[split] if split in monitored else load_split(features_dir, split, dataset)
        if split not in monitored:
            part["x"] = (part["x"] - mean) / std
        metadata = part["metadata"].numpy()
        frames[split] = pd.DataFrame({
            "index": part["index"], "y": part["y"].numpy(), "prob": tumor_probability(model, part["x"]).numpy(),
            "hospital": metadata[:, 0], "slide": metadata[:, 1],
        })
        frames[split].to_parquet(out_dir / "predictions" / f"{split}.parquet", index=False)

    metrics = compute_metrics(dataset, frames, checkpoint, target_specificity=0.95)
    save_json(metrics, out_dir / "metrics.json")
    for name in ("history.csv", "train_summary.json"):
        shutil.copy(run_dir / name, out_dir / name)
    test = metrics["splits"]["test"]
    print(f"Done. Epoch {best_epoch} selected on val (acc {best_val_acc:.4f}) | "
          f"test acc {test['accuracy']:.4f}, AUC {test['auc']:.4f} | {train_minutes:.1f} min | metrics: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
