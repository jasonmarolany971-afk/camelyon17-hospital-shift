"""Train a CNN on Camelyon17-WILDS, selecting the epoch on the OOD validation hospital.

Usage (from the repository root):
    python -m src.train --config configs/cnn_none.yaml --seed 0
    python -m src.train --config configs/cnn_none.yaml --seed 0 --resume             # after an interruption
    python -m src.train --config configs/cnn_none.yaml --seed 0 --subset 512 --epochs 1   # quick test

Outputs, in <output-dir>/<config name>/seed<seed>/:
    config.json          the exact configuration used
    history.csv          loss and accuracies after every epoch
    last.pt              full training state, to resume an interrupted run
    best.pt              weights of the epoch with the best `val` accuracy
    train_summary.json   selected epoch, timings, versions

The test split is never loaded here: src/evaluate.py measures it once, on the
selected checkpoint.
"""

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import build_transform, get_subset, load_dataset, make_loader
from src.evaluate import official_accuracy, predict
from src.models import build_model
from src.utils import get_device, load_config, save_json, set_seed

MONITORED_SPLITS = ["val", "id_val"]  # evaluated after every epoch; `val` selects the epoch


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp) -> dict:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    # Sums stay on the device: calling .item() at every step would make the
    # CPU wait for the GPU and slow training down.
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    seen = 0
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach() * len(y)
        correct += (logits.argmax(dim=1) == y).sum()
        seen += len(y)
    return {"train_loss": total_loss.item() / seen, "train_acc": correct.item() / seen}


def code_commit() -> str | None:
    """Git commit of the code being run, so every result can be traced to its code."""
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def run_dir_for(output_dir: Path, name: str, seed: int, subset: int | None = None) -> Path:
    """Folder of one run; quick tests (--subset) get their own folder."""
    return Path(output_dir) / name / (f"seed{seed}" + (f"-subset{subset}" if subset else ""))


def save_atomically(state: dict, path: Path) -> None:
    """Write to a temporary file, then rename: an interruption never leaves a corrupt file."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a CNN on Camelyon17-WILDS.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--num-workers", type=int, default=None, help="Override the config value.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of epochs (quick tests).")
    parser.add_argument("--subset", type=int, default=None, help="Use N random patches per split (quick tests).")
    parser.add_argument("--resume", action="store_true", help="Continue from last.pt.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    run_dir = run_dir_for(args.output_dir, config["name"], args.seed, args.subset)
    if (run_dir / "last.pt").exists() and not args.resume:
        print(f"{run_dir} already contains a run: use --resume to continue it, or delete the folder.")
        return 1
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device(args.device)
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"

    dataset = load_dataset(args.data_root)
    # Augmentation (if any) on training patches only; evaluation patches are never augmented.
    train_set = get_subset(dataset, "train", build_transform(config, train=True), args.subset, seed=args.seed)
    eval_transform = build_transform(config, train=False)
    monitored = {split: get_subset(dataset, split, eval_transform, args.subset, seed=args.seed)
                 for split in MONITORED_SPLITS}

    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"]
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    generator = torch.Generator()
    train_loader = make_loader(train_set, config["batch_size"], shuffle=True,
                               num_workers=config["num_workers"], generator=generator)

    start_epoch, best_val_acc, history = 0, -1.0, []
    if args.resume and (run_dir / "last.pt").exists():
        state = torch.load(run_dir / "last.pt", map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_val_acc, history = state["epoch"] + 1, state["best_val_acc"], state["history"]
        print(f"Resuming {run_dir} after epoch {state['epoch']}")

    save_json({**config, "seed": args.seed, "subset": args.subset}, run_dir / "config.json")
    print(f"Training {config['name']} seed {args.seed} on {device} (amp={use_amp}): "
          f"{len(train_set):,} training patches, {config['epochs']} epochs")

    for epoch in range(start_epoch, config["epochs"]):
        # The shuffling order depends only on (seed, epoch): a resumed run sees
        # exactly the batches it would have seen without interruption.
        generator.manual_seed(args.seed * 1000 + epoch)
        epoch_start = time.time()
        row = {"epoch": epoch, **train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)}
        for split, subset in monitored.items():
            loader = make_loader(subset, config["eval_batch_size"], shuffle=False, num_workers=config["num_workers"])
            probs, y, metadata = predict(model, loader, device)
            row[f"{split}_acc"] = official_accuracy(dataset, probs, y, metadata)
        row["epoch_minutes"] = (time.time() - epoch_start) / 60
        history.append(row)

        # Model selection, as in WILDS: keep the epoch with the best accuracy on val.
        is_best = row["val_acc"] > best_val_acc
        if is_best:
            best_val_acc = row["val_acc"]
            save_atomically({"model": model.state_dict(), "epoch": epoch, "val_acc": best_val_acc,
                             "config": config, "seed": args.seed}, run_dir / "best.pt")
        save_atomically({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scaler": scaler.state_dict(), "epoch": epoch, "best_val_acc": best_val_acc,
                         "history": history}, run_dir / "last.pt")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"epoch {epoch}: loss {row['train_loss']:.4f} | train acc {row['train_acc']:.4f} | "
              f"id_val acc {row['id_val_acc']:.4f} | val acc {row['val_acc']:.4f}"
              f"{'  <- best so far' if is_best else ''} | {row['epoch_minutes']:.1f} min")

    best = max(history, key=lambda r: r["val_acc"])  # first epoch reaching the maximum, as above
    save_json({
        "run_dir": str(run_dir),
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "epochs": config["epochs"],
        "train_minutes": sum(r["epoch_minutes"] for r in history),
        "device": str(device),
        "amp": use_amp,
        "code_commit": code_commit(),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }, run_dir / "train_summary.json")
    print(f"Done. Best epoch {best['epoch']} (val acc {best['val_acc']:.4f}). "
          f"Next: python -m src.evaluate --run-dir {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
