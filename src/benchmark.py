"""Measure training speed on this machine and estimate the cost of full runs.

Usage (from the repository root):
    python -m src.benchmark --config configs/cnn_none.yaml

Three measurements, each after a few warm-up batches:
  1. data loading  images/s delivered by the DataLoader (PNG decoding, on CPU)
  2. computation   images/s for training steps on random tensors (GPU only)
  3. end to end    real batches through real training steps
End to end is bounded by the slower of 1 and 2: that one is what to optimize.
Writes results/benchmarks/<device>.json.
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.data import build_transform, get_subset, load_dataset, make_loader
from src.models import build_model
from src.utils import get_device, load_config, save_json, set_seed, synchronize

# Official split sizes (see src/prepare_data.py).
N_TRAIN = 302_436
N_MONITORED = 34_904 + 33_560  # val + id_val, evaluated after every epoch
N_TEST_EVAL = 33_560 + 34_904 + 85_054  # final evaluation of id_val, val and test
WARMUP = 5


def loader_throughput(loader, steps: int) -> float:
    batches = iter(loader)
    for _ in range(WARMUP):
        next(batches)
    start, images = time.perf_counter(), 0
    for _ in range(steps):
        x, _, _ = next(batches)
        images += len(x)
    return images / (time.perf_counter() - start)


def train_step(model, optimizer, scaler, x, y, device, use_amp) -> None:
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        loss = nn.functional.cross_entropy(model(x), y)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def compute_throughput(model, optimizer, scaler, batch_size, device, use_amp, steps) -> float:
    x = torch.randn(batch_size, 3, 96, 96, device=device)
    y = torch.randint(0, 2, (batch_size,), device=device)
    for _ in range(WARMUP):
        train_step(model, optimizer, scaler, x, y, device, use_amp)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(steps):
        train_step(model, optimizer, scaler, x, y, device, use_amp)
    synchronize(device)
    return steps * batch_size / (time.perf_counter() - start)


@torch.no_grad()
def inference_throughput(model, batch_size, device, steps) -> float:
    model.eval()
    x = torch.randn(batch_size, 3, 96, 96, device=device)
    for _ in range(WARMUP):
        model(x)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(steps):
        model(x)
    synchronize(device)
    model.train()
    return steps * batch_size / (time.perf_counter() - start)


def end_to_end_throughput(model, optimizer, scaler, loader, device, use_amp, steps) -> float:
    batches = iter(loader)
    for _ in range(WARMUP):
        x, y, _ = next(batches)
        train_step(model, optimizer, scaler, x.to(device), y.to(device), device, use_amp)
    synchronize(device)
    start, images = time.perf_counter(), 0
    for _ in range(steps):
        x, y, _ = next(batches)
        train_step(model, optimizer, scaler, x.to(device), y.to(device), device, use_amp)
        images += len(x)
    synchronize(device)
    return images / (time.perf_counter() - start)


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure training speed and estimate run times.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None, help="Override the config value.")
    parser.add_argument("--steps", type=int, default=50, help="Batches timed per measurement.")
    parser.add_argument("--runs", type=int, default=9, help="Number of full runs to budget for.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/benchmarks"))
    args = parser.parse_args()

    config = load_config(args.config)
    num_workers = args.num_workers if args.num_workers is not None else config["num_workers"]
    device = get_device(args.device)
    set_seed(0)
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else f"{device.type} ({platform.machine()})"
    print(f"Device: {device_name} | CPU cores: {os.cpu_count()} | DataLoader workers: {num_workers}")

    dataset = load_dataset(args.data_root)
    train_set = get_subset(dataset, "train", build_transform(config["color"]))
    loader = make_loader(train_set, config["batch_size"], shuffle=True, num_workers=num_workers,
                         generator=torch.Generator().manual_seed(0))

    results = {"device": device_name, "cpu_cores": os.cpu_count(), "num_workers": num_workers,
               "batch_size": config["batch_size"], "torch": torch.__version__}
    print("1/3 data loading ...")
    results["data_images_per_s"] = loader_throughput(loader, args.steps)

    amp_options = [False, True] if device.type == "cuda" else [False]
    for use_amp in amp_options:
        tag = "amp" if use_amp else "fp32"
        model = build_model(config["model"]).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                                    weight_decay=config["weight_decay"])
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        print(f"2/3 computation ({tag}) ...")
        results[f"train_compute_images_per_s_{tag}"] = compute_throughput(
            model, optimizer, scaler, config["batch_size"], device, use_amp, args.steps)
        results[f"inference_images_per_s_{tag}"] = inference_throughput(
            model, config["eval_batch_size"], device, max(5, args.steps // 5))
        print(f"3/3 end to end ({tag}) ...")
        e2e = end_to_end_throughput(model, optimizer, scaler, loader, device, use_amp, args.steps)
        results[f"end_to_end_images_per_s_{tag}"] = e2e

        # Evaluation is bounded by decoding or by inference, whichever is slower.
        eval_rate = min(results["data_images_per_s"], results[f"inference_images_per_s_{tag}"])
        epoch_minutes = (N_TRAIN / e2e + N_MONITORED / eval_rate) / 60
        run_hours = (config["epochs"] * epoch_minutes + N_TEST_EVAL / eval_rate / 60) / 60
        results[f"estimate_{tag}"] = {
            "epoch_minutes": epoch_minutes,
            "run_hours": run_hours,
            f"hours_for_{args.runs}_runs": run_hours * args.runs,
        }

    out = args.out_dir / f"{device_name.replace(' ', '_').replace('(', '').replace(')', '')}.json"
    save_json(results, out)

    print(f"\nData loading:        {results['data_images_per_s']:8.0f} images/s")
    for use_amp in amp_options:
        tag = "amp" if use_amp else "fp32"
        estimate = results[f"estimate_{tag}"]
        print(f"[{tag}] computation:  {results[f'train_compute_images_per_s_{tag}']:8.0f} images/s (training step)")
        print(f"[{tag}] inference:    {results[f'inference_images_per_s_{tag}']:8.0f} images/s")
        print(f"[{tag}] end to end:   {results[f'end_to_end_images_per_s_{tag}']:8.0f} images/s")
        print(f"[{tag}] estimate:     {estimate['epoch_minutes']:.0f} min/epoch, {estimate['run_hours']:.1f} h/run, "
              f"{estimate[f'hours_for_{args.runs}_runs']:.0f} h for {args.runs} runs")
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
