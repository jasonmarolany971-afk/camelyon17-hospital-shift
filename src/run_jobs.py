"""Run several trainings at the same time, one per GPU, then evaluate each run.

Made for Kaggle sessions, which have two T4 GPUs: the quota counts session
time, not GPUs, so two trainings per session halve its consumption.

Usage (from the repository root):
    python -m src.run_jobs --job configs/cnn_none.yaml:0 --job configs/cnn_none.yaml:1

Each job sees a single GPU (CUDA_VISIBLE_DEVICES) and writes its own log in
<log-dir>/<name>_seed<seed>.log. The last finished epoch of every job is
printed every few minutes. Once all trainings are over, each successful run
is evaluated with src.evaluate.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src.train import run_dir_for
from src.utils import load_config


def parse_job(text: str) -> tuple[str, int]:
    """'configs/cnn_none.yaml:0' -> ('configs/cnn_none.yaml', 0)."""
    config, _, seed = text.rpartition(":")
    if not config or not seed.isdigit():
        raise argparse.ArgumentTypeError(f"Expected CONFIG:SEED, got {text!r}")
    return config, int(seed)


def last_epoch_line(log_text: str) -> str | None:
    """Last 'epoch N: ...' line printed by src.train (progress bars use \\r)."""
    lines = [part.strip() for part in log_text.replace("\r", "\n").split("\n")]
    epochs = [line for line in lines if line.startswith("epoch ")]
    return epochs[-1] if epochs else None


def tail(log_path: Path, n: int = 20) -> str:
    text = log_path.read_text(errors="replace").replace("\r", "\n")
    return "\n".join([line for line in text.split("\n") if line.strip()][-n:])


def start_job(config_path, seed, gpu, args) -> dict:
    name = load_config(config_path)["name"]
    run_dir = run_dir_for(args.output_dir, name, seed, args.subset)
    command = [sys.executable, "-m", "src.train", "--config", config_path, "--seed", str(seed),
               "--output-dir", str(args.output_dir), "--num-workers", str(args.num_workers)]
    if args.resume_from:
        previous = run_dir_for(args.resume_from, name, seed, args.subset)
        if (previous / "last.pt").exists():
            shutil.copytree(previous, run_dir, dirs_exist_ok=True)
            command.append("--resume")
    for option in ("data_root", "device", "epochs", "subset"):
        value = getattr(args, option)
        if value is not None:
            command += [f"--{option.replace('_', '-')}", str(value)]

    log_path = args.log_dir / f"{name}_seed{seed}.log"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"Started {name} seed {seed} on GPU {gpu} (log: {log_path})", flush=True)
    return {"name": name, "seed": seed, "run_dir": run_dir, "log": log_path, "process": process}


def report(jobs: list[dict], start: float) -> None:
    print(f"--- {(time.time() - start) / 60:.0f} min", flush=True)
    for job in jobs:
        code = job["process"].poll()
        status = "running" if code is None else ("done" if code == 0 else f"FAILED (exit {code})")
        line = last_epoch_line(job["log"].read_text(errors="replace")) or "no epoch finished yet"
        print(f"  {job['name']} seed {job['seed']} [{status}] {line}", flush=True)


def evaluate(job: dict, args) -> int:
    command = [sys.executable, "-m", "src.evaluate", "--run-dir", str(job["run_dir"])]
    for option in ("data_root", "device", "subset"):
        value = getattr(args, option)
        if value is not None:
            command += [f"--{option.replace('_', '-')}", str(value)]
    return subprocess.run(command).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run trainings in parallel, one per GPU, then evaluate them.")
    parser.add_argument("--job", type=parse_job, action="append", required=True, help="CONFIG:SEED, repeatable.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers per job (4 CPUs on Kaggle).")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Runs folder of an interrupted session: jobs with a last.pt there are resumed.")
    parser.add_argument("--report-minutes", type=float, default=5.0)
    parser.add_argument("--device", default=None, help="Passed to src.train and src.evaluate (quick tests).")
    parser.add_argument("--epochs", type=int, default=None, help="Passed to src.train (quick tests).")
    parser.add_argument("--subset", type=int, default=None, help="Passed to src.train and src.evaluate (quick tests).")
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    jobs = [start_job(config, seed, gpu, args) for gpu, (config, seed) in enumerate(args.job)]
    start = last_report = time.time()
    try:
        while any(job["process"].poll() is None for job in jobs):
            time.sleep(5)
            if time.time() - last_report >= args.report_minutes * 60:
                report(jobs, start)
                last_report = time.time()
    except KeyboardInterrupt:
        for job in jobs:
            job["process"].terminate()  # do not leave trainings running in the background
        raise
    report(jobs, start)

    failed = [job for job in jobs if job["process"].returncode != 0]
    for job in failed:
        print(f"\n{job['name']} seed {job['seed']} failed. Last lines of {job['log']}:\n{tail(job['log'])}", flush=True)
    # A failure of one job must not prevent evaluating the others.
    for job in jobs:
        if job not in failed and evaluate(job, args) != 0:
            failed.append(job)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
