"""Regenerate every table and figure in results/ from the saved predictions and the data.

Usage (from the repository root), after `pip install -r requirements.txt` and `python -m src.prepare_data`:
    python -m src.reproduce

Training is not rerun: 12 CNNs took about 41 GPU-hours on Kaggle T4s (notebooks/kaggle_runner.ipynb),
the frozen features about 4 hours on a laptop. Everything downstream of the saved predictions
(results/runs/<name>/seed<N>/predictions/) is rerun, in the order of the project's phases.
Afterwards, `git status results/` should list no modified file.

Not regenerated: results/benchmarks/ (hardware timings), results/data_provenance.json (rewritten by
prepare_data with the check date) and the demo app files (they need the model weights). Grad-CAM needs
the CNN weights in artifacts/runs/<name>/seed0/best.pt and runs only if they are there.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = ["cnn_none", "cnn_hed", "cnn_macenko", "probe_dino_vitb16", "probe_phikon"]


def module(*args: str) -> list[str]:
    return [sys.executable, "-m", *args]


def execute_notebook(path: Path) -> None:
    """Run the notebook in memory: its tables and figures are written to results/, while the
    notebook file itself (outputs, execution times) stays as committed."""
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate every table and figure from the saved predictions.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/runs"), help="CNN weights, if any")
    args = parser.parse_args()

    has_weights = all((args.artifacts_dir / c / "seed0" / "best.pt").exists() for c in CONFIGS if c.startswith("cnn_"))
    runs = sorted(Path("results/runs").glob("*/seed*"))
    steps = [("Phase 1: exploration notebook", None),
             ("Phase 4: Macenko reference", module("src.color", "reference")),
             ("Phase 4: color galleries and fallback rates", module("src.color", "gallery")),
             ("Phase 4: Macenko fallback and overflow rates", module("src.color", "check", "--n-patches", "9999"))]
    steps += [(f"Phases 2-4: metrics of {run.parent.name}/{run.name}", module("src.evaluate", "--from-results", str(run)))
              for run in runs]
    steps += [(f"Phases 2-4: summary of {c}", module("src.aggregate", c)) for c in CONFIGS]
    steps += [("Phases 2-4: comparison", module("src.compare", *CONFIGS)),
              ("Phase 5: calibration", module("src.calibration", *CONFIGS)),
              ("Phase 5: bootstrap over patients", module("src.bootstrap", *CONFIGS)),
              ("Phase 5: risk-coverage", module("src.abstention", *CONFIGS)),
              ("Phase 5: error galleries" + (" and Grad-CAM" if has_weights else " (Grad-CAM skipped: no CNN weights)"),
               module("src.explain", *CONFIGS, *([] if has_weights else ["--no-gradcam"])))]

    start = time.time()
    for number, (title, command) in enumerate(steps, 1):
        print(f"\n[{number}/{len(steps)}] {title}", flush=True)
        if command is None:
            execute_notebook(Path("notebooks/01_exploration.ipynb"))
        else:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    print(f"\nDone in {(time.time() - start) / 60:.1f} min. Check that nothing changed: git status results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
