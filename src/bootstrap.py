"""Confidence intervals that account for the choice of patients: cluster bootstrap over patients.

Usage (from the repository root):
    python -m src.bootstrap cnn_none cnn_hed cnn_macenko probe_dino_vitb16 probe_phikon

Each unseen hospital contributes 10 slides from 8 (val) or 9 (test) patients; another hospital,
with other patients, would give other numbers. One replicate redraws the hospital's patients with
replacement (some twice, some not at all), keeps every patch of the drawn patients and recomputes
the metrics of every run. 2,000 replicates give a 95% percentile interval.

- Patients, not patches: patches of one slide share tissue and staining, they are not independent.
- Patients, not slides: two slides of the same patient share the same biology.
- Paired: every configuration sees the same draws, so differences get their own interval.
- A replicate's value is the mean over the seeds of a configuration; seed-to-seed variation is
  reported separately (std in results/tables/comparison.csv).
With 8 or 9 clusters the percentile interval is approximate and tends to be too narrow.

Reads results/runs/<name>/seed<N>/predictions/{val,test}.parquet and the patient IDs in
<data-root>/camelyon17_v1.0/metadata.csv (rows in dataset order); writes:
    results/tables/bootstrap.csv              estimate and 95% interval per (configuration, split, metric)
    results/tables/bootstrap_differences.csv  every pair of configurations: difference and its interval
    results/figures/bootstrap.png             the intervals, on val and test
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration import find_runs
from src.compare import LABELS
from src.plotting import COLORS, INK, INK_SOFT, apply_style, plt
from src.utils import default_data_root

SPLITS = ["val", "test"]
METRICS = ["accuracy", "auc"]
SPLIT_TITLES = {"val": "val (hospital 1)", "test": "test (hospital 2)"}
CHUNK = 100  # replicates computed together (memory: CHUNK x number of distinct probabilities)


def draw_weights(n_clusters: int, n_replicates: int, seed: int) -> np.ndarray:
    """(n_replicates, n_clusters): how many times each patient is drawn in each replicate."""
    draws = np.random.default_rng(seed).integers(0, n_clusters, size=(n_replicates, n_clusters))
    return np.stack([np.bincount(row, minlength=n_clusters) for row in draws]).astype(np.float64)


def cluster_arrays(prob, y, cluster, n_clusters: int) -> dict:
    """Per-patient counts from which any weighted accuracy or AUC follows without a loop over patches."""
    prob, y, cluster = np.asarray(prob), np.asarray(y).astype(bool), np.asarray(cluster)
    values, inverse = np.unique(prob, return_inverse=True)  # sorted; tied probabilities share a column

    def table(mask):
        flat = np.bincount(cluster[mask] * len(values) + inverse[mask], minlength=n_clusters * len(values))
        return flat.reshape(n_clusters, len(values)).astype(np.float64)

    correct = (prob > 0.5) == y  # the official WILDS decision
    return {"pos": table(y), "neg": table(~y),
            "correct": np.bincount(cluster, weights=correct, minlength=n_clusters),
            "n": np.bincount(cluster, minlength=n_clusters).astype(np.float64)}


def weighted_metrics(arrays: dict, weights) -> dict[str, np.ndarray]:
    """Accuracy at 0.5 and AUC when patient k counts weights[r, k] times (one row r per replicate).

    AUC = P(score of a tumor patch > score of a normal patch), ties counted half.
    """
    weights = np.atleast_2d(weights)
    pos, neg = weights @ arrays["pos"], weights @ arrays["neg"]
    below = np.cumsum(neg, axis=1) - neg  # normal patches with a strictly lower probability
    auc = (pos * (below + 0.5 * neg)).sum(axis=1) / (pos.sum(axis=1) * neg.sum(axis=1))
    accuracy = (weights @ arrays["correct"]) / (weights @ arrays["n"])
    return {"accuracy": accuracy, "auc": auc}


def load_patients(data_root: Path) -> np.ndarray:
    metadata = data_root / "camelyon17_v1.0" / "metadata.csv"
    return pd.read_csv(metadata, usecols=["patient"], dtype={"patient": str}).patient.to_numpy()


def bootstrap(names: list[str], runs_dir: Path, patients: np.ndarray, n_replicates: int, seed: int):
    """(estimates, replicates, n_patients): seed means on the full split, and on every replicate."""
    estimates, replicates, n_patients = {}, {}, {}
    for split_number, split in enumerate(SPLITS):
        reference_index = weights = None
        for name in names:
            runs = find_runs(runs_dir, name)
            if not runs:
                raise SystemExit(f"No predictions in {runs_dir / name}")
            per_seed_estimates, per_seed_replicates = [], []
            for run_dir in runs.values():
                frame = pd.read_parquet(run_dir / "predictions" / f"{split}.parquet").sort_values("index")
                index = frame["index"].to_numpy()
                if reference_index is None:  # first run: fixes the patients and the draws for every run
                    reference_index = index
                    order = np.unique(patients[index])
                    n_patients[split] = len(order)
                    weights = draw_weights(len(order), n_replicates, seed + split_number)
                elif not np.array_equal(index, reference_index):
                    raise ValueError(f"{run_dir}: {split} predictions cover other patches than the first run")
                arrays = cluster_arrays(frame.prob, frame.y, np.searchsorted(order, patients[index]), len(order))
                per_seed_estimates.append(weighted_metrics(arrays, np.ones(len(order))))
                chunks = [weighted_metrics(arrays, weights[i:i + CHUNK]) for i in range(0, n_replicates, CHUNK)]
                per_seed_replicates.append({m: np.concatenate([c[m] for c in chunks]) for m in METRICS})
            for metric in METRICS:
                estimates[(name, split, metric)] = float(np.mean([e[metric][0] for e in per_seed_estimates]))
                replicates[(name, split, metric)] = np.mean([r[metric] for r in per_seed_replicates], axis=0)
    return estimates, replicates, n_patients


def interval_table(names, estimates, replicates, n_patients, n_replicates) -> pd.DataFrame:
    rows = []
    for name in names:
        for split in SPLITS:
            for metric in METRICS:
                low, high = np.percentile(replicates[(name, split, metric)], [2.5, 97.5])
                rows.append({"config": name, "split": split, "metric": metric,
                             "estimate": estimates[(name, split, metric)], "ci_low": low, "ci_high": high,
                             "n_patients": n_patients[split], "n_replicates": n_replicates})
    return pd.DataFrame(rows)


def difference_table(names, estimates, replicates) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for metric in METRICS:
            for a, b in combinations(names, 2):
                difference = replicates[(a, split, metric)] - replicates[(b, split, metric)]
                low, high = np.percentile(difference, [2.5, 97.5])
                rows.append({"split": split, "metric": metric, "config_a": a, "config_b": b,
                             "difference": estimates[(a, split, metric)] - estimates[(b, split, metric)],
                             "ci_low": low, "ci_high": high, "share_replicates_a_better": float(np.mean(difference > 0))})
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, names: list[str], path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(len(METRICS), len(SPLITS), figsize=(10, 6.4), sharex="row", sharey=True)
    for row, (metric, xlabel, scale) in enumerate([("accuracy", "Accuracy (%)", 100), ("auc", "AUC", 1)]):
        for col, split in enumerate(SPLITS):
            ax = axes[row, col]
            for y, name in enumerate(names):
                r = table[(table.config == name) & (table.split == split) & (table.metric == metric)].iloc[0]
                ax.hlines(y, r.ci_low * scale, r.ci_high * scale, color=COLORS[0], linewidth=2.5, zorder=2)
                ax.scatter(r.estimate * scale, y, s=36, color=INK, zorder=3)
                ax.text(r.ci_high * scale, y + 0.16, f" {r.estimate * scale:.1f} [{r.ci_low * scale:.1f}, "
                        f"{r.ci_high * scale:.1f}]" if scale == 100 else
                        f" {r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", fontsize=7.5, color=INK_SOFT,
                        ha="right", va="top")  # below the line
            ax.set_yticks(range(len(names)), [LABELS.get(n, n).replace("\n", " ") for n in names])
            ax.set_ylim(len(names) - 0.4, -0.6)
            ax.grid(True, axis="x")
            ax.grid(False, axis="y")
            if row == 0:
                ax.set_title(f"{SPLIT_TITLES[split]}, {int(table[table.split == split].n_patients.iloc[0])} patients")
            ax.set_xlabel(xlabel)
    fig.suptitle("95% intervals from redrawing the patients of each hospital", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    n_replicates = int(table.n_replicates.iloc[0])
    fig.text(0.01, -0.02, f"Dot: mean over 4 seeds on the real hospital. Line: 2.5th-97.5th percentile over "
             f"{n_replicates:,} replicates (patients drawn with replacement, same draws for every method).\n"
             "With 8-9 patients the intervals are approximate and tend to be too narrow.",
             color=INK_SOFT, fontsize=8.5, va="top")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cluster bootstrap over patients on val and test.")
    parser.add_argument("names", nargs="+", help="Configuration names, e.g. cnn_none probe_phikon")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("results/runs"), help="Saved predictions")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    patients = load_patients(args.data_root or default_data_root())
    estimates, replicates, n_patients = bootstrap(args.names, args.runs_dir, patients, args.replicates, args.seed)
    table = interval_table(args.names, estimates, replicates, n_patients, args.replicates)
    differences = difference_table(args.names, estimates, replicates)
    tables = args.results_dir / "tables"
    table.to_csv(tables / "bootstrap.csv", index=False)
    differences.to_csv(tables / "bootstrap_differences.csv", index=False)
    plot(table, args.names, args.results_dir / "figures" / "bootstrap.png")

    for split in SPLITS:
        print(f"{split} ({n_patients[split]} patients)")
        for _, r in table[table.split == split].iterrows():
            print(f"  {r.config:<20} {r.metric:<9} {r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]")
    print(f"-> {tables / 'bootstrap.csv'}\n-> {tables / 'bootstrap_differences.csv'}\n"
          f"-> {args.results_dir / 'figures' / 'bootstrap.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
