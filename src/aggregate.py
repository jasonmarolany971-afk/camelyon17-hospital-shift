"""Aggregate the results of several seeds of one configuration.

Usage (from the repository root):
    python -m src.aggregate cnn_none

Reads results/runs/<name>/seed<N>/metrics.json and history.csv and writes:
    results/tables/summary_<name>.csv   one row per (split, hospital, metric):
                                        mean, standard deviation, min, max and per-seed values
    results/figures/curves_<name>.png   id_val and val accuracy after every epoch, one line per seed

The standard deviation is the sample one (divided by n - 1), the usual
estimator for a few random seeds. Metrics are aggregated seed by seed: the
probabilities of different seeds are never averaged (that would be an
ensemble, a different method).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from src.plotting import INK_SOFT, apply_style, plt, seed_color

SPLITS = ["id_val", "val", "test"]
METRICS = ["accuracy", "auc", "sensitivity_at_threshold", "specificity_at_threshold"]


def load_runs(runs_dir: Path) -> dict[int, dict]:
    """Seed -> {"metrics": dict, "history": DataFrame}, for every complete run."""
    runs = {}
    for folder in runs_dir.glob("seed*"):
        match = re.fullmatch(r"seed(\d+)", folder.name)  # skips quick tests such as seed0-subset512
        if match and (folder / "metrics.json").exists():
            runs[int(match.group(1))] = {
                "metrics": json.loads((folder / "metrics.json").read_text()),
                "history": pd.read_csv(folder / "history.csv"),
            }
    return dict(sorted(runs.items()))


def long_table(runs: dict[int, dict]) -> pd.DataFrame:
    """One row per (seed, split, hospital, metric)."""
    records = []
    for seed, run in runs.items():
        metrics = run["metrics"]
        for split in SPLITS:
            entry = metrics["splits"][split]
            for hospital, values in {"all": entry, **entry["by_hospital"]}.items():
                for metric in METRICS:
                    records.append({"seed": seed, "split": split, "hospital": str(hospital),
                                    "metric": metric, "value": values[metric]})
        # How the model was selected on val, kept next to the results it produced.
        for metric, value in (("selected_epoch", metrics["selected_epoch"]), ("threshold", metrics["threshold"]["value"])):
            records.append({"seed": seed, "split": "val", "hospital": "all", "metric": metric, "value": value})
    return pd.DataFrame(records)


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    grouped = long.groupby(["split", "hospital", "metric"], sort=False)
    summary = grouped["value"].agg(n_seeds="count", mean="mean", std="std", min="min", max="max")  # std: n - 1
    summary["values"] = grouped.apply(
        lambda g: ";".join(f"seed{s}={v:.4f}" for s, v in zip(g.seed, g.value)), include_groups=False
    )
    return summary.reset_index()


def plot_curves(runs: dict[int, dict], title: str, path: Path) -> None:
    apply_style()
    panels = [("id_val_acc", "id_val: training hospitals, same slides"), ("val_acc", "val: hospital 1, never seen")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (column, panel_title) in zip(axes, panels):
        for seed, run in runs.items():
            history, color = run["history"], seed_color(seed)
            ax.plot(history.epoch, history[column] * 100, color=color, linewidth=2, marker="o", markersize=4,
                    label=f"seed {seed}")
            selected = run["metrics"]["selected_epoch"]
            ax.scatter(selected, history.loc[history.epoch == selected, column] * 100, s=110,
                       facecolor="none", edgecolor=color, linewidth=2, zorder=4)
        ax.set_title(panel_title)
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(int(max(r["history"].epoch.max() for r in runs.values())) + 1))
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="lower left", fontsize=9)  # id_val stays high: its lower part is empty
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.02)
    fig.text(0.01, -0.04, "Circle: epoch selected on val (highest val accuracy).", color=INK_SOFT, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def print_summary(name: str, summary: pd.DataFrame, seeds: list[int]) -> None:
    print(f"{name}: {len(seeds)} seeds ({', '.join(map(str, seeds))}); mean ± sample std [min, max]")
    rows = summary[summary.hospital == "all"].set_index(["split", "metric"])
    labels = {"accuracy": "accuracy", "auc": "AUC", "sensitivity_at_threshold": "sens@thr",
              "specificity_at_threshold": "spec@thr"}
    for split in SPLITS:
        cells = []
        for metric, label in labels.items():
            r = rows.loc[(split, metric)]
            cells.append(f"{label} {r['mean']:.3f} ± {r['std']:.3f} [{r['min']:.3f}, {r['max']:.3f}]")
        print(f"  {split:<7} " + " | ".join(cells))
    hospitals = summary[(summary.split == "id_val") & (summary.hospital != "all") & (summary.metric == "accuracy")]
    print("  id_val accuracy by hospital: " + ", ".join(
        f"H{r.hospital} {r['mean']:.3f} ± {r['std']:.3f}" for _, r in hospitals.iterrows()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate the results of several seeds.")
    parser.add_argument("names", nargs="+", help="Configuration names, e.g. cnn_none")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    for name in args.names:
        runs = load_runs(args.results_dir / "runs" / name)
        if not runs:
            print(f"No complete run in {args.results_dir / 'runs' / name}")
            return 1
        summary = summarize(long_table(runs))
        table_path = args.results_dir / "tables" / f"summary_{name}.csv"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(table_path, index=False)
        figure_path = args.results_dir / "figures" / f"curves_{name}.png"
        plot_curves(runs, f"{name}: accuracy after every epoch, one line per seed", figure_path)
        print_summary(name, summary, list(runs))
        print(f"  -> {table_path}\n  -> {figure_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
