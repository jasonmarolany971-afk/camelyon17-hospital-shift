"""Compare several configurations seed by seed, on the same scale.

Usage (from the repository root):
    python -m src.compare cnn_none probe_dino_vitb16 probe_phikon

Reads results/runs/<name>/seed<N>/metrics.json and writes:
    results/tables/comparison.csv   one row per (configuration, split): mean and sample std of
                                    accuracy, AUC, sensitivity/specificity at the val threshold
    results/figures/comparison.png  accuracy and AUC on id_val, val and test: one dot per seed,
                                    a line for the mean, one shared scale per row
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.aggregate import SPLITS, load_runs, long_table
from src.plotting import AXIS, INK, INK_SOFT, apply_style, plt, seed_color

# Short axis labels; the caption below the figure says what each method is.
LABELS = {
    "cnn_none": "CNN\nscratch",
    "cnn_hed": "CNN\n+ HED",
    "cnn_macenko": "CNN\n+ Macenko",
    "probe_dino_vitb16": "DINO\nImageNet",
    "probe_phikon": "Phikon\nhistology",
}
DESCRIPTIONS = {
    "cnn_none": "CNN: DenseNet-121 trained from scratch.",
    "cnn_hed": "+ HED: stain augmentation (training only).",
    "cnn_macenko": "+ Macenko: stain normalization (every patch).",
    "probe_dino_vitb16": "DINO, Phikon: frozen ViT-B/16 pretrained on natural images or on histology, linear layer on top.",
}
DESCRIPTIONS["probe_phikon"] = DESCRIPTIONS["probe_dino_vitb16"]
SPLIT_TITLES = {"id_val": "id_val (training hospitals)", "val": "val (hospital 1)", "test": "test (hospital 2)"}
TABLE_METRICS = ["accuracy", "auc", "sensitivity_at_threshold", "specificity_at_threshold", "predicted_tumor_share"]


def collect(names: list[str], runs_root: Path) -> pd.DataFrame:
    frames = []
    for name in names:
        runs = load_runs(runs_root / name)
        if not runs:
            raise SystemExit(f"No complete run in {runs_root / name}")
        frames.append(long_table(runs).assign(config=name))
    long = pd.concat(frames)
    return long[long.hospital == "all"]


def comparison_table(long: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        for split in SPLITS:
            part = long[(long.config == name) & (long.split == split)]
            row = {"config": name, "split": split, "n_seeds": part.seed.nunique()}
            for metric in TABLE_METRICS:
                values = part[part.metric == metric].value
                row[f"{metric}_mean"], row[f"{metric}_std"] = values.mean(), values.std()  # std: n - 1
            rows.append(row)
    return pd.DataFrame(rows)


def plot(long: pd.DataFrame, names: list[str], path: Path) -> None:
    apply_style()
    metrics = [("accuracy", "Accuracy (%)", 100), ("auc", "AUC", 1)]
    panel_width = 1.15 * len(names) + 0.6  # inches: room for every method label
    fig, axes = plt.subplots(len(metrics), len(SPLITS), figsize=(3 * panel_width, 7.2), sharey="row")
    for row, (metric, ylabel, scale) in enumerate(metrics):
        for col, split in enumerate(SPLITS):
            ax = axes[row, col]
            for x, name in enumerate(names):
                part = long[(long.config == name) & (long.split == split) & (long.metric == metric)]
                offsets = [(i - (len(part) - 1) / 2) * 0.08 for i in range(len(part))]
                for offset, (_, r) in zip(offsets, part.iterrows()):
                    ax.scatter(x + offset, r.value * scale, s=42, color=seed_color(int(r.seed)),
                               edgecolor="white", linewidth=1, zorder=3)
                mean = part.value.mean() * scale
                ax.hlines(mean, x - 0.2, x + 0.2, color=INK, linewidth=2, zorder=4)
                ax.text(x + 0.24, mean, f"{mean:.1f}" if scale == 100 else f"{mean:.3f}",
                        va="center", fontsize=8, color=INK_SOFT)
            ax.set_xticks(range(len(names)), [LABELS.get(n, n) for n in names] if row == len(metrics) - 1 else [])
            ax.set_xlim(-0.6, len(names) - 0.2)
            if row == 0:
                ax.set_title(SPLIT_TITLES[split])
            if col == 0:
                ax.set_ylabel(ylabel)
    seeds = sorted(long.seed.unique())
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=seed_color(int(s)), label=f"seed {s}") for s in seeds]
    handles.append(plt.Line2D([], [], color=INK, linewidth=2, label="mean"))
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), fontsize=9, bbox_to_anchor=(1.0, 1.02))
    fig.suptitle("One dot per seed, same scale for every method", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=1.02)
    descriptions = list(dict.fromkeys(DESCRIPTIONS[n] for n in names if n in DESCRIPTIONS))
    caption = " ".join(descriptions) + "\nEpoch and decision threshold selected on val only; test evaluated once per seed."
    fig.text(0.01, -0.05, caption, color=INK_SOFT, fontsize=9)
    for ax in axes.flat:
        ax.tick_params(axis="x", colors=AXIS, labelcolor=INK_SOFT)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare configurations seed by seed.")
    parser.add_argument("names", nargs="+", help="Configuration names, e.g. cnn_none probe_phikon")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    long = collect(args.names, args.results_dir / "runs")
    table = comparison_table(long, args.names)
    table_path = args.results_dir / "tables" / "comparison.csv"
    table.to_csv(table_path, index=False)
    figure_path = args.results_dir / "figures" / "comparison.png"
    plot(long, args.names, figure_path)

    print(f"{'config':<20} {'split':<7} {'accuracy':>16} {'AUC':>15} {'called tumor':>13}")
    for _, r in table.iterrows():
        print(f"{r.config:<20} {r.split:<7} {r.accuracy_mean:>8.3f} ± {r.accuracy_std:.3f} "
              f"{r.auc_mean:>7.3f} ± {r.auc_std:.3f} {r.predicted_tumor_share_mean:>13.3f}")
    print(f"-> {table_path}\n-> {figure_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
