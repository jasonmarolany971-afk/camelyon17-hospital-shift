"""Abstention: how much does the error drop if the model may refer its least confident patches?

Usage (from the repository root):
    python -m src.abstention cnn_none cnn_hed cnn_macenko probe_dino_vitb16 probe_phikon

Confidence of a patch: distance of its tumor probability to 0.5 (the decision is the official
one, probability > 0.5). Patches are sorted from most to least confident; coverage is the share
the model answers, risk the error rate among them. Temperature scaling does not change this order,
so these results hold before and after calibration.

Referral cutoffs are chosen on val only, by two rules ("answer X% of val patches" or "at most Y%
error on val"), then applied unchanged to id_val and test to see whether they transfer.

Reads results/runs/<name>/seed<N>/predictions/<split>.parquet and writes:
    results/tables/risk_coverage.csv       AURC and E-AURC per (configuration, split): mean and std over seeds
    results/tables/abstention_cutoffs.csv  coverage and risk obtained with each val-chosen cutoff
    results/figures/risk_coverage.png      risk-coverage curves on id_val, val and test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.aggregate import SPLITS
from src.calibration import find_runs
from src.compare import LABELS
from src.plotting import CONFIG_COLORS, INK, INK_SOFT, SURFACE, apply_style, plt

COVERAGE_TARGETS = [0.9, 0.8, 0.7]   # "answer this share of val patches"
RISK_TARGETS = [0.05, 0.02, 0.01]    # "at most this error rate on val"
GRID = np.linspace(0.01, 1.0, 100)   # coverages at which curves are averaged over seeds
Y_MAX = 14                           # % : axis cut so that the good models stay readable; curves above are labeled
SPLIT_TITLES = {"id_val": "id_val (training hospitals)", "val": "val (hospital 1, cutoffs chosen here)",
                "test": "test (hospital 2)"}


def confidence(prob) -> np.ndarray:
    return np.abs(np.asarray(prob, dtype=np.float64) - 0.5)


def errors(prob, y) -> np.ndarray:
    return ((np.asarray(prob) > 0.5) != np.asarray(y).astype(bool)).astype(np.float64)


def risk_curve(prob, y) -> np.ndarray:
    """risk[k - 1]: error rate among the k most confident patches, k = 1..n."""
    order = np.argsort(-confidence(prob), kind="stable")
    return np.cumsum(errors(prob, y)[order]) / np.arange(1, len(order) + 1)


def aurc(prob, y) -> tuple[float, float]:
    """(AURC, E-AURC): area under the risk-coverage curve, and its excess over an oracle
    that ranks every error last (removes the effect of the overall accuracy)."""
    risk = risk_curve(prob, y)
    n, n_errors = len(risk), errors(prob, y).sum()
    k = np.arange(1, n + 1)
    oracle = np.maximum(0.0, k - (n - n_errors)) / k
    return float(risk.mean()), float(risk.mean() - oracle.mean())


def risk_at(prob, y, coverages) -> np.ndarray:
    """Risk when answering the given shares of patches (most confident first)."""
    risk = risk_curve(prob, y)
    k = np.clip(np.ceil(np.asarray(coverages) * len(risk)).astype(int), 1, len(risk))
    return risk[k - 1]


def cutoff_for_coverage(prob, target: float) -> float:
    """Confidence cutoff such that about `target` of these patches have confidence >= cutoff."""
    ranked = np.sort(confidence(prob))[::-1]
    return float(ranked[max(int(np.ceil(target * len(ranked))) - 1, 0)])


def cutoff_for_risk(prob, y, target: float) -> float:
    """Lowest cutoff (largest coverage) whose error rate among answered patches is <= target.
    np.inf (answer nothing) if even the most confident patch is wrong too often."""
    risk = risk_curve(prob, y)
    ok = np.flatnonzero(risk <= target)
    if len(ok) == 0:
        return float(np.inf)
    ranked = np.sort(confidence(prob))[::-1]
    return float(ranked[ok[-1]])


def apply_cutoff(prob, y, cutoff: float) -> tuple[float, float]:
    """(coverage, risk) when answering only patches with confidence >= cutoff (risk NaN if none)."""
    answered = confidence(prob) >= cutoff
    risk = float(errors(prob, y)[answered].mean()) if answered.any() else float("nan")
    return float(answered.mean()), risk


def analyze_run(run_dir: Path) -> tuple[list[dict], list[dict], dict]:
    frames = {split: pd.read_parquet(run_dir / "predictions" / f"{split}.parquet") for split in SPLITS}
    curve_rows, cutoff_rows, curves = [], [], {}
    for split, frame in frames.items():
        value, excess = aurc(frame.prob, frame.y)
        curve_rows.append({"split": split, "aurc": value, "e_aurc": excess,
                           "risk_full_coverage": float(errors(frame.prob, frame.y).mean())})
        curves[split] = risk_at(frame.prob, frame.y, GRID)
    val = frames["val"]
    rules = [("coverage", t, cutoff_for_coverage(val.prob, t)) for t in COVERAGE_TARGETS]
    rules += [("risk", t, cutoff_for_risk(val.prob, val.y, t)) for t in RISK_TARGETS]
    for rule, target, cutoff in rules:
        for split, frame in frames.items():
            coverage, risk = apply_cutoff(frame.prob, frame.y, cutoff)
            cutoff_rows.append({"rule": rule, "target": target, "cutoff_confidence": cutoff,
                                "split": split, "coverage": coverage, "risk": risk})
    return curve_rows, cutoff_rows, curves


def plot(curves: dict, cutoffs: pd.DataFrame, names: list[str], path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(13, 4.4), sharey=True)
    for ax, split in zip(axes, SPLITS):
        clipped = []
        for name in names:
            mean_curve = np.mean(curves[name][split], axis=0) * 100
            label = LABELS.get(name, name).replace("\n", " ")
            ax.plot(GRID * 100, mean_curve, color=CONFIG_COLORS.get(name, INK), linewidth=2, label=label)
            if mean_curve.max() > Y_MAX:  # leaves the frame: say where it goes
                clipped.append((label, CONFIG_COLORS.get(name, INK), mean_curve[-1]))
            point = cutoffs[(cutoffs.config == name) & (cutoffs.rule == "coverage") & (cutoffs.target == 0.8)
                            & (cutoffs.split == split)]
            ax.scatter(point.coverage_mean * 100, point.risk_mean * 100, s=40, color=CONFIG_COLORS.get(name, INK),
                       edgecolor=INK, linewidth=0.8, zorder=4)
        for i, (label, color, full) in enumerate(clipped):
            ax.text(0.02, 0.97 - 0.07 * i, f"\u2191 {label}: {full:.0f}% at full coverage", transform=ax.transAxes,
                    color=color, fontsize=7.5, ha="left", va="top",
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9, "pad": 1})
        ax.axvline(80, color=INK_SOFT, linewidth=0.8, linestyle=":")
        ax.set_title(SPLIT_TITLES[split], fontsize=10)
        ax.set_xlabel("Coverage: share of patches the model answers (%)")
        ax.set_xlim(0, 101)
        ax.set_ylim(0, Y_MAX)
        ax.grid(True, axis="both")
    axes[0].set_ylabel("Risk: error rate among answered patches (%)")
    axes[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Referring the least confident patches: how fast does the error drop?", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.text(0.01, -0.02, "Lines: mean over 4 seeds. Dots: cutoff chosen on val to answer 80% of val patches, applied "
             "unchanged to the other hospitals (dotted line: 80%).\nConfidence = distance of the tumor probability "
             "to 0.5; decision at 0.5. Temperature scaling does not change these curves. Axis cut at "
             f"{Y_MAX}%: curves above it are labeled.",
             color=INK_SOFT, fontsize=8.5, va="top")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Risk-coverage curves and val-chosen referral cutoffs.")
    parser.add_argument("names", nargs="+", help="Configuration names, e.g. cnn_none probe_phikon")
    parser.add_argument("--runs-dir", type=Path, default=Path("results/runs"), help="Saved predictions")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    curve_rows, cutoff_rows, curves = [], [], {}
    for name in args.names:
        runs = find_runs(args.runs_dir, name)
        if not runs:
            raise SystemExit(f"No predictions in {args.runs_dir / name}")
        curves[name] = {split: [] for split in SPLITS}
        for seed, run_dir in runs.items():
            rows, cuts, run_curves = analyze_run(run_dir)
            curve_rows += [{"config": name, "seed": seed, **r} for r in rows]
            cutoff_rows += [{"config": name, "seed": seed, **r} for r in cuts]
            for split in SPLITS:
                curves[name][split].append(run_curves[split])

    keys = ["config", "split"]
    curve_table = pd.DataFrame(curve_rows).groupby(keys, sort=False).agg(
        n_seeds=("seed", "count"), aurc_mean=("aurc", "mean"), aurc_std=("aurc", "std"),
        e_aurc_mean=("e_aurc", "mean"), e_aurc_std=("e_aurc", "std"),
        risk_full_coverage_mean=("risk_full_coverage", "mean")).reset_index()  # std: n - 1
    keys = ["config", "rule", "target", "split"]
    cutoff_table = pd.DataFrame(cutoff_rows).groupby(keys, sort=False).agg(
        n_seeds=("seed", "count"), coverage_mean=("coverage", "mean"), coverage_std=("coverage", "std"),
        risk_mean=("risk", "mean"), risk_std=("risk", "std"),
        n_seeds_answering=("risk", "count")).reset_index()  # risk is undefined for a seed that answers nothing
    tables = args.results_dir / "tables"
    curve_table.to_csv(tables / "risk_coverage.csv", index=False)
    cutoff_table.to_csv(tables / "abstention_cutoffs.csv", index=False)
    figure_path = args.results_dir / "figures" / "risk_coverage.png"
    plot(curves, cutoff_table, args.names, figure_path)

    print("Error at full coverage -> with the cutoff chosen on val to answer 80% of val patches (coverage)")
    for name in args.names:
        parts = []
        for split in SPLITS:
            full = curve_table[(curve_table.config == name) & (curve_table.split == split)].risk_full_coverage_mean.iloc[0]
            r = cutoff_table[(cutoff_table.config == name) & (cutoff_table.rule == "coverage")
                             & (cutoff_table.target == 0.8) & (cutoff_table.split == split)].iloc[0]
            parts.append(f"{split} {full:.1%} -> {r.risk_mean:.1%} ({r.coverage_mean:.0%})")
        print(f"  {name:<20} " + " | ".join(parts))
    print(f"-> {tables / 'risk_coverage.csv'}\n-> {tables / 'abstention_cutoffs.csv'}\n-> {figure_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
