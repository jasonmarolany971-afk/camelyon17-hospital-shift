"""Calibration: are the predicted probabilities honest, before and after temperature scaling?

Usage (from the repository root):
    python -m src.calibration cnn_none cnn_hed cnn_macenko probe_dino_vitb16 probe_phikon

For every run (configuration, seed), reads artifacts/runs/<name>/seed<N>/predictions/<split>.parquet,
fits one temperature on val only, and writes:
    results/runs/<name>/seed<N>/calibration.json  temperature, ECE and Brier score before and after
                                                   scaling, per split and per hospital
    results/tables/calibration.csv                one row per (configuration, split, hospital, metric):
                                                   mean and sample std over seeds
    results/figures/reliability.png               reliability diagrams before and after scaling

Calibrated: among patches given a tumor probability of about 80%, about 80% are tumor.
Temperature scaling (Guo et al., 2017) divides the logit by one number T > 0. It changes
neither the ranking (AUC), nor the decision at 0.5, nor the threshold chosen on val, so
every metric already reported stays valid. T is fitted on val (hospital 1), never on test:
the ECE on val is therefore optimistic, the honest check is on test and id_val.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

from src.aggregate import SPLITS
from src.compare import LABELS
from src.plotting import COLORS, GRID, INK, INK_SOFT, apply_style, plt
from src.utils import save_json

N_BINS = 15            # equal-width probability bins (Guo et al., 2017)
MIN_BIN_COUNT = 100    # bins with fewer patches are too noisy to draw
EPS = 1e-7             # probabilities are stored in float32: none is closer to 0 or 1 than this
TEMPERATURE_BOUNDS = (0.05, 20.0)
SPLIT_TITLES = {"id_val": "id_val (training hospitals)", "val": "val (hospital 1, T fitted here)",
                "test": "test (hospital 2)"}


def to_logit(prob) -> np.ndarray:
    """Probability of tumor -> logit, log(p / (1 - p)), the difference of the two class logits."""
    p = np.clip(np.asarray(prob, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def apply_temperature(prob, temperature: float) -> np.ndarray:
    return expit(to_logit(prob) / temperature)


def fit_temperature(prob, y) -> float:
    """T minimizing the negative log-likelihood of (prob, y), searched on log T so that T stays positive."""
    z, y = to_logit(prob), np.asarray(y, dtype=np.float64)

    def nll(log_t: float) -> float:
        zt = z / np.exp(log_t)
        return float(np.mean(np.logaddexp(0.0, zt) - y * zt))  # -log sigmoid likelihood, numerically stable

    low, high = np.log(TEMPERATURE_BOUNDS[0]), np.log(TEMPERATURE_BOUNDS[1])
    result = minimize_scalar(nll, bounds=(low, high), method="bounded", options={"xatol": 1e-6})
    if not low + 1e-3 < result.x < high - 1e-3:
        raise ValueError(f"Temperature at the edge of its search range: {np.exp(result.x):.3g}")
    return float(np.exp(result.x))


def ece(prob, y, n_bins: int = N_BINS) -> float:
    """Expected calibration error: |mean predicted - observed tumor share| in each equal-width
    probability bin, weighted by the share of patches in the bin."""
    prob, y = np.asarray(prob, dtype=np.float64), np.asarray(y, dtype=np.float64)
    bins = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    sum_prob = np.bincount(bins, weights=prob, minlength=n_bins)
    sum_y = np.bincount(bins, weights=y, minlength=n_bins)
    # sum over bins of (n_b / N) * |sum_prob_b / n_b - sum_y_b / n_b|
    return float(np.abs(sum_prob - sum_y).sum() / len(prob))


def brier(prob, y) -> float:
    """Mean squared error of the probability: no binning, a complement to the ECE."""
    return float(np.mean((np.asarray(prob, dtype=np.float64) - np.asarray(y)) ** 2))


def reliability_curve(prob, y, n_bins: int = N_BINS) -> pd.DataFrame:
    """One row per non-empty bin: number of patches, mean predicted probability, observed tumor share."""
    frame = pd.DataFrame({"prob": np.asarray(prob, dtype=np.float64), "y": np.asarray(y)})
    frame["bin"] = np.minimum((frame.prob * n_bins).astype(int), n_bins - 1)
    return frame.groupby("bin").agg(count=("y", "size"), mean_prob=("prob", "mean"),
                                    tumor_share=("y", "mean")).reset_index()


def calibrate_run(run_dir: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Fit T on val, then measure calibration before and after on every split and hospital."""
    frames = {split: pd.read_parquet(run_dir / "predictions" / f"{split}.parquet") for split in SPLITS}
    temperature = fit_temperature(frames["val"].prob, frames["val"].y)
    result = {"temperature": temperature, "fitted_on": "val", "n_bins": N_BINS, "splits": {}}
    for split, frame in frames.items():
        frame["prob_scaled"] = apply_temperature(frame.prob, temperature)
        groups = {"all": frame, **{str(h): part for h, part in frame.groupby("hospital")}}
        result["splits"][split] = {
            hospital: {"n": len(part),
                       "ece_raw": ece(part.prob, part.y), "ece_scaled": ece(part.prob_scaled, part.y),
                       "brier_raw": brier(part.prob, part.y), "brier_scaled": brier(part.prob_scaled, part.y)}
            for hospital, part in groups.items()
        }
    return result, frames


def find_runs(artifacts_dir: Path, name: str) -> dict[int, Path]:
    runs = {}
    for folder in (artifacts_dir / name).glob("seed*"):
        match = re.fullmatch(r"seed(\d+)", folder.name)  # skips quick tests such as seed0-subset512
        if match and all((folder / "predictions" / f"{s}.parquet").exists() for s in SPLITS):
            runs[int(match.group(1))] = folder
    return dict(sorted(runs.items()))


def long_records(name: str, seed: int, result: dict) -> list[dict]:
    records = [{"config": name, "seed": seed, "split": "val", "hospital": "all", "metric": "temperature",
                "value": result["temperature"]}]
    for split, hospitals in result["splits"].items():
        for hospital, values in hospitals.items():
            records += [{"config": name, "seed": seed, "split": split, "hospital": hospital, "metric": metric,
                         "value": values[metric]} for metric in ("ece_raw", "ece_scaled", "brier_raw", "brier_scaled")]
    return records


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    grouped = long.groupby(["config", "split", "hospital", "metric"], sort=False)
    return grouped["value"].agg(n_seeds="count", mean="mean", std="std").reset_index()  # std: n - 1


def plot(curves: dict, summary: pd.DataFrame, names: list[str], path: Path) -> None:
    """Rows: configurations; columns: splits. Curves pool the patches of every seed (concatenated,
    never averaged); the ECE written in each panel is the mean over seeds."""
    apply_style()
    fig, axes = plt.subplots(len(names), len(SPLITS), figsize=(9.6, 3.1 * len(names)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for row, name in enumerate(names):
        temperature = summary[(summary.config == name) & (summary.metric == "temperature")].iloc[0]
        for col, split in enumerate(SPLITS):
            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.2, linestyle="--", zorder=1)
            frame = curves[name][split]
            for column, color, label in (("prob", INK_SOFT, "before"), ("prob_scaled", COLORS[0], "after scaling")):
                curve = reliability_curve(frame[column], frame.y)
                curve = curve[curve["count"] >= MIN_BIN_COUNT]
                ax.plot(curve.mean_prob, curve.tumor_share, marker="o", markersize=3.5, linewidth=1.6,
                        color=color, label=label, zorder=3)
            values = summary[(summary.config == name) & (summary.split == split) & (summary.hospital == "all")]
            raw, scaled = (values[values.metric == m].iloc[0] for m in ("ece_raw", "ece_scaled"))
            ax.text(0.97, 0.04, f"ECE {raw['mean']:.3f} → {scaled['mean']:.3f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8.5, color=INK)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, axis="both")
            if row == 0:
                ax.set_title(SPLIT_TITLES[split], fontsize=10)
            if row == len(names) - 1:
                ax.set_xlabel("Predicted tumor probability")
            if col == 0:
                ax.set_ylabel(f"{LABELS.get(name, name).replace(chr(10), ' ')}\nT = {temperature['mean']:.2f}"
                              f" ± {temperature['std']:.2f}\n\nObserved tumor share")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=9, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Reliability: on the dashed diagonal, a predicted 80% means 80% of tumors", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.text(0.01, -0.01, f"One temperature T per run, fitted on val only. {N_BINS} equal-width bins; bins with fewer "
             f"than {MIN_BIN_COUNT} patches are not drawn.\nCurves pool the patches of the 4 seeds; ECE: mean over "
             "seeds, before → after scaling.", color=INK_SOFT, fontsize=8.5, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Temperature scaling on val; calibration before and after.")
    parser.add_argument("names", nargs="+", help="Configuration names, e.g. cnn_none probe_phikon")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    records, curves = [], {}
    for name in args.names:
        runs = find_runs(args.artifacts_dir, name)
        if not runs:
            raise SystemExit(f"No predictions in {args.artifacts_dir / name}")
        pooled = {split: [] for split in SPLITS}
        for seed, run_dir in runs.items():
            result, frames = calibrate_run(run_dir)
            save_json(result, args.results_dir / "runs" / name / f"seed{seed}" / "calibration.json")
            records += long_records(name, seed, result)
            for split in SPLITS:
                pooled[split].append(frames[split][["y", "prob", "prob_scaled"]])
        curves[name] = {split: pd.concat(parts) for split, parts in pooled.items()}

    summary = summarize(pd.DataFrame(records))
    table_path = args.results_dir / "tables" / "calibration.csv"
    summary.to_csv(table_path, index=False)
    figure_path = args.results_dir / "figures" / "reliability.png"
    plot(curves, summary, args.names, figure_path)

    print(f"{'config':<20} {'T (fitted on val)':>18}   ECE before -> after scaling (mean over seeds)")
    for name in args.names:
        part = summary[(summary.config == name) & (summary.hospital == "all")]
        t = part[part.metric == "temperature"].iloc[0]
        eces = "  ".join(f"{split} {part[(part.split == split) & (part.metric == 'ece_raw')]['mean'].iloc[0]:.3f}"
                         f"->{part[(part.split == split) & (part.metric == 'ece_scaled')]['mean'].iloc[0]:.3f}"
                         for split in SPLITS)
        print(f"{name:<20} {t['mean']:>11.2f} ± {t['std']:.2f}   {eces}")
    print(f"-> {table_path}\n-> {figure_path}\n-> results/runs/<name>/seed<N>/calibration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
