"""Error galleries for every method, and Grad-CAM for the CNNs.

Usage (from the repository root):
    python -m src.explain cnn_none cnn_hed cnn_macenko probe_dino_vitb16 probe_phikon

Seed 0 of every configuration, fixed in advance (never the seed that looks best on test).

1. Error galleries, per unseen hospital (val = hospital 1, test = hospital 2): the most confident
   false negatives (tumor patches given the lowest tumor probability) and false positives of
   every method, and patches that every method gets wrong. The most confident errors are the
   most dangerous and the most informative, not the most typical.
2. Error overlap: errors of each method, and errors shared by all of them.
3. Grad-CAM (Selvaraju et al., 2017) for the CNNs, on the last DenseNet block. For a 96 x 96 patch
   that block is a 3 x 3 grid of 32-pixel cells, and the label describes exactly the central
   32 x 32 region: maps are drawn as they are (no smoothing, which would invent detail) with the
   label region outlined, and a summary measures the share of the map in the central cell for
   correctly detected tumor patches (a uniform map gives 1/9).

Writes:
    results/figures/errors_val.png, results/figures/errors_test.png
    results/tables/error_overlap.csv, results/tables/shared_errors_by_slide.csv
    results/figures/gradcam.png
    results/tables/gradcam_center.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from src.compare import LABELS
from src.data import build_transform, load_dataset
from src.models import build_model
from src.plotting import INK, INK_SOFT, apply_style, plt

SEED = 0
SPLITS = ["val", "test"]
SPLIT_TITLES = {"val": "val (hospital 1)", "test": "test (hospital 2)"}
N_PER_KIND = 4          # gallery: false negatives and false positives shown per method
N_CAM_TUMOR, N_CAM_NORMAL = 4, 2   # Grad-CAM figure: same patches for every CNN
N_CENTER_SAMPLE = 300   # tumor patches per hospital for the central-cell summary
PATCH, CELL = 96, 32
CAM_COLOR = "#eda100"


def label(name: str) -> str:
    return LABELS.get(name, name).replace("\n", " ")


def load_predictions(runs_dir: Path, name: str, split: str) -> pd.DataFrame:
    path = runs_dir / name / f"seed{SEED}" / "predictions" / f"{split}.parquet"
    return pd.read_parquet(path).sort_values("index").reset_index(drop=True)


def most_confident_errors(frame: pd.DataFrame, kind: str, k: int) -> pd.DataFrame:
    """kind "fn": tumor patches with the lowest probability; "fp": normal patches with the highest."""
    if kind == "fn":
        return frame[(frame.y == 1) & (frame.prob <= 0.5)].nsmallest(k, "prob")
    return frame[(frame.y == 0) & (frame.prob > 0.5)].nlargest(k, "prob")


def shared_errors(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Patches wrong for every method, with the mean probability; most confident first."""
    first = next(iter(frames.values()))
    probs = pd.DataFrame({name: frame.prob.to_numpy() for name, frame in frames.items()})
    for frame in frames.values():
        if not np.array_equal(frame["index"].to_numpy(), first["index"].to_numpy()):
            raise ValueError("Prediction files cover different patches")
    wrong = ((probs > 0.5).to_numpy() != first.y.to_numpy()[:, None]).all(axis=1)
    shared = first[wrong].assign(prob=probs[wrong].mean(axis=1).to_numpy())
    shared["confidence"] = (shared.prob - 0.5).abs()
    return shared.sort_values("confidence", ascending=False)


def overlap_table(frames_by_split: dict) -> pd.DataFrame:
    rows = []
    for split, frames in frames_by_split.items():
        for name, frame in frames.items():
            wrong = (frame.prob > 0.5) != (frame.y == 1)
            rows.append({"split": split, "group": name, "n_patches": len(frame), "errors": int(wrong.sum()),
                         "false_negatives": int((wrong & (frame.y == 1)).sum()),
                         "false_positives": int((wrong & (frame.y == 0)).sum())})
        shared = shared_errors(frames)
        rows.append({"split": split, "group": "all_methods_wrong", "n_patches": len(next(iter(frames.values()))),
                     "errors": len(shared), "false_negatives": int((shared.y == 1).sum()),
                     "false_positives": int((shared.y == 0).sum())})
    return pd.DataFrame(rows)


class GradCAM:
    """Grad-CAM on one layer: each activation map is weighted by the spatial mean of the gradient
    of the class logit with respect to it; the weighted sum is kept where it is positive."""

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model = model.eval()
        self.activations = None
        layer.register_forward_hook(self._keep)

    def _keep(self, module, inputs, output):
        self.activations = output

    def __call__(self, x: torch.Tensor, target: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """(maps of shape batch x h x w, tumor probabilities)."""
        with torch.enable_grad():
            logits = self.model(x)
            (grads,) = torch.autograd.grad(logits[:, target].sum(), self.activations)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        maps = torch.relu((weights * self.activations).sum(dim=1))
        return maps.detach(), torch.softmax(logits.detach(), dim=1)[:, 1]


def load_cnn(artifacts_dir: Path, name: str, device: torch.device):
    checkpoint = torch.load(artifacts_dir / name / f"seed{SEED}" / "best.pt", map_location="cpu")
    model = build_model(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return GradCAM(model, model.features), build_transform(checkpoint["config"], train=False)


def cam_batch(cam: GradCAM, transform, dataset, indices, device) -> tuple[np.ndarray, np.ndarray]:
    x = torch.stack([transform(dataset.get_input(int(i))) for i in indices]).to(device)
    maps, probs = cam(x)
    return maps.cpu().numpy(), probs.cpu().numpy()


def check_probabilities(name: str, probs: np.ndarray, frame: pd.DataFrame, indices) -> None:
    """Recomputed here, the probabilities must match the saved predictions (same model, same preprocessing)."""
    saved = frame.set_index("index").loc[list(indices), "prob"].to_numpy()
    gap = float(np.abs(probs - saved).max())
    if gap > 1e-3:
        raise ValueError(f"{name}: probabilities differ from the saved predictions by {gap:.4f}")


def hide_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_gallery(dataset, frames: dict[str, pd.DataFrame], split: str, path: Path) -> None:
    rows = [(label(name), {"fn": most_confident_errors(f, "fn", N_PER_KIND), "fp": most_confident_errors(f, "fp", N_PER_KIND)})
            for name, f in frames.items()]
    shared = shared_errors(frames)
    rows.append((f"All {len(frames)} wrong\n(mean probability)",
                 {"fn": shared[shared.y == 1].head(N_PER_KIND), "fp": shared[shared.y == 0].head(N_PER_KIND)}))
    n_cols = 2 * N_PER_KIND
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.35 * n_cols + 1.6, 1.55 * len(rows)),
                             gridspec_kw={"wspace": 0.08, "hspace": 0.32})
    for r, (row_label, kinds) in enumerate(rows):
        for kind_number, kind in enumerate(("fn", "fp")):
            patches = kinds[kind]
            for k in range(N_PER_KIND):
                ax = axes[r, kind_number * N_PER_KIND + k]
                hide_axes(ax)
                if k < len(patches):
                    patch = patches.iloc[k]
                    ax.imshow(dataset.get_input(int(patch["index"])))
                    ax.add_patch(Rectangle((CELL - 0.5, CELL - 0.5), CELL, CELL, fill=False, edgecolor="white",
                                           linewidth=0.8, linestyle="--"))
                    ax.set_xlabel(f"slide {int(patch.slide)}  p={patch.prob:.2f}", fontsize=7, color=INK_SOFT,
                                  labelpad=2)
        axes[r, 0].set_ylabel(row_label, rotation=0, ha="right", va="center", fontsize=8.5, color=INK, labelpad=8)
    axes[0, 0].set_title("Missed tumors (most confident false negatives)", loc="left", fontsize=9.5)
    axes[0, N_PER_KIND].set_title("False alarms (most confident false positives)", loc="left", fontsize=9.5)
    fig.suptitle(f"Most confident errors, {SPLIT_TITLES[split]}, seed {SEED}", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.text(0.01, 0.0, "p: predicted tumor probability. Dashed square: the central 32 x 32 region the label "
             "describes. Extreme cases, not typical errors.", color=INK_SOFT, fontsize=8, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_gradcam(dataset, panels: dict, path: Path) -> None:
    """panels[split] = {"indices": [...], "labels": [...], "maps": {name: maps}, "probs": {name: probs}}."""
    names = list(next(iter(panels.values()))["maps"])
    n_rows = len(panels) * (1 + len(names))
    n_cols = N_CAM_TUMOR + N_CAM_NORMAL
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.45 * n_cols + 1.6, 1.6 * n_rows),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.45})
    row = 0
    for split, panel in panels.items():
        images = [dataset.get_input(int(i)) for i in panel["indices"]]
        for c, (image, truth) in enumerate(zip(images, panel["labels"])):
            hide_axes(axes[row, c])
            axes[row, c].imshow(image)
            axes[row, c].set_title("tumor" if truth else "normal", fontsize=8, loc="center")
        axes[row, 0].set_ylabel(f"{SPLIT_TITLES[split]}\noriginal", rotation=0, ha="right", va="center", fontsize=8.5,
                                color=INK, labelpad=8)
        row += 1
        for name in names:
            for c, image in enumerate(images):
                ax = axes[row, c]
                hide_axes(ax)
                cam = panel["maps"][name][c]
                ax.imshow(image)
                note = ""
                if cam.max() > 0:  # amber, more opaque where the map is high; each map scaled to its own maximum
                    overlay = np.zeros(cam.shape + (4,))
                    overlay[..., :3] = to_rgb(CAM_COLOR)
                    overlay[..., 3] = 0.75 * cam / cam.max()
                    ax.imshow(overlay, interpolation="nearest", extent=(-0.5, PATCH - 0.5, PATCH - 0.5, -0.5))
                else:
                    note = ", map = 0"  # no cell pushes towards 'tumor'
                ax.add_patch(Rectangle((CELL - 0.5, CELL - 0.5), CELL, CELL, fill=False, edgecolor="white",
                                       linewidth=0.8, linestyle="--"))
                ax.set_xlabel(f"p={panel['probs'][name][c]:.2f}{note}", fontsize=7, color=INK_SOFT, labelpad=2)
            axes[row, 0].set_ylabel(label(name), rotation=0, ha="right", va="center", fontsize=8.5, color=INK,
                                    labelpad=8)
            row += 1
    fig.suptitle(f"Grad-CAM for the tumor class, last DenseNet block (3 x 3 cells), seed {SEED}", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.text(0.01, 0.0, "Amber: cells whose activations push towards 'tumor' (more opaque = stronger; each map scaled to "
             "its own maximum, no smoothing). Each cell's activations\nalso see beyond the cell: deep layers have large "
             "receptive fields. Dashed square: the central 32 x 32 region the label describes. Same random "
             "patches for every CNN; p: predicted tumor probability.", color=INK_SOFT, fontsize=8, va="top")
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Error galleries and Grad-CAM (seed 0).")
    parser.add_argument("names", nargs="+", help="Configuration names; CNNs (cnn_*) also get Grad-CAM")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("results/runs"), help="Saved predictions")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/runs"), help="CNN weights (Grad-CAM)")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM (needs the CNN weights)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cpu", help="cpu by default: small job, reproducible")
    args = parser.parse_args()

    apply_style()
    dataset = load_dataset(args.data_root)
    figures, tables = args.results_dir / "figures", args.results_dir / "tables"
    frames_by_split = {split: {name: load_predictions(args.runs_dir, name, split) for name in args.names}
                       for split in SPLITS}

    overlap = overlap_table(frames_by_split)
    overlap.to_csv(tables / "error_overlap.csv", index=False)
    by_slide = pd.concat([shared_errors(frames).assign(split=split) for split, frames in frames_by_split.items()])
    by_slide = by_slide.groupby(["split", "slide"]).agg(shared_false_negatives=("y", "sum"), shared_errors=("y", "size"))
    by_slide.reset_index().to_csv(tables / "shared_errors_by_slide.csv", index=False)
    for split, frames in frames_by_split.items():
        plot_gallery(dataset, frames, split, figures / f"errors_{split}.png")

    cnns = [] if args.no_gradcam else [name for name in args.names if name.startswith("cnn_")]
    device = torch.device(args.device)
    rng = np.random.default_rng(SEED)
    panels, center_rows = {}, []
    models = {name: load_cnn(args.artifacts_dir, name, device) for name in cnns}
    for split in SPLITS:
        reference = next(iter(frames_by_split[split].values()))
        tumor = rng.choice(reference[reference.y == 1]["index"].to_numpy(), N_CAM_TUMOR + N_CENTER_SAMPLE, replace=False)
        normal = rng.choice(reference[reference.y == 0]["index"].to_numpy(), N_CAM_NORMAL, replace=False)
        shown = list(tumor[:N_CAM_TUMOR]) + list(normal)
        panels[split] = {"indices": shown, "labels": [1] * N_CAM_TUMOR + [0] * N_CAM_NORMAL, "maps": {}, "probs": {}}
        for name, (cam, transform) in models.items():
            maps, probs = cam_batch(cam, transform, dataset, shown, device)
            check_probabilities(name, probs, frames_by_split[split][name], shown)
            panels[split]["maps"][name], panels[split]["probs"][name] = maps, probs
            sample = tumor[N_CAM_TUMOR:]
            maps, probs = zip(*(cam_batch(cam, transform, dataset, sample[i:i + 50], device)
                                for i in range(0, len(sample), 50)))
            maps, probs = np.concatenate(maps), np.concatenate(probs)
            check_probabilities(name, probs, frames_by_split[split][name], sample)
            detected = (probs > 0.5) & (maps.sum(axis=(1, 2)) > 0)
            share = maps[detected, 1, 1] / maps[detected].sum(axis=(1, 2))
            center_rows.append({"config": name, "split": split, "seed": SEED, "n_tumor_patches": len(sample),
                                "n_detected": int(detected.sum()), "center_share_mean": float(share.mean()),
                                "center_share_median": float(np.median(share)), "uniform_share": 1 / 9})
    if cnns:
        plot_gradcam(dataset, panels, figures / "gradcam.png")
        pd.DataFrame(center_rows).to_csv(tables / "gradcam_center.csv", index=False)

    print(overlap.to_string(index=False))
    if center_rows:
        print(pd.DataFrame(center_rows)[["config", "split", "n_detected", "center_share_mean"]].to_string(index=False))
    print(f"-> {figures}/errors_val.png, errors_test.png, gradcam.png\n"
          f"-> {tables}/error_overlap.csv, gradcam_center.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
