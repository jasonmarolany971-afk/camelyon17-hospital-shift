"""Shared figure style: one look for every figure of the project."""

import matplotlib

matplotlib.use("Agg")  # files only, no window: works on a server or in a script
import matplotlib.pyplot as plt  # noqa: E402

INK, INK_SOFT, AXIS, GRID, SURFACE = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9", "#fcfcfb"
# Categorical colors in a fixed order (validated for color-vision deficiencies).
# A seed always gets the same color: COLORS[seed], never its rank in a plot.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# Figures with one line per configuration (no seeds shown): a fixed color per configuration.
CONFIG_COLORS = {"cnn_none": INK_SOFT, "cnn_hed": "#eda100", "cnn_macenko": "#2a78d6",
                 "probe_dino_vitb16": "#1baf7a", "probe_phikon": "#e87ba4"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_SOFT, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_SOFT, "ytick.labelcolor": INK_SOFT,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def seed_color(seed: int) -> str:
    if seed >= len(COLORS):
        raise ValueError(f"No fixed color for seed {seed}: add one to COLORS rather than cycling.")
    return COLORS[seed]
