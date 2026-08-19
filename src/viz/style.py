"""Consistent plot styling and one place that knows where figures go.

Import `apply_style()` once per script and `save_fig()` for every figure.
The point is not prettiness: consistent styling across 20 charts is what
makes a dashboard and a README look like one project rather than twenty
notebooks stapled together.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from src.config import PATHS

# Colour-blind safe. Used positionally, so the same index means the same
# thing across charts.
PALETTE = ["#3B6EA5", "#D97706", "#0F766E", "#B91C1C", "#6D28D9", "#64748B"]
GRID = "#E2E8F0"
INK = "#1F2937"


def apply_style() -> None:
    """Set global rcParams. Call once at the top of a script."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "medium",
            "axes.labelsize": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
        }
    )


def save_fig(fig: plt.Figure, name: str, subdir: str = "") -> Path:
    """Save to reports/figures/<subdir>/<name>.png and return the path."""
    out_dir = PATHS["figures"] / subdir if subdir else PATHS["figures"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def annotate_bars(ax: plt.Axes, fmt: str = "{:.0f}", pad: int = 3) -> None:
    """Write each bar's value at its end. Removes the need to read gridlines."""
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=pad, fontsize=9)


__all__ = ["apply_style", "save_fig", "annotate_bars", "PALETTE", "GRID", "INK"]