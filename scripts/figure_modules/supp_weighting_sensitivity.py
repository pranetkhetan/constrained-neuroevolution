"""
Supplementary S6b — Fitness weighting sensitivity (Supplementary Figure supp_s_weighting).

Two-panel layout:
  A: Mean specialisation index per weighting scheme (5 bars; Published highlighted).
  B: Per-mouse specialisation index across all five schemes (grouped bars).

All quantities are specialisation INDEX = 1 - (own/cross ratio).
Dashed reference at 0 (no specialisation); positive = agent specialised.

Weighting schemes:
  Published 2:2:1:1    — main analysis
  Equal 1:1:1:1        — uniform baseline
  Turn-dominant 1:1:1:3 — maximum weight on the most individually-specific metric
  Markov-only 1:0:0:0  — single metric extreme
  Turn+Occupancy 0:2:0:1 — exclude Markov, probe complementary subset

Data source: analysis/cross_mouse_per_metric.pkl
Output:      figures/supp_s_weighting.pdf  (+ .png)
"""

import os
import sys
import pickle
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figure_modules._style import (
    pub_despine, save_figure, label_panel,
    FS_LABEL, FS_TICK, FS_ANNOT, FS_LEGEND,
    MICE, MOUSE_COLORS, EVOLVED_COL,
)

# ---------------------------------------------------------------------------
# Weighting schemes — (display name, weights in [markov, occupancy, tortuosity, turn_bias] order)
# ---------------------------------------------------------------------------
SCHEMES: List[Tuple[str, List[float]]] = [
    ("Published\n2:2:1:1",       [2.0, 2.0, 1.0, 1.0]),
    ("Equal\n1:1:1:1",           [1.0, 1.0, 1.0, 1.0]),
    ("Turn-dominant\n1:1:1:3",   [1.0, 1.0, 1.0, 3.0]),
    ("Markov-only\n1:0:0:0",     [1.0, 0.0, 0.0, 0.0]),
    ("Turn+Occupancy\n0:2:0:1",  [0.0, 2.0, 0.0, 1.0]),
]
COMPONENTS = ["markov", "occupancy", "tortuosity", "turn_bias"]

# Colour for each scheme (Published gets EVOLVED_COL; others neutral grey variants)
SCHEME_COLORS = [
    EVOLVED_COL,   # Published — highlighted
    "#888888",
    "#AAAAAA",
    "#BBBBBB",
    "#CCCCCC",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("cupy"):
            module = module.replace("cupy._core.core", "numpy").replace("cupy", "numpy")
        return super().find_class(module, name)


def _load(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (ModuleNotFoundError, AttributeError):
        with open(path, "rb") as f:
            return _CpuUnpickler(f).load()


def _weighted_matrix(comps: Dict[str, np.ndarray], weights: List[float]) -> np.ndarray:
    total = sum(weights)
    assert total > 0
    mat = np.zeros_like(next(iter(comps.values())), dtype=float)
    for w, key in zip(weights, COMPONENTS):
        if w > 0:
            mat += w * comps[key]
    return mat / total


def _spec_index(mat: np.ndarray) -> float:
    d = np.diag(mat)
    o = mat[~np.eye(mat.shape[0], dtype=bool)]
    return float(1.0 - d.mean() / o.mean())


def _per_mouse_indices(mat: np.ndarray, mice: List[str]) -> Dict[str, float]:
    n = mat.shape[0]
    return {
        m: float(1.0 - mat[i, i] / np.delete(mat[i, :], i).mean())
        for i, m in enumerate(mice)
    }


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _overall_panel(ax, overall_indices: List[float], scheme_names: List[str]) -> None:
    """Panel A: mean specialisation index per scheme."""
    y_pos = np.arange(len(scheme_names))
    bars = ax.barh(
        y_pos, overall_indices,
        color=SCHEME_COLORS, height=0.55, edgecolor="white", linewidth=0.6,
    )
    ax.axvline(0.0, color="#444444", lw=0.8, ls="--", label="No specialisation")

    for bar, val in zip(bars, overall_indices):
        sign = 1 if val >= 0 else -1
        ax.text(
            val + sign * 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right",
            fontsize=FS_ANNOT, fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(scheme_names, fontsize=FS_TICK)
    ax.set_xlabel("Mean specialisation index\n(1 − own/cross fitness)", fontsize=FS_LABEL)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_xlim(-0.05, max(overall_indices) * 1.35)
    pub_despine(ax)


def _per_mouse_panel(ax, per_mouse_all: List[Dict[str, float]],
                     scheme_names: List[str], mice: List[str]) -> None:
    """Panel B: per-mouse specialisation index, grouped by mouse."""
    n_schemes = len(scheme_names)
    n_mice = len(mice)
    x = np.arange(n_mice)
    width = 0.75 / n_schemes

    for si, (scheme_pm, color) in enumerate(zip(per_mouse_all, SCHEME_COLORS)):
        vals = [scheme_pm[m] for m in mice]
        offset = (si - n_schemes / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width=width * 0.88,
            color=color, edgecolor="white", linewidth=0.3,
            label=scheme_names[si].replace("\n", " "),
        )

    ax.axhline(0.0, color="#444444", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(mice, fontsize=FS_TICK, rotation=30, ha="right")
    ax.set_ylabel("Specialisation index\n(1 − own/cross fitness)", fontsize=FS_LABEL)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.legend(frameon=False, fontsize=FS_LEGEND, loc="upper right",
              ncol=1, bbox_to_anchor=(1.0, 1.0))
    pub_despine(ax)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate(store, figures_dir: str) -> list:
    analysis_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "analysis", "cross_mouse_per_metric.pkl",
    )
    data = _load(analysis_path)
    comps = data["component_matrices"]
    mice = MICE

    scheme_names = [s[0] for s in SCHEMES]
    overall_indices = []
    per_mouse_all = []
    for _, weights in SCHEMES:
        mat = _weighted_matrix(comps, weights)
        overall_indices.append(_spec_index(mat))
        per_mouse_all.append(_per_mouse_indices(mat, mice))

    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(
        1, 2, wspace=0.42,
        left=0.12, right=0.97, top=0.93, bottom=0.15,
        width_ratios=[0.9, 1.4],
    )
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    _overall_panel(ax_a, overall_indices, scheme_names)
    _per_mouse_panel(ax_b, per_mouse_all, scheme_names, mice)

    for ax, lbl in [(ax_a, "A"), (ax_b, "B")]:
        label_panel(ax, lbl)

    out = os.path.join(figures_dir, "supp_s_weighting.pdf")
    return save_figure(fig, out)


if __name__ == "__main__":
    _figures_dir = os.path.join(_PROJECT, "figures")
    generate(None, _figures_dir)
