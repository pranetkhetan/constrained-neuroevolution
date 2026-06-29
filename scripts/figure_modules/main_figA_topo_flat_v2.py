"""
paper_v2 Fig A — Topology-behaviour degeneracy (§2.3) + MI analysis (§2.4).

5-panel layout:
  A: blank placeholder — fill with imshow() before saving (external schematic from Inkscape)
  B: 54×54 topology Jaccard distance heatmap, agents ordered by mouse group
  C: Joint KDE — topology distance vs behavioural distance, with top and right marginals split as Within vs between distance KDEs
  D: NMI bar chart — all three structural axes simultaneously (§2.4)

Panel C is a matplotlib subfigure so its internal GridSpec
is self-contained with no alignment issues across the outer grid.

Output: figA_topo_flat_v2.pdf

API
---
create_figure(store)          -> (fig, axes_dict)   # for further editing before save
generate(store, figures_dir)  -> [paths]            # standard pipeline entry point

Panel A workflow (external schematic):
    fig, axes = create_figure({})
    img = plt.imread("schematic.png")
    axes["A"].imshow(img)
    # axis is already off; imshow shows without tick marks
    fig.savefig(...)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import seaborn as sns
import pickle
from scipy.stats import spearmanr, mannwhitneyu

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figure_modules._style import (
    apply_pub_style, pub_despine, save_figure, label_panel,
    FIGSIZE,
    FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    MICE,
    LW_SCALE, MARKER_SCALE,
)
from scripts.figure_modules.fig_method_topology_schematic import draw_into_spec

# Pair-type colours — shared across all panels for visual consistency.
WITHIN_COL  = "#CC6677"   # rose
BETWEEN_COL = "#332288"   # indigo


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_pkl(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        class _U(pickle.Unpickler):
            def find_class(self, mod, name):
                if mod.startswith("cupy"):
                    mod = mod.replace("cupy._core.core", "numpy").replace("cupy", "numpy")
                elif mod == "core" or mod.startswith("core."):
                    mod = "numpy." + mod
                return super().find_class(mod, name)
        with open(path, "rb") as f:
            return _U(f).load()


def _load_all(analysis_dir: str) -> tuple[dict, dict, dict]:
    load = lambda name: _load_pkl(os.path.join(analysis_dir, name))
    return load("A1_results.pkl"), load("A3_results.pkl"), load("A5_results.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Panel A — blank schematic placeholder
# ─────────────────────────────────────────────────────────────────────────────

def _blank_panel(ax: plt.Axes) -> None:
    """Blank axis for external schematic. User calls imshow() on axes['A'] before saving."""
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────────────
# Panel B — 54×54 topology Jaccard heatmap
# ─────────────────────────────────────────────────────────────────────────────

def _topo_heatmap_panel(ax: plt.Axes, A1: dict) -> None:
    from scipy.spatial.distance import squareform
    from collections import Counter

    topo_dists   = np.array(A1["topo_dists"])
    agent_labels = list(A1["agent_labels"])

    mat   = squareform(topo_dists)
    order = np.argsort([MICE.index(m) if m in MICE else 99 for m in agent_labels])
    mat_ord  = mat[np.ix_(order, order)]
    mice_ord = [agent_labels[i] for i in order]

    im = ax.imshow(mat_ord, aspect="equal", cmap="GnBu",
                   vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Jaccard distance")

    counts = Counter(mice_ord)
    pos, ticks, tlabels = 0, [], []
    for m in MICE:
        if m in counts:
            ticks.append(pos + counts[m] / 2 - 0.5)
            tlabels.append(m)
            ax.axhline(pos - 0.5, color="white", lw=1.0 * LW_SCALE, alpha=0.8)
            ax.axvline(pos - 0.5, color="white", lw=1.0 * LW_SCALE, alpha=0.8)
            pos += counts[m]

    ax.set_yticks(ticks)
    ax.set_yticklabels(tlabels, fontsize=FS_TICK)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabels, fontsize=FS_TICK, rotation=45, ha="right")
    ax.set_xlabel("Agent (grouped by mouse)", fontsize=FS_LABEL)
    ax.set_ylabel("Agent (grouped by mouse)", fontsize=FS_LABEL)
    pub_despine(ax)


# ─────────────────────────────────────────────────────────────────────────────
# Panel C — joint KDE with top and right marginals as split within vs between KDE curves
# Built inside a subfigure; axes created externally and passed in.
# ─────────────────────────────────────────────────────────────────────────────

def _joint_kde_panel(
    ax_main:  plt.Axes,
    ax_top:   plt.Axes,
    ax_right: plt.Axes,
    A1: dict,
    A5: dict,
) -> None:
    topo_dists = np.array(A1["topo_dists"])
    beh_dists  = np.array(A1["beh_dists_upper"])
    pair_types = np.array(A1["pair_types"])

    within_mask  = pair_types == "within"
    between_mask = pair_types == "between"

    # ── main panel: KDE contours only, no scatter ─────────────────────────────
    for mask, color, label in [
        (between_mask, BETWEEN_COL, "Between-mouse"),
        (within_mask,  WITHIN_COL,  "Within-mouse"),
    ]:
        x, y = topo_dists[mask], beh_dists[mask]
        kw = dict(x=x, y=y, levels=5, thresh=0.10, bw_adjust=0.8, )
        sns.kdeplot(**kw, fill=True,  color=color, alpha=0.15, zorder=1, ax=ax_main)
        sns.kdeplot(**kw, fill=False, color=color, alpha=0.70,
                    linewidths=1.0 * LW_SCALE, zorder=2,
                    ax=ax_main, label=label)
        ax_main.plot([],[], color=color, label=label)
    ax_main.legend(frameon=True, fontsize=FS_LEGEND, loc=(0.6,0.85))

    rho_ev, _ = spearmanr(topo_dists, beh_dists)
    rho_rand  = float(A5["rho_rand"])
    ax_main.text(
        0.03, 0.97,
        f"Evolved:  $\\rho = {rho_ev:+.3f}$\n"
        f"Random:  $\\rho = {rho_rand:+.3f}$\n"
        f"both $\\approx 0$: flatness is architectural",
        transform=ax_main.transAxes, ha="left", va="top",
        fontsize=FS_ANNOT, color="#333",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#ccc", alpha=0.85),
        zorder=6,
    )
    ax_main.set_xlabel("Topology distance (Jaccard)", fontsize=FS_LABEL)
    ax_main.set_ylabel("Behavioural distance (cosine)", fontsize=FS_LABEL)
    ax_main.set_xlim(0.6, 1.0)
    ax_main.set_ylim(-0.01, 0.08)
    ax_main.tick_params(labelsize=FS_TICK)
    # pub_despine(ax_main)

    # ── top marginal: KDE of topology distance ────────────────────────────────
    for mask, color in [(between_mask, BETWEEN_COL), (within_mask, WITHIN_COL)]:
        sns.kdeplot(topo_dists[mask], ax=ax_top, color=color,
                    fill=True, alpha=0.25, linewidth=1.0 * LW_SCALE, bw_adjust=0.8)
    for sp in ax_top.spines.values():
        sp.set_visible(False)
    ax_top.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax_top.set_ylabel("")

    # ── right marginal: KDE of behavioural distance (rotated) ─────────────────
    for mask, color in [(between_mask, BETWEEN_COL), (within_mask, WITHIN_COL)]:
        sns.kdeplot(y=beh_dists[mask], ax=ax_right, color=color,
                    fill=True, alpha=0.25, linewidth=1.0 * LW_SCALE, bw_adjust=0.8)
    for sp in ax_right.spines.values():
        sp.set_visible(False)
    ax_right.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax_right.set_xlabel("")


# ─────────────────────────────────────────────────────────────────────────────
# Panel D — NMI bar chart (§2.4)
# ─────────────────────────────────────────────────────────────────────────────

def _nmi_bar_panel(ax: plt.Axes, A3: dict) -> None:
    axis_keys = ["Topology", "Magnitude", "Sign (E/I)"]
    # Mouse-identity NMI — populated by scripts/update_A3_mouse_identity.py
    mi_vals   = [float(A3[k]["MI_mouse"])  for k in axis_keys]
    nmi_vals  = [float(A3[k]["NMI_mouse"]) for k in axis_keys]
    colors    = [BETWEEN_COL, "#DDCC77", "#44AA99"]

    x_pos = np.arange(len(axis_keys))
    ax.bar(x_pos, nmi_vals, color=colors, width=0.5, edgecolor="white", zorder=2)

    for x, nmi, mi in zip(x_pos, nmi_vals, mi_vals):
        ax.text(x, nmi+0.01,      f"MI={mi:.4f}",  ha="center", va="center",
                fontsize=FS_ANNOT, color="black", fontweight="bold")
        # ax.text(x, nmi + 0.005,  f"NMI={nmi:.4f}", ha="center", va="bottom",
        #         fontsize=FS_ANNOT, color="#333")

    # Reference at 0.5 so readers have a calibration point; ylim includes it.
    # ax.axhline(0.5, color="gray", lw=0.7 * LW_SCALE, ls=":", alpha=0.6, label="NMI = 0.5 (moderate)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(axis_keys, fontsize=FS_TICK)
    ax.set_ylabel("Normalised Mutual Information (NMI)", fontsize=FS_LABEL)
    ax.set_ylim(0, 0.25)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    nmi_max = max(nmi_vals)
    # ax.text(0.03, 0.97, f"Max NMI = {nmi_max:.2f}\nAMI ≈ 0 (chance level)",
    #         transform=ax.transAxes, ha="left", va="top",
    #         fontsize=FS_ANNOT, style="italic", color="#444")
    ax.legend(frameon=False, fontsize=FS_LEGEND, loc="upper right")
    pub_despine(ax)


# ─────────────────────────────────────────────────────────────────────────────
# Figure assembly
# ─────────────────────────────────────────────────────────────────────────────

def create_figure(store: dict) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Build the 5-panel figure and return (fig, axes_dict) for further editing.

    Requires matplotlib >= 3.4 for subfigure support.

    axes_dict keys
    --------------
    'A'        blank schematic placeholder
    'B'        topology Jaccard heatmap
    'C_main'   joint KDE main panel
    'C_top'    joint KDE top marginal
    'C_right'  joint KDE right marginal
    'D'        NMI bar chart
    """
    analysis_dir = os.path.join(_PROJECT, "analysis", "degeneracy_analyses")
    A1, A3, A5 = _load_all(analysis_dir)

    fig = plt.figure(figsize=FIGSIZE['figA'])

    # New layout: row 0 = A schematic spanning full width;
    # row 1 = [B heatmap | C joint-KDE subfigure | D NMI bars].
    outer = fig.add_gridspec(
        2, 3,
        height_ratios=[1.0, 1.15],
        width_ratios=[1.0, 1.5, 0.9],
        hspace=0.28, wspace=0.30,
        left=0.05, right=0.96, top=0.96, bottom=0.07,
    )

    # ── Panel A: method schematic (3-panel) spans entire top row ─────────────
    ax_a = draw_into_spec(fig, outer[0, :])
    # ── simple axes: B (heatmap) and D (NMI bars) on bottom row ──────────────
    ax_b = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 2])

    # ── Panel C: subfigure in bottom-row centre with 2×2 joint KDE layout ────
    sfig_c     = fig.add_subfigure(outer[1, 1])
    c_gs       = sfig_c.add_gridspec(
        2, 2,
        height_ratios=[1, 5],
        width_ratios=[5, 1],
        hspace=0.05, wspace=0.05,
    )
    ax_c_main   = sfig_c.add_subplot(c_gs[1, 0])
    ax_c_top    = sfig_c.add_subplot(c_gs[0, 0], sharex=ax_c_main)
    ax_c_right  = sfig_c.add_subplot(c_gs[1, 1], sharey=ax_c_main)
    ax_c_corner = sfig_c.add_subplot(c_gs[0, 1])
    ax_c_corner.axis("off")

    # ── populate ──────────────────────────────────────────────────────────────
    _topo_heatmap_panel(ax_b, A1)
    _joint_kde_panel(ax_c_main, ax_c_top, ax_c_right, A1, A5)
    _nmi_bar_panel(ax_d, A3)

    # ── panel labels ──────────────────────────────────────────────────────────
    label_panel(ax_a,    "A")
    label_panel(ax_b,    "B")
    label_panel(ax_c_top,"C")
    label_panel(ax_d,    "D")

    axes = {
        "A":       ax_a,
        "B":       ax_b,
        "C_main":  ax_c_main,
        "C_top":   ax_c_top,
        "C_right": ax_c_right,
        "D":       ax_d,
    }
    return fig, axes


def generate(store: dict, figures_dir: str) -> list[str]:
    apply_pub_style(font_scale=1.65)
    fig, _ = create_figure(store)
    out = os.path.join(figures_dir, "figA_topo_flat_v2.pdf")
    return save_figure(fig, out)
