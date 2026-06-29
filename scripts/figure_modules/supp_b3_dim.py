"""
paper_v2 §2.7 Supplementary — Manifold dimensionality is near-maximal and uniform.

Two-panel figure:
  Left:  Cumulative variance curves for all 54 agents (mouse-coloured), 90% threshold.
  Right: Participation ratio per mouse (strip plot with median + IQR).

Source: analysis/activity_embeddings/B_results.pkl  B3 block
Output: figures/fig_supp_b3_dim.pdf + .png
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figure_modules._style import (
    apply_pub_style, save_figure, label_panel, pub_despine,
    FIGSIZE, MOUSE_COLORS, MICE,
    FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    LW_SCALE, MARKER_SCALE,
)

_N_REPS = 6


def _load_pkl(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (ModuleNotFoundError, AttributeError):
        class _U(pickle.Unpickler):
            def find_class(self, mod, name):
                if mod.startswith("cupy"):
                    mod = mod.replace("cupy._core.core", "numpy").replace("cupy", "numpy")
                return super().find_class(mod, name)
        with open(path, "rb") as f:
            return _U(f).load()


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()

    act_dir = os.path.join(store._analysis, "activity_embeddings")
    B  = _load_pkl(os.path.join(act_dir, "B_results.pkl"))
    B3 = B["B3"]

    cum_var            = np.asarray(B3["cumulative_var"])        # (54, 6)
    part_ratio         = np.asarray(B3["participation_ratio"])   # (54,)
    agent_mouse_labels = np.asarray(B["agent_mouse_labels"])     # (54,)

    n_pc = cum_var.shape[1]
    pcs  = np.arange(1, n_pc + 1)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIGSIZE['supp_b3_dim'])

    # ── Left: cumulative variance curves ─────────────────────────────────────
    for agent_idx in range(len(cum_var)):
        mouse = agent_mouse_labels[agent_idx]
        ax_l.plot(pcs, cum_var[agent_idx], color=MOUSE_COLORS[mouse],
                  lw=0.9 * LW_SCALE, alpha=0.55, zorder=2)

    ax_l.axhline(0.90, color="0.45", lw=1.0 * LW_SCALE, ls="--", zorder=3,
                 label="90% variance threshold")
    ax_l.set_xlabel("Number of PCs", fontsize=FS_LABEL)
    ax_l.set_ylabel("Cumulative variance explained", fontsize=FS_LABEL)
    ax_l.set_xticks(pcs)
    ax_l.set_ylim(0.15, 1.02)
    ax_l.tick_params(labelsize=FS_TICK)

    # Mouse colour legend
    mouse_handles = [
        mlines.Line2D([], [], color=MOUSE_COLORS[m], lw=1.4 * LW_SCALE, label=m)
        for m in MICE
    ]
    mouse_handles.append(
        mlines.Line2D([], [], color="0.45", lw=1.0 * LW_SCALE, ls="--", label="90% threshold")
    )
    ax_l.legend(handles=mouse_handles, frameon=False, fontsize=FS_LEGEND,
                ncol=2, loc="lower right")
    pub_despine(ax_l)
    label_panel(ax_l, "A")

    # ── Right: participation ratio per mouse (strip + median/IQR) ────────────
    rng = np.random.default_rng(42)
    for m_idx, mouse in enumerate(MICE):
        mask = agent_mouse_labels == mouse
        vals = part_ratio[mask]
        jitter = rng.uniform(-0.18, 0.18, size=mask.sum())
        ax_r.scatter(np.full(mask.sum(), m_idx) + jitter, vals,
                     color=MOUSE_COLORS[mouse], s=28 * MARKER_SCALE, alpha=0.85,
                     linewidths=0 * LW_SCALE, zorder=3)
        med = np.median(vals)
        q1, q3 = np.percentile(vals, [25, 75])
        ax_r.plot([m_idx - 0.25, m_idx + 0.25], [med, med],
                  color="0.2", lw=1.8 * LW_SCALE, zorder=4)
        ax_r.vlines(m_idx, q1, q3, color="0.4", lw=1.2 * LW_SCALE, zorder=4)

    grand_mean = part_ratio.mean()
    grand_std  = part_ratio.std()
    ax_r.axhline(grand_mean, color="0.55", lw=0.9 * LW_SCALE, ls=":", zorder=2)
    ax_r.text(0.97, 0.04,
              f"Grand mean PR $= {grand_mean:.2f} \\pm {grand_std:.2f}$",
              transform=ax_r.transAxes, ha="right", va="bottom",
              fontsize=FS_ANNOT, color="0.35")

    ax_r.set_xticks(range(len(MICE)))
    ax_r.set_xticklabels(MICE, fontsize=FS_TICK, rotation=30, ha="right")
    ax_r.set_ylabel("Participation ratio", fontsize=FS_LABEL)
    ax_r.set_ylim(bottom=0)
    ax_r.tick_params(axis="y", labelsize=FS_TICK)
    pub_despine(ax_r)
    label_panel(ax_r, "B")

    fig.tight_layout()

    out = os.path.join(figures_dir, "fig_supp_b3_dim.pdf")
    save_figure(fig, out)
    return [out]


if __name__ == "__main__":
    class _Store:
        _analysis = os.path.join(_PROJECT, "analysis")
    generate(_Store(), os.path.join(_PROJECT, "figures"))
