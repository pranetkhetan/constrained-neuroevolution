"""
paper_v2 Supplementary §2.7 — Activity embedding control figures.

2-panel layout:
  A (left):  B2 motor positive control — strip + box comparing speed-turn
             distance vs mean interneuron-pair distance in PC1-PC2 loading space.
             MW p < 0.001 confirms embedding is informative at N=14.
  B (right): B3 manifold dimensionality — distribution of effective dimensionality
             (at 90% variance threshold) and participation ratio across 54 agents.
             Justifies choice of 6 retained PCs.

Output: figures/fig_supp_act_emb_ctrl.pdf
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import mannwhitneyu

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figure_modules._style import (
    apply_pub_style, save_figure, label_panel, pub_despine,
    FIGSIZE,
    FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    EVOLVED_COL, GEN_COL,
    LW_SCALE, MARKER_SCALE,
)


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
    """Generate fig_supp_act_emb_ctrl.pdf — B2 + B3 control panels."""
    apply_pub_style()

    act_emb_dir = os.path.join(store._analysis, "activity_embeddings")
    b_path = os.path.join(act_emb_dir, "B_results.pkl")
    if not os.path.exists(b_path):
        raise FileNotFoundError(f"B_results.pkl not found: {b_path}")

    B  = _load_pkl(b_path)
    B2 = B["B2"]
    B3 = B["B3"]

    motor_dists     = np.asarray(B2["motor_dists"])      # (54,) speed-vs-turn distance
    inter_dists     = np.asarray(B2["inter_dists_mean"]) # (54,) mean inter-interneuron dist

    eff_dim = np.asarray(B3["eff_dim_90"])               # (54,)
    pr_all  = np.asarray(B3["participation_ratio"])       # (54,)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE['supp_act_emb'])

    # ── Panel A: B2 motor separation ─────────────────────────────────────
    ax = axes[0]
    data_plot = {
        "Motor\n(speed–turn)": motor_dists,
        "Interneuron\n(pair mean)": inter_dists,
    }
    colors = [EVOLVED_COL, GEN_COL]
    for i, (label, vals) in enumerate(data_plot.items()):
        ax.boxplot(vals, positions=[i], widths=0.35,
                   patch_artist=True,
                   boxprops=dict(facecolor=colors[i], alpha=0.4),
                   medianprops=dict(color="k", lw=1.4 * LW_SCALE),
                   whiskerprops=dict(color="0.4"),
                   capprops=dict(color="0.4"),
                   flierprops=dict(marker="o", ms=3 * MARKER_SCALE, color="0.5", alpha=0.5))
        ax.scatter(np.full(len(vals), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(vals)),
                   vals, s=10 * MARKER_SCALE, color=colors[i], alpha=0.55, zorder=3, linewidths=0 * LW_SCALE)

    stat, p_mw = mannwhitneyu(motor_dists, inter_dists, alternative="greater")
    p_str = "p < 0.001" if p_mw < 0.001 else f"p = {p_mw:.3f}"

    # significance bracket
    ymax = max(motor_dists.max(), inter_dists.max()) * 1.05
    ax.plot([0, 0, 1, 1], [ymax, ymax * 1.03, ymax * 1.03, ymax], lw=1.0 * LW_SCALE, color="k")
    ax.text(0.5, ymax * 1.05, p_str, ha="center", va="bottom",
            fontsize=FS_ANNOT, color="k")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(list(data_plot.keys()), fontsize=FS_TICK)
    ax.set_ylabel("PC1–PC2 loading distance", fontsize=FS_LABEL)
    pub_despine(ax)
    label_panel(ax, "A")

    # ── Panel B: B3 manifold dimensionality ──────────────────────────────
    ax2 = axes[1]
    bins_e = np.arange(int(eff_dim.min()), int(eff_dim.max()) + 2, 1)
    ax2.hist(eff_dim, bins=bins_e, color=EVOLVED_COL, alpha=0.55, label="Eff. dim (90%)")
    ax2.hist(pr_all,  bins=np.linspace(pr_all.min() - 0.5, pr_all.max() + 0.5, 12),
             color=GEN_COL, alpha=0.45, label="Participation ratio")

    ax2.axvline(eff_dim.mean(), color=EVOLVED_COL, lw=1.4 * LW_SCALE, ls="--")
    ax2.axvline(pr_all.mean(),  color=GEN_COL,     lw=1.4 * LW_SCALE, ls="--")

    ax2.text(0.97, 0.92,
             f"Eff.dim = {eff_dim.mean():.2f} ± {eff_dim.std():.2f}\n"
             f"PR = {pr_all.mean():.2f} ± {pr_all.std():.2f}",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=FS_ANNOT, color="0.3")

    ax2.axvline(6, color="0.5", lw=1.0 * LW_SCALE, ls=":", label="6 components retained")
    ax2.set_xlabel("Dimensionality", fontsize=FS_LABEL)
    ax2.set_ylabel("Count (agents)", fontsize=FS_LABEL)
    ax2.legend(frameon=False, fontsize=FS_LEGEND)
    pub_despine(ax2)
    label_panel(ax2, "B")

    fig.tight_layout()
    out = os.path.join(figures_dir, "fig_supp_act_emb_ctrl.pdf")
    save_figure(fig, out)
    return [out]
