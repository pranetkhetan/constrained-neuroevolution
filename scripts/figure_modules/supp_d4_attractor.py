"""
Supplementary Figure S16 (fig:supp_d4) — Attractor landscape comparison.

Three-panel figure characterising the pairwise attractor-state distances across
the 54 evolved circuits:
  A: 54×54 heatmap of pairwise attractor distances, rows/cols sorted by mouse
  B: KDE of within-mouse vs between-mouse attractor distances (null result:
     p = 0.544, Mann-Whitney)
  C: 2D PCA scatter of per-agent final hidden states, coloured by mouse

Output: fig_supp_d4_attractor.pdf
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from ._style import (
    apply_pub_style,
    FIGSIZE,
    FS_ANNOT,
    FS_LABEL,
    FS_PANEL,
    FS_TICK,
    MICE,
    MOUSE_COLORS,
    label_panel,
    pub_despine,
    save_figure,
    LW_SCALE, MARKER_SCALE,
)

N_REPS = 6


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    D = store.act_emb_d()
    d4 = D["D4"]

    D_attr      = d4["D_attr"]        # (54, 54)
    within_attr = d4["within_attr"]   # (135,)
    between_attr= d4["between_attr"]  # (1296,)
    p_mw        = float(d4["p_mw"])
    all_finals  = d4["all_finals"]    # list of 54 arrays (T, hidden_dim)

    # Agent ordering: B5r1..r6, B6r1..r6, ... D9r1..r6
    agent_labels = [f"{m}r{r}" for m in MICE for r in range(1, N_REPS + 1)]
    mouse_per_agent = [m for m in MICE for _ in range(N_REPS)]

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE['supp_d4_attr'])
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.97, top=0.88, bottom=0.12)

    # ── Panel A: 54×54 heatmap ──────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(D_attr, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Attractor distance", fontsize=FS_TICK)
    cb.ax.tick_params(labelsize=FS_ANNOT)

    # Mouse boundary lines
    for k in range(1, len(MICE)):
        boundary = k * N_REPS - 0.5
        ax.axhline(boundary, color="white", lw=1.2 * LW_SCALE, alpha=0.8)
        ax.axvline(boundary, color="white", lw=1.2 * LW_SCALE, alpha=0.8)

    # Mouse tick labels at group centres
    centres = np.arange(N_REPS / 2 - 0.5, 54, N_REPS)
    ax.set_xticks(centres)
    ax.set_yticks(centres)
    ax.set_xticklabels(MICE, fontsize=FS_TICK, rotation=45, ha="right")
    ax.set_yticklabels(MICE, fontsize=FS_TICK)
    ax.set_xlabel("Agent (mouse × rep)", fontsize=FS_LABEL)
    ax.set_ylabel("Agent (mouse × rep)", fontsize=FS_LABEL)
    label_panel(ax, "A")

    # ── Panel B: within vs between KDE ──────────────────────────────────────
    ax = axes[1]
    WITHIN_COL  = "#44AA99"
    BETWEEN_COL = "#DDDDDD"

    sns.kdeplot(
        within_attr, ax=ax, color=WITHIN_COL, lw=2.0 * LW_SCALE,
        fill=True, alpha=0.35, label=f"Within-mouse (n={len(within_attr)})",
    )
    sns.kdeplot(
        between_attr, ax=ax, color=BETWEEN_COL, lw=1.5 * LW_SCALE,
        fill=True, alpha=0.40, label=f"Between-mouse (n={len(between_attr)})",
    )

    p_str = f"p = {p_mw:.3f}" if p_mw >= 0.001 else "p < 0.001"
    ax.text(
        0.97, 0.96, f"Mann-Whitney\n{p_str} (n.s.)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=FS_ANNOT,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85),
    )
    ax.set_xlabel("Attractor distance", fontsize=FS_LABEL)
    ax.set_ylabel("Density", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_ANNOT, frameon=False)
    pub_despine(ax)
    label_panel(ax, "B")

    # ── Panel C: 2D PCA scatter of final hidden states ──────────────────────
    ax = axes[2]

    # Take the final hidden state of each trajectory
    finals = np.vstack([af[-1] for af in all_finals])   # (54, hidden_dim)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(finals)
    var = pca.explained_variance_ratio_ * 100

    for mouse in MICE:
        mask = np.array([m == mouse for m in mouse_per_agent])
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            color=MOUSE_COLORS[mouse],
            s=48 * MARKER_SCALE, alpha=0.85,
            edgecolors="white", linewidths=0.5 * LW_SCALE,
            zorder=3, label=mouse,
        )

    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var.)", fontsize=FS_LABEL)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var.)", fontsize=FS_LABEL)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=MOUSE_COLORS[m],
               markeredgecolor='white', markersize=7 * MARKER_SCALE, label=m)
        for m in MICE
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=FS_ANNOT,
        frameon=False,
        ncol=2,
        loc="best",
    )
    pub_despine(ax)
    label_panel(ax, "C")

    out = os.path.join(figures_dir, "fig_supp_d4_attractor.pdf")
    return save_figure(fig, out)
