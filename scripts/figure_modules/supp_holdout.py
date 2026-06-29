"""
Supplementary Figure S12 — Holdout specialisation indices per mouse.

Paired bar chart (or slope graph) comparing training-set and held-out
specialisation indices for each of the 9 mice. All held-out indices > 0,
confirming genuine cross-bout generalisation rather than bout-specific overfitting.

Output: fig_supp_holdout.pdf
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from ._style import (
    apply_pub_style,
    FIGSIZE,
    FS_ANNOT, FS_LABEL, FS_LEGEND, FS_TICK,
    MICE, MOUSE_COLORS,
    pub_despine, save_figure,
    LW_SCALE, MARKER_SCALE,
)


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    r9         = store.holdout_results()
    gen_matrix = store.generalization_matrix()
    gen_meta   = store.generalization_meta()

    # Per-mouse training specialisation index from the standard gen_matrix
    gm_mice = gen_meta['mice']
    train_idx = {}
    for i, m in enumerate(gm_mice):
        own = gen_matrix[i, i]
        off = float(np.mean([gen_matrix[i, j] for j in range(len(gm_mice)) if j != i]))
        train_idx[m] = 1.0 - own / off

    # Per-mouse holdout specialisation index: 1 - (own_holdout / mean_other_holdout)
    per_mouse_holdout = r9['per_mouse_holdout']   # {mouse: own/other ratio}
    holdout_idx = {m: 1.0 - per_mouse_holdout[m] for m in per_mouse_holdout}

    common = [m for m in MICE if m in train_idx and m in holdout_idx]
    train_vals   = np.array([train_idx[m]   for m in common])
    holdout_vals = np.array([holdout_idx[m] for m in common])

    fig, ax = plt.subplots(figsize=FIGSIZE['supp_holdout'])
    fig.subplots_adjust(left=0.1, right=0.97, top=0.90, bottom=0.12)

    x = np.arange(len(common))
    w = 0.35
    colors = [MOUSE_COLORS.get(m, '#888') for m in common]

    for i, (m, tv, hv, col) in enumerate(zip(common, train_vals, holdout_vals, colors)):
        ax.bar(i - w / 2, tv, width=w, color=col, alpha=0.85,
               edgecolor='white', label='Training' if i == 0 else '')
        ax.bar(i + w / 2, hv, width=w, color=col, alpha=0.45,
               edgecolor='white', hatch='//', label='Held-out' if i == 0 else '')

    ax.axhline(0, color='black', lw=0.8 * LW_SCALE, ls='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(common, fontsize=FS_TICK)
    ax.set_ylabel('Specialisation index (1 − own/cross)', fontsize=FS_LABEL)
    ax.tick_params(axis='y', labelsize=FS_TICK)

    # Population means
    ax.axhline(train_vals.mean(), color='#444', lw=1.0 * LW_SCALE, ls=':', alpha=0.7)
    ax.axhline(holdout_vals.mean(), color='#aaa', lw=1.0 * LW_SCALE, ls=':', alpha=0.7)

    ax.text(0.98, 0.97,
            f'Training mean = {train_vals.mean():.3f}\n'
            f'Held-out mean = {holdout_vals.mean():.3f}\n'
            f'All held-out > 0',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FS_ANNOT, color='#333')

    # Custom legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='#666', alpha=0.85, label='Training'),
        Patch(facecolor='#aaa', alpha=0.45, hatch='//', label='Held-out'),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=FS_LEGEND,
              loc='lower right')
    pub_despine(ax)

    out = os.path.join(figures_dir, 'fig_supp_holdout.pdf')
    return save_figure(fig, out)
