"""
Supplementary Figure — Per-source ablation sensitivity heatmap (54 x 14).

Agents ordered by mouse (6 per mouse, separated by white lines). No source
neuron shows significant mouse-level organisation: 0/14 source neurons pass
Bonferroni correction; within-mouse sensitivity profiles are no more correlated
than between-mouse profiles. Moved from main fig3 Panel B (null result).

Output: fig_supp_ablation_heatmap.pdf
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from ._style import (
    apply_pub_style,
    FIGSIZE,
    FS_ANNOT, FS_LABEL, FS_TICK,
    MICE, MOUSE_COLORS, NEURON_LABELS,
    pub_despine, save_figure,
    LW_SCALE, MARKER_SCALE,
)


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    abl    = store.source_sensitivity()
    matrix = np.array(abl['sensitivity_matrix'])  # (54, 14)
    labels = list(abl['labels'])

    order    = np.argsort([MICE.index(m) if m in MICE else 99 for m in labels])
    mat_ord  = matrix[order]
    mice_ord = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=FIGSIZE['supp_ablation'])
    fig.subplots_adjust(left=0.1, right=0.92, top=0.93, bottom=0.12)

    im = ax.imshow(mat_ord, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=float(np.percentile(mat_ord, 95)),
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, shrink=0.8, label='MSE increase (per-source ablation)')

    for sep in range(6, 54, 6):
        ax.axhline(sep - 0.5, color='white', lw=1.5 * LW_SCALE)

    counts = Counter(mice_ord)
    pos = 0
    yticks, ylabels = [], []
    for m in MICE:
        if m in counts:
            yticks.append(pos + counts[m] / 2 - 0.5)
            ylabels.append(m)
            pos += counts[m]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=FS_TICK)
    for tick, m in zip(ax.get_yticklabels(), ylabels):
        tick.set_color(MOUSE_COLORS.get(m, '#333'))
        tick.set_fontweight('bold')

    ax.set_xticks(range(14))
    ax.set_xticklabels(NEURON_LABELS, fontsize=FS_TICK, rotation=45, ha='right')
    ax.set_xlabel('Source neuron', fontsize=FS_LABEL)
    ax.set_ylabel('Agent (ordered by mouse, 6 replicates each)', fontsize=FS_LABEL)

    ax.text(0.02, 0.97,
            '0/14 source neurons significant\n(Bonferroni corrected)',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=FS_ANNOT, color='#333',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    out = os.path.join(figures_dir, 'fig_supp_ablation_heatmap.pdf')
    return save_figure(fig, out)
