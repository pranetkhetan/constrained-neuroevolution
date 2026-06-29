"""
Supplementary Figure S6 -- Per-metric 9x9 cross-evaluation matrices.
4-panel 2x2 layout: Markov | Occupancy | Tortuosity | Turn Bias.
(Total Fitness panel omitted; shown in main text Figure 2D.)

Output: fig_per_metric_crosseval.pdf / .png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from ._style import (
    apply_pub_style,
    MICE, MOUSE_COLORS,
    pub_despine, save_figure, label_panel,
    FS_LABEL, FS_TICK, FS_ANNOT, FS_SMALL,
    FIGSIZE,
    LW_SCALE, MARKER_SCALE,
)

METRIC_TITLES = {
    'markov':     'Markov',
    'occupancy':  'Occupancy',
    'tortuosity': 'Tortuosity',
    'turn_bias':  'Turn Bias',
}
PANEL_LETTERS = ['A', 'B', 'C', 'D']


def _heatmap_panel(ax, mat: np.ndarray, mice: list, metric_key: str) -> None:
    n = len(mice)
    im = ax.imshow(mat, cmap='YlOrRd', aspect='equal', interpolation='nearest')

    for k in range(n):
        ax.add_patch(mpatches.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                       fill=False, edgecolor='#3C5488', lw=1.8 * LW_SCALE, zorder=5))

    ax.axhline(2.5, color='white', lw=1.0 * LW_SCALE, zorder=4)
    ax.axvline(2.5, color='white', lw=1.0 * LW_SCALE, zorder=4)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(mice, fontsize=FS_SMALL, rotation=45, ha='right')
    ax.set_yticklabels(mice, fontsize=FS_SMALL)
    ax.set_xlabel('Test mouse', fontsize=FS_LABEL)
    ax.set_ylabel('Train mouse', fontsize=FS_LABEL)
    ax.set_title(METRIC_TITLES.get(metric_key, metric_key),
                 fontsize=FS_LABEL, pad=4, loc='center')
    plt.colorbar(im, ax=ax, shrink=0.85, fraction=0.046, pad=0.04)


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    pm = store.cross_mouse_per_metric()
    component_matrices = pm['component_matrices']
    metrics = list(pm['metrics'])
    mice = pm.get('mice', MICE)

    fig = plt.figure(figsize=FIGSIZE.get('s10a', (18, 12)))
    gs = gridspec.GridSpec(
        2, 2,
        hspace=0.45, wspace=0.35,
        left=0.07, right=0.97, top=0.95, bottom=0.08,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    for ax, metric_key, letter in zip(axes, metrics[:4], PANEL_LETTERS):
        _heatmap_panel(ax, component_matrices[metric_key], mice, metric_key)
        label_panel(ax, letter)

    out = os.path.join(figures_dir, 'fig_per_metric_crosseval.pdf')
    return save_figure(fig, out)
