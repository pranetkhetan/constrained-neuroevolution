"""
Supplementary Figure S9 (fig:supp_nonzero) — Union of non-zero connection positions.
Output: fig_c4_nonzero_positions.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_pub_style, NEURON_LABELS, FS_TICK, FS_ANNOT, FIGSIZE, LW_SCALE, MARKER_SCALE


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    wd = store.weight_data()
    non_zero_mask = wd['non_zero_mask']
    n = non_zero_mask.shape[0]

    fig, ax = plt.subplots(figsize=FIGSIZE['supp_s9'])
    im = ax.imshow(non_zero_mask.astype(float), cmap='Blues', aspect='equal',
                   vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(NEURON_LABELS, fontsize=FS_TICK, rotation=45, ha='right')
    ax.set_yticklabels(NEURON_LABELS, fontsize=FS_TICK)
    ax.set_xlabel('Target neuron (j)')
    ax.set_ylabel('Source neuron (i)')
    ax.set_title(f'Non-zero connection positions\n'
                 f'({int(non_zero_mask.sum())} / {non_zero_mask.size} active across 54 agents)',
                 fontweight='bold')

    # Boundary lines between neuron types
    for boundary in [5.5, 11.5]:
        ax.axhline(boundary, color='gray', lw=0.5 * LW_SCALE, alpha=0.5)
        ax.axvline(boundary, color='gray', lw=0.5 * LW_SCALE, alpha=0.5)

    # Add type labels
    ax.text(-2.5, 2.5, 'Sensory', ha='center', va='center', fontsize=FS_ANNOT,
            rotation=90, color='gray')
    ax.text(-2.5, 8.5, 'Inter', ha='center', va='center', fontsize=FS_ANNOT,
            rotation=90, color='gray')
    ax.text(-2.5, 12.5, 'Motor', ha='center', va='center', fontsize=FS_ANNOT,
            rotation=90, color='gray')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Non-zero in \u22651 agent')
    plt.tight_layout()
    out = os.path.join(figures_dir, 'fig_c4_nonzero_positions.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return [out]
