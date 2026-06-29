"""
Supplementary Figure S11 (fig:supp_random_null) — Random-circuit null distribution.
Output: supp_A5_random_null.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ._style import apply_pub_style, FIGSIZE, LW_SCALE, MARKER_SCALE


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    d = store.a5_random_null()

    ratios = np.asarray(d['ratios'])
    n_random = len(ratios)

    fig, ax = plt.subplots(figsize=FIGSIZE['supp_s11'])
    ax.hist(ratios, bins=20, color='#999999', alpha=0.8, edgecolor='white',
            lw=0.5 * LW_SCALE, label=f'Random circuits (n={n_random})')
    ax.axvline(0.656, color='#c0392b', lw=2 * LW_SCALE,
               label='Evolved specialists (0.656)', zorder=5)
    ax.axvline(np.mean(ratios), color='#555', lw=1.5 * LW_SCALE, ls='--',
               label=f'Random mean ({np.mean(ratios):.3f})', alpha=0.8)
    ax.set_xlabel('Behavioral specialization ratio (own / other fitness)')
    ax.set_ylabel('Count')
    ax.set_title('Specialization null distribution: random vs evolved circuits')
    ax.legend()

    z = (0.656 - np.mean(ratios)) / np.std(ratios)
    p_one = float(stats.norm.cdf(z))
    ax.text(0.05, 0.92, f'Evolved z = {z:.1f}\n(p < {p_one:.0e})',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    out = os.path.join(figures_dir, 'supp_A5_random_null.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return [out]
