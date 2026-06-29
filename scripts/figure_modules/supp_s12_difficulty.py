"""
Supplementary Figure S12 (fig:supp_difficulty) — Per-mouse difficulty correlation.
Output: supp_A3_difficulty_correlation.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from ._style import apply_pub_style, MICE, FS_ANNOT, FIGSIZE, LW_SCALE, MARKER_SCALE


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    d = store.a3_difficulty()

    # spec_own and gen_fit are dicts {mouse: value}
    spec_own = np.array([d['spec_own'][m] for m in MICE])
    gen_fit = np.array([d['gen_fit'][m] for m in MICE])
    r_val = float(d['pearson_r'])
    p_val = float(d['pearson_p'])

    # Compute regression line
    m_slope, b_int = np.polyfit(spec_own, gen_fit, 1)

    colors_map = {
        'B5': '#2196F3', 'B6': '#1565C0', 'B7': '#0D47A1',
        'D3': '#E53935', 'D4': '#C62828', 'D5': '#B71C1C',
        'D7': '#FF8F00', 'D8': '#E65100', 'D9': '#BF360C',
    }

    fig, ax = plt.subplots(figsize=FIGSIZE['supp_s12'])

    for i, mouse in enumerate(MICE):
        c = colors_map.get(mouse, '#555')
        ax.scatter(spec_own[i], gen_fit[i], color=c, s=70 * MARKER_SCALE, zorder=3)
        ax.annotate(mouse, (spec_own[i], gen_fit[i]),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=FS_ANNOT, color=c)

    x_range = np.linspace(spec_own.min() * 0.97, spec_own.max() * 1.03, 100)
    ax.plot(x_range, m_slope * x_range + b_int, 'k--', lw=1.2 * LW_SCALE, alpha=0.7)

    sig_str = f'p = {p_val:.3f}' if p_val >= 0.001 else 'p < 0.001'
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0.5, 1)
    ax.set_xlabel('Specialist own-mouse fitness (9\u00d79 diagonal)')
    ax.set_ylabel('Generalist per-mouse fitness')
    ax.set_title(f'Mouse-intrinsic difficulty\n(r = {r_val:.3f}, {sig_str})')
    sns.despine(ax=ax, trim=True, offset=5)

    patch_b = mpatches.Patch(color='#1565C0', label='B-strain (B5\u2013B7)')
    patch_d = mpatches.Patch(color='#C62828', label='D-strain (D3\u2013D9)')
    ax.legend(handles=[patch_b, patch_d])

    plt.tight_layout()
    out = os.path.join(figures_dir, 'supp_A3_difficulty_correlation.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return [out]
