"""
Supplementary Figure S5 (fig:supp_fixedpoints) — Fixed point and attractor analysis.
Output: fig_s5_fixedpoints.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

from ._style import apply_pub_style, MICE, MOUSE_COLORS, FS_ANNOT, FIGSIZE, LW_SCALE, MARKER_SCALE


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    dyn = store.dynamics_results()
    fp = dyn['fixed_points']

    fp_results = fp['fp_results']
    F_fp = fp['F_fp']
    p_fp = fp['p_fp']
    F_lam = fp['F_lam']
    p_lam = fp['p_lam']

    # Build per-mouse aggregations from fp_results
    fp_by_mouse = {m: [] for m in MICE}
    lam_by_mouse = {m: [] for m in MICE}
    for r in fp_results:
        m = r['mouse']
        fp_by_mouse[m].append(r['n_fixed_points'])
        if 'mean_lam_max' in r and r['mean_lam_max'] is not None:
            lam_by_mouse[m].append(r['mean_lam_max'])

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE['supp_s5'])

    # Panel A: number of fixed points per mouse
    ax = axes[0]
    for k, m in enumerate(MICE):
        vals = fp_by_mouse.get(m, [])
        if not vals:
            continue
        ax.scatter([k] * len(vals), vals, color=MOUSE_COLORS[m],
                   alpha=0.7, s=30 * MARKER_SCALE, zorder=3)
        ax.plot([k - 0.2, k + 0.2], [np.mean(vals)] * 2,
                color=MOUSE_COLORS[m], lw=2 * LW_SCALE)
    ax.set_xticks(range(len(MICE)))
    ax.set_xticklabels(MICE)
    ax.set_ylabel('Number of fixed points')
    ax.set_title(f'A  Fixed points per circuit\n'
                 f'ANOVA F={F_fp:.2f}, p={p_fp:.3f}', fontweight='bold')

    # Panel B: dominant eigenvalue per mouse
    ax = axes[1]
    for k, m in enumerate(MICE):
        vals = lam_by_mouse.get(m, [])
        if not vals:
            continue
        ax.scatter([k] * len(vals), vals, color=MOUSE_COLORS[m],
                   alpha=0.7, s=30 * MARKER_SCALE, zorder=3)
        ax.plot([k - 0.2, k + 0.2], [np.mean(vals)] * 2,
                color=MOUSE_COLORS[m], lw=2 * LW_SCALE)
    ax.axhline(1.0, color='red', lw=1 * LW_SCALE, ls='--', alpha=0.6,
               label='|\u03bb|=1 (stability boundary)')
    ax.set_xticks(range(len(MICE)))
    ax.set_xticklabels(MICE)
    ax.set_ylabel('Dominant eigenvalue |\u03bb_max|')
    ax.set_title(f'B  Spectral radius at fixed points\n'
                 f'ANOVA F={F_lam:.2f}, p={p_lam:.3f}', fontweight='bold')
    ax.legend()

    # Panel C: PCA of all fixed point locations
    ax = axes[2]
    all_fps_flat = []
    fp_labels = []
    for r in fp_results:
        for fpt in r['fixed_points']:
            all_fps_flat.append(fpt)
            fp_labels.append(r['mouse'])

    if len(all_fps_flat) > 10:
        pca_fp = PCA(n_components=2)
        fp_coords = pca_fp.fit_transform(np.array(all_fps_flat))
        for m in MICE:
            idxs = [i for i, lb in enumerate(fp_labels) if lb == m]
            if idxs:
                ax.scatter(fp_coords[idxs, 0], fp_coords[idxs, 1],
                           color=MOUSE_COLORS[m], alpha=0.5, s=20 * MARKER_SCALE, label=m)
        ax.set_xlabel(
            f'PC1 ({pca_fp.explained_variance_ratio_[0] * 100:.1f}%)')
        ax.set_ylabel(
            f'PC2 ({pca_fp.explained_variance_ratio_[1] * 100:.1f}%)')
        ax.set_title('C  Fixed point locations\n(all agents, PCA projection)',
                     fontweight='bold')
        ax.legend(fontsize=FS_ANNOT, ncol=3)
        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1.5,1.5)
        sns.despine(ax=ax, trim=True, offset=5)
    else:
        ax.text(0.5, 0.5, 'Insufficient\nfixed points',
                transform=ax.transAxes, ha='center')

    plt.tight_layout()
    out = os.path.join(figures_dir, 'fig_s5_fixedpoints.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return [out]
