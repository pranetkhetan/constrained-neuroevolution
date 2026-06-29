"""
Supplementary Figure S2 (fig:supp_strain) — Strain clustering analysis.
Output: strain_clustering.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from ._style import apply_pub_style, FS_TICK, FIGSIZE, LW_SCALE, MARKER_SCALE


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    gen_matrix = store.generalization_matrix()
    gen_meta = store.generalization_meta()
    mice = gen_meta['mice']

    # Symmetrize and convert to condensed distance
    sym = (gen_matrix + gen_matrix.T) / 2
    np.fill_diagonal(sym, 0)
    condensed = squareform(sym)
    Z = linkage(condensed, method='average')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE['supp_s2'])

    # Dendrogram
    dn = dendrogram(Z, labels=mice, ax=ax1, leaf_font_size=FS_TICK)
    ax1.set_title('Hierarchical Clustering of Generalization Matrix')
    ax1.set_ylabel('Average cross-evaluation distance')
    for lbl in ax1.get_xticklabels():
        m = lbl.get_text()
        lbl.set_color('#E64B35' if m.startswith('B') else '#3C5488')
        lbl.set_fontweight('bold')

    # Reordered heatmap
    order = dn['leaves']
    reordered = gen_matrix[np.ix_(order, order)]
    reordered_mice = [mice[i] for i in order]

    im = ax2.imshow(reordered, cmap='RdYlGn_r', aspect='auto')
    ax2.set_xticks(range(len(mice)))
    ax2.set_yticks(range(len(mice)))
    ax2.set_xticklabels(reordered_mice, rotation=45)
    ax2.set_yticklabels(reordered_mice)
    for lbl in ax2.get_xticklabels():
        lbl.set_color('#E64B35' if lbl.get_text().startswith('B') else '#3C5488')
    for lbl in ax2.get_yticklabels():
        lbl.set_color('#E64B35' if lbl.get_text().startswith('B') else '#3C5488')
    plt.colorbar(im, ax=ax2, shrink=0.8, label='Fitness')
    ax2.set_title('Generalization Matrix (clustered)')

    plt.tight_layout()
    out = os.path.join(figures_dir, 'strain_clustering.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return [out]
