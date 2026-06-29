"""
Supplementary Figure S8 (fig:supp_cosine) — Evolved vs random weight-vector cosine similarity.
Output: fig_c1_cosine_evolved_vs_random.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import mannwhitneyu

from ._style import FIGSIZE


def generate(store, figures_dir: str) -> list[str]:
    wd = store.weight_data()
    evolved_vecs = wd['weight_vectors']
    random_vecs = wd['random_weight_vectors']

    # Pairwise cosine within each group
    evolved_sim = cosine_similarity(evolved_vecs)
    random_sim = cosine_similarity(random_vecs)

    n_e = len(evolved_vecs)
    n_r = len(random_vecs)
    triu_e = evolved_sim[np.triu_indices(n_e, k=1)]
    triu_r = random_sim[np.triu_indices(n_r, k=1)]

    _, p_val = mannwhitneyu(triu_e, triu_r, alternative='greater')

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(triu_e, bins=40, alpha=0.7, color='#3C5488', label='Evolved (54 agents)',
            density=True, edgecolor='white', linewidth=0.3)
    ax.hist(triu_r, bins=40, alpha=0.7, color='#B09C85', label='Random (54 agents)',
            density=True, edgecolor='white', linewidth=0.3)

    ax.axvline(np.mean(triu_e), color='#3C5488', ls='--', lw=1.2,
               label=f'Evolved mean={np.mean(triu_e):.3f}')
    ax.axvline(np.mean(triu_r), color='#B09C85', ls='--', lw=1.2,
               label=f'Random mean={np.mean(triu_r):.3f}')

    ax.set_xlabel('Pairwise cosine similarity')
    ax.set_ylabel('Density')
    ax.set_title(f'Evolved vs Random Weight-Vector Similarity\n'
                 f'(Mann-Whitney p < {p_val:.4f})')
    ax.legend()

    plt.tight_layout()
    out = os.path.join(figures_dir, 'fig_c1_cosine_evolved_vs_random.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return [out]
