"""
Figure EVR (fig:evr) — Forest plot: evolved vs random circuit features (Cohen's d).

Standalone figure extracted from fig5 and placed adjacent to Table 1 in §2.3.
Output: fig_evr_forest.pdf
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.stats import cohens_d_bootstrap_ci
from ._style import FS_LABEL, FS_TICK, FS_LEGEND, FS_TITLE


FEAT_LABELS = {
    'sm_count':       'S→M connections',
    'density':        'Network density',
    'n_connections':  'Total connections',
    'inter_in_mean':  'Interneuron mean fan-in',
    'im_count':       'I→M connections',
    'inter_out_mean': 'Interneuron mean fan-out',
    'frac_strong':    'Fraction at max weight',
    'w_mean_mag':     'Mean weight magnitude',
    'mm_count':       'M→M connections',
    'ei_ratio':       'E/I ratio',
    'ii_count':       'I→I connections',
    'n_inh':          'Inhibitory neuron count',
    'n_exc':          'Excitatory neuron count',
    'im_exc_frac':    'I→M excitatory fraction',
    'ii_exc_frac':    'I→I excitatory fraction',
    'mi_count':       'M→I connections',
    'si_count':       'S→I connections',
    'si_exc_frac':    'S→I excitatory fraction',
}


def _forest_panel(ax, evr_results: dict, d_cis: dict) -> None:
    feats = sorted(evr_results.keys(), key=lambda f: abs(evr_results[f]['cohens_d']))
    ds    = [evr_results[f]['cohens_d'] for f in feats]
    ps    = [evr_results[f]['p']        for f in feats]

    y_pos = np.arange(len(feats))
    for i, (feat, d, p) in enumerate(zip(feats, ds, ps)):
        color = '#3C5488' if p < 0.05 else '#999999'
        ax.plot(d, i, 'o', color=color, markersize=5.5, zorder=3)
        if d_cis and feat in d_cis:
            _, d_lo, d_hi = d_cis[feat]
            ax.plot([d_lo, d_hi], [i, i], '-', color=color, lw=1.3, zorder=2)

    ax.axvline(0, ls='--', color='gray', lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_xlim(-1.5, 1.5)
    ax.set_yticklabels([FEAT_LABELS.get(f, f) for f in feats], fontsize=FS_LABEL)
    ax.set_xlabel("Cohen's $d$ (evolved $-$ random)", fontsize=FS_LABEL)
    ax.tick_params(axis='x', labelsize=FS_TICK)

    legend_els = [
        Line2D([0], [0], marker='o', color='#3C5488', lw=0, markersize=5.5, label='$p < 0.05$'),
        Line2D([0], [0], marker='o', color='#999999', lw=0, markersize=5.5, label='$p \geq 0.05$'),
    ]
    ax.legend(handles=legend_els, loc='lower right', frameon=False, fontsize=FS_LEGEND)
    sns.despine(ax=ax)


def generate(store, figures_dir: str) -> list[str]:
    circuit_data    = store.circuit_features()
    random_baseline = store.random_baseline()
    stats_results   = store.stats_results()

    evr_results   = stats_results['evolved_vs_random']
    feature_names = list(evr_results.keys())

    d_cis = {}
    for feat in feature_names:
        evolved_vals = np.array([r[feat] for r in circuit_data])
        random_vals  = np.array([r[feat] for r in random_baseline['all']])
        d_cis[feat]  = cohens_d_bootstrap_ci(evolved_vals, random_vals)

    fig, ax = plt.subplots(figsize=(7, 8))
    fig.subplots_adjust(left=0.35, right=0.97, top=0.94, bottom=0.08)
    _forest_panel(ax, evr_results, d_cis)

    out = os.path.join(figures_dir, 'fig_evr_forest.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out}')
    return [out]
