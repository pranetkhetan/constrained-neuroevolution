"""
Supplementary Figure S6 (fig:supp_permutation) — Weight permutation ablation.
Output: fig_s6_permutation.pdf

Panel C data (orig_feature_means, perm_feature_means, orig_feature_stds,
feature_names) is computed on first run and patched into dynamics_results.pkl
so that subsequent runs load it directly without recomputation.
"""

import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_pub_style, MICE, MOUSE_COLORS, FS_ANNOT, FIGSIZE, LW_SCALE, MARKER_SCALE


# ── Short display labels for the 18 circuit features (Panel C x-axis) ─────────
_FEAT_LABELS = {
    'n_connections':  'n conn',
    'density':        'density',
    'ei_ratio':       'E/I',
    'n_exc':          'n exc',
    'n_inh':          'n inh',
    'si_count':       'S→I',
    'sm_count':       'S→M',
    'ii_count':       'I→I',
    'im_count':       'I→M',
    'mi_count':       'M→I',
    'mm_count':       'M→M',
    'si_exc_frac':    'SI exc%',
    'ii_exc_frac':    'II exc%',
    'im_exc_frac':    'IM exc%',
    'inter_in_mean':  'I in°',
    'inter_out_mean': 'I out°',
    'w_mean_mag':     '|w|',
    'frac_strong':    'str%',
}


# ── Source-preserving permutation (mirrors phase3_colab_analyses.py) ──────────
_IDX_S = list(range(6))
_IDX_I = list(range(6, 12))
_IDX_M = list(range(12, 14))

_PATHWAY_BLOCKS = [
    (_IDX_S, _IDX_I),
    (_IDX_S, _IDX_M),
    (_IDX_I, _IDX_I),
    (_IDX_I, _IDX_M),
    (_IDX_M, _IDX_I),
    (_IDX_M, _IDX_M),
]


def _source_preserving_permutation(W_np: np.ndarray) -> np.ndarray:
    W = W_np.copy()
    for sources, targets in _PATHWAY_BLOCKS:
        for src in sources:
            row = W[src, :]
            connected = [t for t in targets if row[t] != 0]
            if len(connected) < 2:
                continue
            mags = [row[t] for t in connected]
            np.random.shuffle(mags)
            for t in targets:
                W[src, t] = 0
            available = list(targets)
            np.random.shuffle(available)
            for i, t in enumerate(available[:len(mags)]):
                W[src, t] = mags[i]
    return W


def _load_best_agent(agents_dir: str, mouse: str, rep: int, gen: int = 150):
    # agents_dir is '<project>/data/agents'; map to project root and prefer
    # data/best_agents.pkl (repo default), falling back to data/agents/.
    from scripts.figure_modules._loaders import load_best_agent as _lba
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(agents_dir)))
    try:
        return _lba(project_dir, mouse, rep, gen=gen)
    except FileNotFoundError:
        return None  # preserve original "missing -> None" contract


def _ensure_panelC_data(perm: dict, dyn_pkl_path: str, project_dir: str) -> None:
    """Compute and persist Panel C feature data if not already in the pkl."""
    if 'orig_feature_means' in perm:
        return  # already computed; straight to plotting

    # Import extract_features lazily to avoid circular imports
    scripts_dir = os.path.join(project_dir, 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    from analyze_circuits import extract_features, FEATURE_NAMES  # noqa: PLC0415

    np.random.seed(42)
    agents_dir = os.path.join(project_dir, 'data', 'agents')
    orig_rows, perm_rows = [], []

    for mouse in MICE:
        for rep in range(1, 7):
            agent = _load_best_agent(agents_dir, mouse, rep)
            if agent is None:
                continue
            W_orig = np.array(agent.weights, dtype=np.float64)
            orig_rows.append(extract_features(agent))
            W_perm = _source_preserving_permutation(W_orig)
            proxy = type('_A', (), {
                'weights': W_perm,
                'node_types': agent.node_types,
                'idx_sensory': agent.idx_sensory,
                'idx_inter': agent.idx_inter,
                'idx_motor': agent.idx_motor,
            })()
            perm_rows.append(extract_features(proxy))

    feat_names = FEATURE_NAMES
    orig_mat = np.array([[r[f] for f in feat_names] for r in orig_rows])
    perm_mat = np.array([[r[f] for f in feat_names] for r in perm_rows])

    perm['orig_feature_means'] = orig_mat.mean(axis=0).tolist()
    perm['orig_feature_stds']  = orig_mat.std(axis=0).tolist()
    perm['perm_feature_means'] = perm_mat.mean(axis=0).tolist()
    perm['feature_names']      = feat_names

    # Persist so repeat runs skip this block
    with open(dyn_pkl_path, 'rb') as f:
        dyn = pickle.load(f)
    dyn['permutation'].update({
        'orig_feature_means': perm['orig_feature_means'],
        'orig_feature_stds':  perm['orig_feature_stds'],
        'perm_feature_means': perm['perm_feature_means'],
        'feature_names':      perm['feature_names'],
    })
    with open(dyn_pkl_path, 'wb') as f:
        pickle.dump(dyn, f)


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    dyn = store.dynamics_results()
    perm = dyn['permutation']

    dyn_pkl_path = os.path.join(store._analysis, 'dynamics_results.pkl')
    _ensure_panelC_data(perm, dyn_pkl_path, store._project)

    perm_results   = perm['perm_results']
    same_mouse_mse = perm['same_mouse_mse']
    p_perm         = perm['mannwhitney_p']
    orig_means     = perm['orig_feature_means']
    perm_means     = perm['perm_feature_means']
    orig_stds      = perm['orig_feature_stds']
    feature_names  = perm['feature_names']

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE['s6'])

    # Panel A: MSE distribution per mouse
    ax = axes[0]
    mse_by_mouse = {m: [] for m in MICE}
    for r in perm_results:
        mse_by_mouse[r['mouse']].extend(r['mse_list'])

    bp_data = [mse_by_mouse[m] for m in MICE]
    bp = ax.boxplot(bp_data, positions=range(len(MICE)),
                    patch_artist=True, widths=0.6, showfliers=False)
    for patch, m in zip(bp['boxes'], MICE):
        patch.set_facecolor(MOUSE_COLORS[m])
        patch.set_alpha(0.7)
    ax.set_xticks(range(len(MICE)))
    ax.set_xticklabels(MICE)
    ax.set_ylabel('Motor output MSE\n(original vs permuted)')
    ax.set_title('A  Permutation disrupts\nmotor computation', fontweight='bold')

    # Panel B: replicate vs permutation violin
    ax = axes[1]
    all_perm_mse = [mse for r in perm_results for mse in r['mse_list']]
    vp = ax.violinplot([same_mouse_mse, all_perm_mse], positions=[0, 1],
                       showmedians=True, showextrema=False)
    vp['bodies'][0].set_facecolor('#4CAF50')
    vp['bodies'][1].set_facecolor('#F44336')
    for body in vp['bodies']:
        body.set_alpha(0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Same-mouse\nreplicate pairs', 'Original vs\npermuted'])
    ax.set_ylabel('Motor output MSE')
    ax.set_title(f'B  Permutation > replicate noise\np={p_perm:.4f}', fontweight='bold')

    # Panel C: structural features preserved
    ax = axes[2]
    x_pos = np.arange(len(feature_names))
    ax.scatter(x_pos - 0.15, orig_means, color='steelblue', s=30 * MARKER_SCALE,
               zorder=3, label='Original')
    ax.scatter(x_pos + 0.15, perm_means, color='tomato', s=30 * MARKER_SCALE,
               marker='D', zorder=3, label='Permuted')
    ax.errorbar(x_pos - 0.15, orig_means, yerr=orig_stds,
                fmt='none', color='steelblue', alpha=0.5, lw=0.8 * LW_SCALE)
    tick_labels = [_FEAT_LABELS.get(f, f) for f in feature_names]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels, fontsize=FS_ANNOT,
                       rotation=45, ha='right', rotation_mode='anchor')
    ax.set_ylabel('Feature value')
    ax.legend(frameon=False, fontsize=FS_ANNOT)
    ax.set_title('C  Structural features preserved\n(permutation control)',
                 fontweight='bold')

    plt.tight_layout()
    out = os.path.join(figures_dir, 'fig_s6_permutation.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return [out]
