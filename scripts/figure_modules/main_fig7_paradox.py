"""
Figure 7 (fig:paradox) — The structural convergence finding is robust.
Panel A: η² vs achieved power (post-hoc power analysis).
Panel B: η² vs required replicates for 80% power.
Panel C: Raw vs BH-FDR p-values for all 18 circuit features.

Output: fig7_paradox.pdf
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from adjustText import adjust_text

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.stats import post_hoc_power, required_replicates
from ._style import FS_PANEL, FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT, FS_LEGEND, FS_SMALL


# Human-readable feature labels (shared across all three panels)
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


def _power_panel(ax, power_results: dict) -> None:
    """Panel A: observed η² vs achieved power."""
    feats    = sorted(power_results.keys())
    eta2s    = [power_results[f]['eta_squared'] for f in feats]
    powers   = [power_results[f]['power']       for f in feats]
    min_eta2 = power_results[feats[0]].get('min_detectable_eta2', 0.247)

    texts = []
    for f, e, pw in zip(feats, eta2s, powers):
        color = '#00A087' if pw >= 0.8 else '#DC9A6C'
        ax.scatter(e, pw, color=color, s=80, zorder=3, edgecolor='white', linewidth=0.4)
        label = FEAT_LABELS.get(f, f)
        texts.append(ax.text(e, pw, label, fontsize=FS_ANNOT))

    ax.axhline(0.8, ls='--', color='gray', lw=1.0, label='80% power threshold')
    ax.axvline(min_eta2, ls=':', color='gray', lw=1.0,
               label=f'Min detectable η²={min_eta2:.3f}')

    ax.set_xlabel('Observed η²', fontsize=FS_LABEL)
    ax.set_ylabel('Achieved Power', fontsize=FS_LABEL)
    ax.set_title('A. Post-Hoc Power Analysis', loc='left',
                 fontweight='bold', fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(-0.02, max(eta2s) + 0.06)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=FS_LEGEND)
    sns.despine(ax=ax)

    if texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.7),
                    expand_points=(1.5, 1.5), expand_text=(1.4, 1.4))


def _reps_panel(ax, reps_data: dict) -> None:
    """Panel B: observed η² vs required replicates for 80% power."""
    from scipy.stats import ncf, f as f_dist

    eta2_range = np.linspace(0.02, 0.45, 200)
    k = 9
    req_ns = []
    for e2 in eta2_range:
        f_sq = e2 / (1 - e2)
        lo, hi = 2, 500
        for _ in range(50):
            mid = (lo + hi) // 2
            n_total = mid * k
            df1, df2 = k - 1, n_total - k
            ncp    = f_sq * n_total
            f_crit = f_dist.ppf(0.95, df1, df2)
            pw     = 1 - ncf.cdf(f_crit, df1, df2, ncp)
            if pw < 0.8:
                lo = mid + 1
            else:
                hi = mid
        req_ns.append(hi)

    ax.plot(eta2_range, req_ns, '-', color='#3C5488', lw=1.5)
    ax.axhline(6, ls='--', color='#DC9A6C', lw=1.0, label='Current n = 6 per mouse')

    Y_MAX = 30
    MIN_LABEL_GAP = 1.1
    points = []
    for feat, r in reps_data.items():
        eta2  = r['eta_squared']
        req_n = r['required_n_per_group']
        if np.isinf(req_n) or eta2 <= 0:
            continue
        y_pt  = min(req_n, Y_MAX - 0.5)
        color = '#DC9A6C' if req_n > 6 else '#00A087'
        points.append((eta2, y_pt, color, FEAT_LABELS.get(feat, feat)))

    points.sort(key=lambda p: (p[1], p[0]))
    label_ys = [p[1] for p in points]
    for idx in range(1, len(label_ys)):
        if abs(points[idx][0] - points[idx - 1][0]) < 0.06:
            if label_ys[idx] - label_ys[idx - 1] < MIN_LABEL_GAP:
                label_ys[idx] = label_ys[idx - 1] + MIN_LABEL_GAP

    texts = []
    for (eta2, y_pt, color, label), label_y in zip(points, label_ys):
        ax.scatter(eta2, y_pt, color=color, s=80, zorder=3,
                   edgecolor='white', linewidth=0.4)
        t = ax.text(eta2 + 0.004, label_y, label, fontsize=FS_ANNOT, va='center')
        texts.append(t)

    ax.set_xlabel('Observed η²', fontsize=FS_LABEL)
    ax.set_ylabel('Required n per mouse\n(for 80% power)', fontsize=FS_LABEL)
    ax.set_title('B. Replicates Needed to Detect Effects', loc='left',
                 fontweight='bold', fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylim(0, Y_MAX)
    ax.set_xlim(0, 0.45)
    ax.legend(frameon=False, fontsize=FS_LEGEND)

    if texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.7),
                    expand_points=(1.5, 1.2), expand_text=(1.3, 1.1),
                    force_text=(0.3, 0.6))


def _pval_panel(ax, anova_results: dict) -> None:
    """Panel C: raw vs BH-FDR p-values for all 18 features.

    Visual point: all raw p-values exceed 0.45, far above the α=0.05 threshold,
    so the null result is robust regardless of correction method.
    Features sorted ascending by raw p (most borderline on left/bottom).
    """
    feats  = sorted(anova_results.keys(), key=lambda f: anova_results[f]['p_raw'])
    p_raws = [anova_results[f]['p_raw'] for f in feats]
    p_fdrs = [anova_results[f]['p_fdr'] for f in feats]

    y_pos = np.arange(len(feats))

    # Connecting lines between raw and FDR values
    for i, (pr, pf) in enumerate(zip(p_raws, p_fdrs)):
        ax.plot([pr, pf], [i, i], '-', color='#CCCCCC', lw=0.8, zorder=1)

    ax.scatter(p_fdrs, y_pos, marker='o', color='#3C5488',
               s=45, zorder=3, label='BH-FDR $p$')
    ax.scatter(p_raws, y_pos, marker='o', facecolors='none', edgecolors='#666666',
               linewidths=1.2, s=45, zorder=4, label='Raw $p$')

    ax.axvline(0.05, ls='--', color='#CC0000', lw=1.0, label='$\\alpha = 0.05$')

    ax.set_xlim(-0.02, 1.05)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FEAT_LABELS.get(f, f) for f in feats], fontsize=FS_SMALL)
    ax.set_xlabel('$p$-value', fontsize=FS_LABEL)
    ax.tick_params(axis='x', labelsize=FS_TICK)
    ax.legend(frameon=True, fontsize=FS_LEGEND, loc=(0.1,0.8))
    sns.despine(ax=ax)


def generate(store, figures_dir: str) -> list[str]:
    stats_results = store.stats_results()
    anova_results = stats_results['anova']

    power_results = post_hoc_power(anova_results, n_total=54, k_groups=9)
    reps_data     = required_replicates(anova_results)

    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(
        1, 3,
        wspace=0.35,
        left=0.06, right=0.97,
        top=0.93,  bottom=0.10,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    _power_panel(ax_a, power_results)
    _reps_panel(ax_b, reps_data)
    _pval_panel(ax_c, anova_results)

    out = os.path.join(figures_dir, 'fig7_paradox.pdf')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'Saved -> {out}')
    return [out]
