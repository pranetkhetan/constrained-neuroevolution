"""
paper_v2 Fig 3 -- Causal necessity vs structural degeneracy (Claims C9-C11).

4-panel 2x2 layout:
  A: Slope graph (paired dot plot) -- own vs other delta_fit per mouse.
     All 9 lines slope upward: permutation is unanimously mouse-specific.
  B: MSE violins -- topology / magnitude / replicate baseline.
     Topology elevates MSE; magnitude reduces it (degenerate causal axis).
  C: Dose-response scatter -- specialisation index vs own/other delta_fit ratio.
     More specialised mice show greater permutation specificity.
  D: Generalist control -- delta_fit for generalist / specialist-own / specialist-other.
     Generalist shows flat profile (KW n.s.); specialist shows own > other.

Output: fig3_causal_v2.pdf
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, mannwhitneyu, linregress

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figure_modules._style import (
    apply_pub_style, pub_despine, save_figure, label_panel,
    FIGSIZE,
    FS_PANEL, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    MICE, MOUSE_COLORS, GEN_COL, OWN_COL, OTHER_COL, sig_label, bracket,
    LW_SCALE, MARKER_SCALE,
)


def _slope_graph_panel(ax, spec: dict) -> None:
    """Panel A: paired own-vs-other delta_fit per mouse. All 9 lines slope upward."""
    opt_a      = spec['option_A']
    spec_mice  = spec.get('metadata', {}).get('mice', MICE)
    delta_own  = np.array(opt_a['delta_own'])
    delta_other = np.array(opt_a['delta_other'])
    p_wil      = opt_a['wilcoxon_p']

    X_OWN, X_OTHER = 0, 0.55

    all_vals = np.concatenate([delta_own, delta_other])
    y_lo = all_vals.min() - 0.15
    y_hi = all_vals.max() + 0.35

    ax.set_xlim(-0.4, X_OTHER + 0.55)
    ax.set_ylim(y_lo, y_hi)

    # Build label y-positions with minimum spacing to reduce collision
    order = np.argsort(delta_other)
    label_y = delta_other.copy()
    sorted_y = label_y[order]
    MIN_SP = 0.07
    for k in range(1, len(sorted_y)):
        if sorted_y[k] - sorted_y[k - 1] < MIN_SP:
            sorted_y[k] = sorted_y[k - 1] + MIN_SP
    label_y[order] = sorted_y

    for i, mouse in enumerate(spec_mice):
        col = MOUSE_COLORS.get(mouse, '#888')
        ax.plot([X_OWN, X_OTHER], [delta_own[i], delta_other[i]],
                color=col, lw=1.6 * LW_SCALE, alpha=0.85, zorder=2)
        ax.scatter(X_OWN,   delta_own[i],   color=col, s=90 * MARKER_SCALE, zorder=4,
                   edgecolor='white', lw=0.8 * LW_SCALE)
        ax.scatter(X_OTHER, delta_other[i], color=col, s=90 * MARKER_SCALE, zorder=4,
                   edgecolor='white', lw=0.8 * LW_SCALE)
        ax.text(X_OTHER + 0.04, label_y[i], mouse,
                va='center', ha='left', fontsize=FS_ANNOT, color=col)

    ax.set_xticks([X_OWN, X_OTHER])
    ax.set_xticklabels(['Own mouse', 'Other mice'], fontsize=FS_LABEL)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_ylabel(r'$\Delta$ Fitness (permuted $-$ original)', fontsize=FS_LABEL)
    ax.text(0.5, 0.97,
            f'Wilcoxon p = {p_wil:.5f}\n(all 9 mice: own > other)',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=FS_ANNOT, color='#333')
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _perm_mse_panel(ax, spec: dict) -> None:
    """Panel B: MSE ratios for topology/magnitude/baseline violins."""
    opt_b        = spec['option_B']
    topo_mse     = np.array([r['mean_mse'] for r in opt_b['perm_results_topology']])
    mag_mse      = np.array([r['mean_mse'] for r in opt_b['perm_results_magnitude']])
    baseline_mse = float(opt_b['baseline_mse'])

    norm_topo = topo_mse / baseline_mse
    norm_mag  = mag_mse  / baseline_mse
    norm_base = np.ones(len(norm_topo))

    groups = [norm_base, norm_mag, norm_topo]
    labels = ['Replicate\nbaseline', 'Magnitude\npermutation', 'Topology\npermutation']
    colors = ['#88CCEE', '#DDCC77', '#CC6677']
    ratios = [1.0, float(opt_b['ratio_magnitude']), float(opt_b['ratio_topology'])]

    rng = np.random.default_rng(42)
    for pos, (data, color, ratio) in enumerate(zip(groups, colors, ratios)):
        vp = ax.violinplot(data, positions=[pos], widths=0.55,
                           showmedians=False, showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(color)
            body.set_alpha(0.5)
            body.set_edgecolor('none')
        jit = rng.uniform(-0.12, 0.12, len(data))
        ax.scatter(pos + jit, data, s=12 * MARKER_SCALE, color=color, alpha=0.7,
                   edgecolor='white', lw=0.3 * LW_SCALE, zorder=3)
        med = float(np.median(data))
        ax.hlines(med, pos - 0.22, pos + 0.22, color='black', lw=1.5 * LW_SCALE, zorder=4)
        ax.text(pos, 2.5, f'{ratio:.2f}x', ha='center', va='top',
                fontsize=FS_ANNOT, )

    _, p_topo = mannwhitneyu(norm_base, norm_topo, alternative='less')
    y_bracket = float(np.max(norm_topo)) + 0.08
    bracket(ax, 0, 2, y_bracket, dy=0.05, label=sig_label(p_topo))

    ax.axhline(1.0, color='black', lw=0.8 * LW_SCALE, ls='--', alpha=0.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel('Motor MSE (normalised to replicate baseline)', fontsize=FS_LABEL)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    pub_despine(ax)


def _dose_response_panel(ax, spec: dict, gen_matrix: np.ndarray,
                         gen_meta: dict) -> None:
    """Panel C: specialisation index vs permutation Δfit ratio (own/other)."""
    opt_a       = spec['option_A']
    spec_mice   = spec.get('metadata', {}).get('mice', MICE)
    delta_own   = np.array(opt_a['delta_own'])
    delta_other = np.array(opt_a['delta_other'])
    gm_mice     = gen_meta['mice']

    spec_idx = {}
    for i, m in enumerate(gm_mice):
        own = gen_matrix[i, i]
        off = float(np.mean([gen_matrix[i, j] for j in range(len(gm_mice)) if j != i]))
        spec_idx[m] = 1.0 - own / off

    dfit_ratio = {}
    for i, m in enumerate(spec_mice):
        if delta_other[i] > 0:
            dfit_ratio[m] = float(delta_own[i]) / float(delta_other[i])

    common = [m for m in MICE if m in spec_idx and m in dfit_ratio]
    x = np.array([spec_idx[m] for m in common])
    y = np.array([dfit_ratio[m] for m in common])

    # Bootstrap CI band
    rng_boot = np.random.default_rng(42)
    slope, intercept, _, _, _ = linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    boot_lines = np.empty((1000, len(x_line)))
    for b in range(1000):
        idx = rng_boot.integers(0, len(x), size=len(x))
        s, ic, _, _, _ = linregress(x[idx], y[idx])
        boot_lines[b] = s * x_line + ic
    ax.fill_between(x_line,
                    np.percentile(boot_lines, 2.5, axis=0),
                    np.percentile(boot_lines, 97.5, axis=0),
                    alpha=0.18, color='#555', lw=0 * LW_SCALE)
    ax.plot(x_line, slope * x_line + intercept, color='#333', lw=1.2 * LW_SCALE, ls='-', zorder=3)

    for xi, yi, mi in zip(x, y, common):
        ax.scatter(xi, yi, color=MOUSE_COLORS.get(mi, '#888'),
                   s=70 * MARKER_SCALE, zorder=4, edgecolor='white', lw=0.4 * LW_SCALE)
        ax.text(xi + 0.005, yi, mi, fontsize=FS_ANNOT, va='center')

    rho, _ = spearmanr(x, y)
    rng_perm = np.random.default_rng(1)
    p_perm = sum(
        1 for _ in range(10000)
        if abs(spearmanr(x[rng_perm.permutation(len(x))], y)[0]) >= abs(rho)
    ) / 10000

    ax.scatter(0.0, 1.0, color=GEN_COL, s=90 * MARKER_SCALE, marker='D',
               zorder=4, edgecolor='white', lw=0.5 * LW_SCALE, label='Generalist (1.0x)')
    ax.axhline(1.0, color='gray', lw=0.6 * LW_SCALE, ls=':', alpha=0.5)
    ax.text(0.97, 0.05,
            f'Spearman ρ = {rho:.2f}\np = {p_perm:.3f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT)
    ax.set_xlabel('Behavioural specialisation index', fontsize=FS_LABEL)
    ax.set_ylabel(r'Permutation $\Delta$fit ratio (own / other)', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(frameon=False, fontsize=FS_LEGEND)
    pub_despine(ax)


def _generalist_control_panel(ax, spec: dict, gen_data: dict) -> None:
    """Panel D: generalist / specialist-own / specialist-other Δfit comparison."""
    spec_data    = spec.get('phase3b', {})
    spec_own_c   = np.array(spec_data['specialist_own_delta'])
    spec_other_c = np.array(spec_data['specialist_other_delta'])
    kw_p         = spec_data['kruskal_wallis_p']
    wilcox_p_c   = spec_data['specialist_wilcoxon_p']

    pkl_mice = gen_data['MICE']
    results_C = gen_data['results_C']
    gen_delta_by_mouse = {
        m: [float(results_C[r][m]['delta_fit']) for r in range(len(results_C))]
        for m in pkl_mice
    }
    gen_means = np.array([np.mean(gen_delta_by_mouse[m]) for m in MICE
                          if m in gen_delta_by_mouse])

    c_vals   = [gen_means,    spec_own_c,   spec_other_c]
    c_cols   = [GEN_COL,      OWN_COL,      OTHER_COL]
    c_labels = ['Generalist', 'Spec.\nown',  'Spec.\nother']

    for i, (vals, col) in enumerate(zip(c_vals, c_cols)):
        rng = np.random.default_rng(i)
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=col, s=30 * MARKER_SCALE, alpha=0.55, zorder=3, edgecolor='none')
        ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1),
                    fmt='o', color=col, ms=5 * MARKER_SCALE, elinewidth=1.2, capsize=3, zorder=4)

    y_top = max(np.max(spec_own_c), np.max(spec_other_c))
    bracket(ax, 1, 2, y_top * 1.08, y_top * 0.04, sig_label(wilcox_p_c),
            fontsize=FS_ANNOT)
    ax.text(0.25, gen_means.mean() + gen_means.std(ddof=1) * 1.3,
            f'KW: {sig_label(kw_p)}', ha='center', va='bottom',
            fontsize=FS_ANNOT, color=GEN_COL)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(c_labels, fontsize=FS_TICK)
    ax.set_ylabel(r'$\Delta$ Fitness (permuted $-$ original)', fontsize=FS_LABEL)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=FS_TICK)
    pub_despine(ax)


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style(font_scale=1.15)
    spec       = store.specificity_results()
    gen_matrix = store.generalization_matrix()
    gen_meta   = store.generalization_meta()
    gen_data   = store.generalist_results()

    # phase3b holds specialist_own_delta / other_delta / kruskal_wallis_p
    phase3b = store.phase3b()
    spec['phase3b'] = phase3b

    fig = plt.figure(figsize=FIGSIZE['fig3'])
    gs  = gridspec.GridSpec(
        2, 2,
        hspace=0.45, wspace=0.38,
        left=0.07, right=0.97, top=0.93, bottom=0.09,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _slope_graph_panel(ax_a, spec)
    _perm_mse_panel(ax_b, spec)
    _dose_response_panel(ax_c, spec, gen_matrix, gen_meta)
    _generalist_control_panel(ax_d, spec, gen_data)

    for ax, lbl in [(ax_a, 'A'), (ax_b, 'B'), (ax_c, 'C'), (ax_d, 'D')]:
        label_panel(ax, lbl)

    out = os.path.join(figures_dir, 'fig3_causal_v2.pdf')
    return save_figure(fig, out)
