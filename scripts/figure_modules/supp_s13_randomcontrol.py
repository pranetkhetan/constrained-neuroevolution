"""
Supplementary Figure S13 (fig:supp_random_control) — Random-agent permutation control.
Output: fig_supp_random_control.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ._style import apply_pub_style, EVOLVED_COL, RANDOM_COL, sig_label, bracket, FS_ANNOT, FS_SUPTITLE, FIGSIZE, LW_SCALE, MARKER_SCALE


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    d = store.phase3a()

    N_PERM = 20
    evol_perm_mse = d['evol_perm_mse']
    evol_rep_mse = d['evol_replicate_mse']
    rand_perm_mse = d['rand_perm_mse']
    rand_cross_mse = d['rand_cross_mse']

    n_evol = len(evol_perm_mse) // N_PERM
    n_rand = len(rand_perm_mse) // N_PERM

    evol_mean_rep = np.mean(evol_rep_mse)
    rand_mean_cross = np.mean(rand_cross_mse)

    evol_per_agent = np.array([
        np.mean(evol_perm_mse[i * N_PERM:(i + 1) * N_PERM]) / evol_mean_rep
        for i in range(n_evol)
    ])
    rand_per_agent = np.array([
        np.mean(rand_perm_mse[i * N_PERM:(i + 1) * N_PERM]) / rand_mean_cross
        for i in range(n_rand)
    ])

    _, p_mw = stats.mannwhitneyu(evol_per_agent, rand_per_agent, alternative='greater')

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE['s13'])
    fig.subplots_adjust(wspace=0.45)

    # Panel A: strip + box
    ax = axes[0]
    rng = np.random.RandomState(1)
    for i, (data, col, lbl) in enumerate([
        (evol_per_agent, EVOLVED_COL, "Evolved"),
        (rand_per_agent, RANDOM_COL, "Random"),
    ]):
        jitter = rng.uniform(-0.15, 0.15, len(data))
        ax.scatter([i + jitter[k] for k in range(len(data))], data,
                   color=col, alpha=0.55, s=10 * MARKER_SCALE, zorder=3, linewidths=0 * LW_SCALE)
        ax.boxplot(data, positions=[i], widths=0.3,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(color='white', linewidth=1.5 * LW_SCALE),
                   boxprops=dict(facecolor=col, alpha=0.35, linewidth=0.7 * LW_SCALE),
                   whiskerprops=dict(linewidth=0.7 * LW_SCALE),
                   capprops=dict(linewidth=0.7 * LW_SCALE))

    y_top = max(evol_per_agent.max(), rand_per_agent.max()) * 1.05
    bracket(ax, 0, 1, y_top, y_top * 0.04, sig_label(p_mw), fontsize=FS_ANNOT)
    ax.axhline(1.0, ls='--', lw=0.7 * LW_SCALE, color='0.55', label='Ratio = 1')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Evolved\n(n=54)", "Random\n(n=54)"])
    ax.set_ylabel("Permutation / baseline MSE ratio")
    ax.set_title("(A) Per-agent permutation sensitivity")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)

    # Panel B: mean +/- SD summary bar
    ax = axes[1]
    means = [evol_per_agent.mean(), rand_per_agent.mean()]
    sds = [evol_per_agent.std(ddof=1), rand_per_agent.std(ddof=1)]
    cols = [EVOLVED_COL, RANDOM_COL]

    ax.bar([0, 1], means, yerr=sds, color=cols, alpha=0.75, width=0.45,
           error_kw=dict(elinewidth=0.8, capsize=3), zorder=2)
    ax.axhline(1.0, ls='--', lw=0.7 * LW_SCALE, color='0.55')
    for idx, (m, s) in enumerate(zip(means, sds)):
        ax.text(idx, m + s + 0.02, f"{m:.2f}\u00d7",
                ha='center', va='bottom', fontsize=FS_ANNOT, color=cols[idx])

    y_top_b = max(m + s for m, s in zip(means, sds)) * 1.15
    bracket(ax, 0, 1, y_top_b, y_top_b * 0.05, sig_label(p_mw), fontsize=FS_ANNOT)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Evolved", "Random"])
    ax.set_ylabel("Mean permutation / baseline ratio")
    ax.set_title("(B) Mean \u00b1 SD summary")
    ax.set_ylim(0, max(m + s for m, s in zip(means, sds)) * 1.30)

    fig.suptitle("Permutation sensitivity: learned vs random circuits",
                 fontsize=FS_SUPTITLE, y=1.02)

    out = os.path.join(figures_dir, "fig_supp_random_control.png")
    fig.savefig(out)
    plt.close(fig)
    return [out]
