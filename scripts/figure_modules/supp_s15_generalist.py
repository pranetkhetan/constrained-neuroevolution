"""
Supplementary Figure S15 (fig:supp_generalist_formal) — Generalist vs specialist deltafit.
Output: fig_supp_generalist_formal.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_pub_style, MICE, OWN_COL, OTHER_COL, GEN_COL, sig_label, bracket, FS_ANNOT, FS_SUPTITLE, FIGSIZE, LW_SCALE, MARKER_SCALE


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    gen_data = store.generalist_results()
    spec_data = store.phase3b()

    # Derive per-mouse Δfit from results_C (pkl doesn't store a flat dict).
    results_C = gen_data['results_C']
    pkl_mice  = gen_data['MICE']
    gen_delta = {
        m: [float(results_C[r][m]['delta_fit']) for r in range(len(results_C))]
        for m in pkl_mice
    }
    spec_own = np.array(spec_data['specialist_own_delta'])
    spec_other = np.array(spec_data['specialist_other_delta'])
    kw_p = spec_data['kruskal_wallis_p']
    wilcox_p = spec_data['specialist_wilcoxon_p']

    gen_means = np.array([np.mean(gen_delta[m]) for m in MICE])
    gen_sds = np.array([np.std(gen_delta[m], ddof=1) for m in MICE])

    x = np.arange(len(MICE))
    w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE['supp_s15'], gridspec_kw={'width_ratios': [2, 1]})

    # Panel A: per-mouse deltafit
    ax = axes[0]
    ax.bar(x - w, gen_means, width=w, color=GEN_COL, alpha=0.75,
           label="Generalist $\\Delta$fit", zorder=2)
    ax.errorbar(x - w, gen_means, yerr=gen_sds,
                fmt='none', ecolor=GEN_COL, elinewidth=0.8, capsize=2, zorder=3)
    ax.bar(x, spec_own, width=w, color=OWN_COL, alpha=0.75,
           label="Specialist $\\Delta$fit$_{\\rm own}$", zorder=2)
    ax.bar(x + w, spec_other, width=w, color=OTHER_COL, alpha=0.75,
           label="Specialist $\\Delta$fit$_{\\rm other}$", zorder=2)

    ax.text(0.02, 0.97,
            f"Generalist KW: {sig_label(kw_p)}\n(flat profile confirmed)",
            transform=ax.transAxes, va='top', ha='left', fontsize=FS_ANNOT,
            color=GEN_COL,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GEN_COL,
                      lw=0.6 * LW_SCALE, alpha=0.9))
    ax.text(0.98, 0.97,
            f"Specialist own > other\n{sig_label(wilcox_p)} (Wilcoxon, n=9)",
            transform=ax.transAxes, va='top', ha='right', fontsize=FS_ANNOT,
            color=OWN_COL,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=OWN_COL,
                      lw=0.6 * LW_SCALE, alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(MICE)
    ax.set_ylabel("$\\Delta$fit (permuted $-$ original fitness)")
    ax.set_title("(A) Per-mouse $\\Delta$fit: generalist vs specialist")
    ax.legend(frameon=False, loc='lower right', fontsize=FS_ANNOT)
    ax.set_ylim(bottom=0)

    # Panel B: summary dot-and-whisker
    ax = axes[1]
    categories = ["Generalist\n(all mice)", "Specialist\n$\\Delta$fit$_{\\rm own}$",
                  "Specialist\n$\\Delta$fit$_{\\rm other}$"]
    values = [gen_means.mean(), spec_own.mean(), spec_other.mean()]
    errors = [gen_means.std(ddof=1), spec_own.std(ddof=1), spec_other.std(ddof=1)]
    colors = [GEN_COL, OWN_COL, OTHER_COL]

    for i, (v, e, c) in enumerate(zip(values, errors, colors)):
        ax.errorbar(i, v, yerr=e, fmt='o', color=c, ms=5 * MARKER_SCALE,
                    elinewidth=1.0, capsize=3, zorder=3)

    y_br = max(v + e for v, e in zip(values, errors)) * 1.04
    bracket(ax, 1, 2, y_br, 0.04, sig_label(wilcox_p), fontsize=FS_ANNOT)

    ax.set_xticks(range(3))
    ax.set_xticklabels(categories, fontsize=FS_ANNOT)
    ax.set_ylabel("Mean $\\Delta$fit")
    ax.set_title("(B) Summary")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(bottom=0)

    fig.suptitle("Generalist vs specialist permutation $\\Delta$fit profiles",
                 fontsize=FS_SUPTITLE, y=1.01)
#     fig.tight_layout()

    out = os.path.join(figures_dir, "fig_supp_generalist_formal.png")
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    return [out]
