"""
Supplementary Figure S15 (fig:supp_generalist_formal) — Generalist vs specialist deltafit.
Output: fig_supp_generalist_formal.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_pub_style, MICE, OWN_COL, OTHER_COL, GEN_COL, sig_label, FS_ANNOT, FIGSIZE, LW_SCALE


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

    # Single panel: per-mouse Δfit (the summary dot-and-whisker previously in Panel B
    # is identical to main Figure fig:causal Panel D and was removed to avoid duplication).
    fig, ax = plt.subplots(1, 1, figsize=(FIGSIZE['supp_s15'][0] * 0.7, FIGSIZE['supp_s15'][1]))

    ax.bar(x - w, gen_means, width=w, color=GEN_COL, alpha=0.75,
           label="Generalist $\\Delta$fit", zorder=2)
    ax.errorbar(x - w, gen_means, yerr=gen_sds,
                fmt='none', ecolor=GEN_COL, elinewidth=0.8, capsize=2, zorder=3)
    ax.bar(x, spec_own, width=w, color=OWN_COL, alpha=0.75,
           label="Specialist $\\Delta$fit$_{\\rm own}$", zorder=2)
    ax.bar(x + w, spec_other, width=w, color=OTHER_COL, alpha=0.75,
           label="Specialist $\\Delta$fit$_{\\rm other}$", zorder=2)

    ax.text(0.02, 0.97,
            "Generalist: no own-mouse peak\n(mean at specialist-other level)",
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
    ax.set_title("Per-mouse $\\Delta$fit: generalist vs specialist")
    ax.legend(frameon=False, loc='lower right', fontsize=FS_ANNOT)
    ax.set_ylim(bottom=0)

    out = os.path.join(figures_dir, "fig_supp_generalist_formal.png")
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    return [out]
