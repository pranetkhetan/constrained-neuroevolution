"""
Supplementary Figure — Per-mouse specialisation trajectories (fig:supp_spec_evol_per_mouse).

3x3 grid showing the specialisation index trajectory (mean ± SD across 6 replicates)
for each of the 9 mice individually. Complements fig2 Panel E (population-level view)
by showing between-replicate variance and per-mouse convergence speed.

Output: fig_supp_spec_evol_per_mouse.pdf
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from ._style import (
    apply_pub_style,
    FIGSIZE,
    FS_ANNOT,
    FS_LABEL,
    FS_TICK,
    MICE,
    MOUSE_COLORS,
    label_panel,
    pub_despine,
    save_figure,
    LW_SCALE, MARKER_SCALE,
)


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    se = store.spec_evol()

    gens     = np.array(se["SAMPLE_GENS"])  # (16,)
    sp_mean  = se["spec_mean"]              # (16, 9)
    sp_std   = se["spec_std"]               # (16, 9)
    sp_mice  = se["MICE"]

    fig, axes = plt.subplots(3, 3, figsize=FIGSIZE['supp_spec_evol'], sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.25, wspace=0.18,
                        left=0.08, right=0.97, top=0.95, bottom=0.08)

    for idx, mouse in enumerate(MICE):
        ax = axes[idx // 3, idx % 3]
        mi = sp_mice.index(mouse)
        color = MOUSE_COLORS[mouse]

        ax.fill_between(gens,
                        sp_mean[:, mi] - sp_std[:, mi],
                        sp_mean[:, mi] + sp_std[:, mi],
                        color=color, alpha=0.25)
        ax.plot(gens, sp_mean[:, mi], color=color, lw=2.0 * LW_SCALE)
        ax.axhline(0, ls="--", color="lightgray", lw=0.8 * LW_SCALE)

        ax.set_title(mouse, fontsize=FS_TICK + 1, color=color, fontweight="bold", pad=3)
        ax.tick_params(labelsize=FS_ANNOT)
        pub_despine(ax)

    # Shared axis labels on outer panels
    for ax in axes[2, :]:
        ax.set_xlabel("Generation", fontsize=FS_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel("Spec. index", fontsize=FS_LABEL)

    out = os.path.join(figures_dir, "fig_supp_spec_evol_per_mouse.pdf")
    return save_figure(fig, out)
