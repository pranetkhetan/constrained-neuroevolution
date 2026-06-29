"""
Supplementary Figure — Sensitivity commitment temporal co-development (fig:supp_sens_commitment_evo).

Three-panel figure showing how functional sensitivity commitment co-develops with
behavioural specialisation over the course of evolution (nb18):

  A: Population mean behavioural specialisation index ± SD over 150 generations.
     Dashed line marks gen 20 — the approximate onset of divergence in Panels B/C.
  B: Mean per-neuron sensitivity variance over generations: specialists (blue)
     vs generalists (orange), log scale. Specialist variance remains above
     generalist from approximately gen 20 onward.
  C: Within-mouse pairwise cosine similarity of sensitivity profile vectors
     (solid) vs between-mouse baseline (dashed). Within-mouse similarity falls
     below the between-mouse baseline over training, indicating that within-mouse
     circuits develop idiosyncratic mouse-specific sensitivity profiles.

All panels share the x-axis (generations). Gen 20 dashed line marks onset of
divergence visible across all three panels.

Output: fig_supp_sens_commitment_evo.pdf
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from ._style import (
    apply_pub_style,
    FIGSIZE,
    EVOLVED_COL,
    FS_ANNOT,
    FS_LABEL,
    FS_PANEL,
    FS_TICK,
    GEN_COL,
    label_panel,
    pub_despine,
    save_figure,
    LW_SCALE, MARKER_SCALE,
)

GEN20_ALPHA = 0.55


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    se  = store.spec_evol()
    sc  = store.sens_commitment_evo()

    gens       = np.array(se["SAMPLE_GENS"])      # (16,)
    sp_mean    = se["spec_mean"]                  # (16, 9)
    pop_mean   = sp_mean.mean(axis=1)
    pop_std    = sp_mean.std(axis=1)

    sc_gens    = np.array(sc["sample_gens"])      # (16,)
    spec_var   = sc["spec_var_mean_traj"]         # (16,) mean sens var — specialists
    gen_var    = sc["gen_var_mean_traj"]          # (16,) mean sens var — generalists
    within_sim = sc["within_sim_mean"]            # (16,)
    within_std = sc["within_sim_std"]             # (16,)
    between_sim= sc["between_sim_mean"]           # (16,)

    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE['supp_sens_comm'], sharex=True)
    fig.subplots_adjust(hspace=0.12, left=0.14, right=0.96, top=0.95, bottom=0.08)

    for ax in axes:
        ax.axvline(20, color="gray", lw=1.0 * LW_SCALE, ls="--", alpha=GEN20_ALPHA, zorder=0)

    # ── Panel A: behavioural specialisation trajectory ────────────────────
    ax = axes[0]
    ax.fill_between(gens, pop_mean - pop_std, pop_mean + pop_std,
                    color=EVOLVED_COL, alpha=0.20)
    ax.plot(gens, pop_mean, color=EVOLVED_COL, lw=2.0 * LW_SCALE,
            label="Specialists mean ± SD")
    ax.axhline(0, ls=":", color="lightgray", lw=0.8 * LW_SCALE)
    ax.set_ylabel("Specialisation index", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_ANNOT, frameon=False, loc="lower right")
    pub_despine(ax)
    label_panel(ax, "A")

    # ── Panel B: sensitivity variance over generations (log scale) ─────────
    ax = axes[1]
    ax.plot(sc_gens, spec_var, color=EVOLVED_COL, lw=1.8 * LW_SCALE,
            label="Specialists", marker="*", markersize=5 * MARKER_SCALE, markevery=[-1])
    ax.plot(sc_gens, gen_var,  color=GEN_COL,     lw=1.8 * LW_SCALE,
            label="Generalists", marker="*", markersize=5 * MARKER_SCALE, markevery=[-1])
    ax.set_yscale("log")
    ax.set_ylabel("Mean sens. variance\n(log scale)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_ANNOT, frameon=False, loc="upper right")
    pub_despine(ax)
    label_panel(ax, "B")

    # ── Panel C: within vs between-mouse sensitivity cosine similarity ─────
    ax = axes[2]
    ax.fill_between(sc_gens,
                    within_sim - within_std,
                    within_sim + within_std,
                    color=EVOLVED_COL, alpha=0.18)
    ax.plot(sc_gens, within_sim, color=EVOLVED_COL, lw=2.0 * LW_SCALE,
            label="Within-mouse (mean ± SD)")
    ax.plot(sc_gens, between_sim, color="gray", lw=1.5 * LW_SCALE, ls="--",
            label="Between-mouse baseline")
    ax.set_xlabel("Generation", fontsize=FS_LABEL)
    ax.set_ylabel("Cosine similarity\n(sensitivity profiles)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_ANNOT, frameon=False, loc="upper right")
    pub_despine(ax)
    label_panel(ax, "C")

    # Gen-20 annotation on Panel A only
    y_top = axes[0].get_ylim()[1] if axes[0].get_ylim()[1] > 0 else 0.35
    axes[0].text(21, 0.02, "gen 20", fontsize=FS_ANNOT - 1,
                 color="gray", va="bottom", ha="left")

    out = os.path.join(figures_dir, "fig_supp_sens_commitment_evo.pdf")
    return save_figure(fig, out)
