"""
Supplementary Figure S17 (fig:supp_dynamics_null) — Dynamical null results.

Two-panel figure showing that aggregate dynamics cannot distinguish circuits
trained on different mice:
  A: Per-mouse maximal Lyapunov exponent (lambda_1) — all agents slightly stable,
     no between-mouse variation (ANOVA null).
  B: Within-mouse vs between-mouse activation trajectory cosine similarity
     (violin + strip) — distributions are indistinguishable (Mann-Whitney null).

Output: fig_s17_dynamics_null.pdf
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ._style import (
    apply_pub_style,
    FIGSIZE,
    EVOLVED_COL,
    FS_ANNOT,
    FS_LABEL,
    FS_PANEL,
    FS_SUPTITLE,
    FS_TICK,
    MICE,
    MOUSE_COLORS,
    LW_SCALE, MARKER_SCALE,
)


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    dyn_full = store.dynamics_results_full()
    dyn = store.dynamics_results()

    lya = dyn_full["lyapunov"]
    lya_by_mouse = lya["lya_by_mouse"]

    traj = dyn["trajectory"]
    within_sims = np.array(traj["within_sims"])
    between_sims = np.array(traj["between_sims"])
    rsm_p = traj["mannwhitney_p"]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE['supp_s17'])
    fig.subplots_adjust(wspace=0.35)
    rng = np.random.default_rng(1)

    # ── Panel A: per-mouse λ₁ strip plot ─────────────────────────────────
    ax = axes[0]

    for xi, mouse in enumerate(MICE):
        vals = np.array(lya_by_mouse[mouse])
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(
            np.full(len(vals), xi) + jitter,
            vals,
            color=MOUSE_COLORS[mouse],
            s=36 * MARKER_SCALE,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.5 * LW_SCALE,
            zorder=3,
        )
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([xi - 0.28, xi + 0.28], [med, med], color="black", lw=1.4 * LW_SCALE, zorder=4)
        ax.plot([xi, xi], [q1, q3], color="black", lw=0.9 * LW_SCALE, zorder=4)

    ax.axhline(0, ls="--", color="gray", lw=1.0 * LW_SCALE, alpha=0.8)
    ax.text(
        8.6, 0.003, "$\\lambda_1 = 0$\n(stability\nboundary)",
        fontsize=FS_ANNOT, color="gray", va="bottom", ha="right",
    )
    ax.set_xticks(range(len(MICE)))
    ax.set_xticklabels(MICE, fontsize=FS_TICK)
    ax.set_ylabel("Maximal Lyapunov exponent ($\\lambda_1$)", fontsize=FS_LABEL)
    ax.set_title(
        "A  Per-mouse $\\lambda_1$\n(all agents: $\\lambda_1 < 0$, ANOVA $p = 0.943$)",
        fontweight="bold",
        fontsize=FS_PANEL,
    )
    sns.despine(ax=ax, offset=5, trim=True)

    # ── Panel B: within vs between RSM violin ────────────────────────────
    ax = axes[1]

    BETWEEN_COL = "#DDDDDD"
    vp = ax.violinplot(
        [within_sims, between_sims],
        positions=[0, 1],
        showmedians=True,
        showextrema=False,
    )
    vp["bodies"][0].set_facecolor(EVOLVED_COL)
    vp["bodies"][0].set_alpha(0.55)
    vp["bodies"][1].set_facecolor(BETWEEN_COL)
    vp["bodies"][1].set_alpha(0.7)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(1.6)

    # Jittered strip overlay — subsample for readability
    w_idx = rng.choice(len(within_sims), size=min(len(within_sims), 80), replace=False)
    b_idx = rng.choice(len(between_sims), size=min(len(between_sims), 180), replace=False)
    ax.scatter(
        rng.uniform(-0.13, 0.13, len(w_idx)),
        within_sims[w_idx],
        color=EVOLVED_COL, s=14 * MARKER_SCALE, alpha=0.45, zorder=3,
    )
    ax.scatter(
        1 + rng.uniform(-0.13, 0.13, len(b_idx)),
        between_sims[b_idx],
        color="#999999", s=10 * MARKER_SCALE, alpha=0.30, zorder=3,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Within-mouse\npairs\n(n=135)", "Between-mouse\npairs\n(n=1296)"],
                       fontsize=FS_TICK)
    ax.set_ylabel("Cosine similarity (mean hidden state)", fontsize=FS_LABEL)
    p_str = f"p = {rsm_p:.3f}" if rsm_p >= 0.001 else "p < 0.001"
    ax.set_title(
        f"B  Trajectory RSM: within vs between mouse\n(Mann-Whitney {p_str}; n.s.)",
        fontweight="bold",
        fontsize=FS_PANEL,
    )
    sns.despine(ax=ax, offset=5, trim=True)

    out = os.path.join(figures_dir, "fig_s17_dynamics_null.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")
    return [out]
