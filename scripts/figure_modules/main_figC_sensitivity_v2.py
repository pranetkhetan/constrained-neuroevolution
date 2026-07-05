"""
paper_v2 New Fig C -- Functional sensitivity commitment (Claims C12-C14).
4-panel 2x2 layout:
  A: Topology similarity -- generalist vs specialist, per-mouse box + strip (MW p=0.856).
     The null result IS the claim; annotate p prominently.
  B: Fitness cost of generalism per mouse -- horizontal bars, % cost.
  C: Sensitivity variance per neuron -- specialist vs generalist, LOG scale.
     Uses normalized generalist variance (gen_sens_var_norm, apples-to-apples with spec).
     Mean ratio ~2.6x. Log scale useful: per-neuron values span ~4 orders of magnitude.
     Individual neuron points shown (14 per group).
  D: Sensitivity variance trajectory over 150 generations: specialist mean (blue) vs
     generalist mean (orange), log scale. Panel C shows the gen-150 endpoint; Panel D
     shows the process. Specialist variance remains elevated above the generalist level
     from approximately generation 20 onward.

Output: figC_sensitivity_v2.pdf
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
from scipy.stats import mannwhitneyu

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figure_modules._style import (
    apply_pub_style, pub_despine, save_figure, label_panel,
    FIGSIZE,
    FS_PANEL, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT, FS_SMALL,
    MICE, MOUSE_COLORS, EVOLVED_COL, GEN_COL, sig_label,
    LW_SCALE, MARKER_SCALE,
)


def _load_pkl(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        class _U(pickle.Unpickler):
            def find_class(self, mod, name):
                if mod.startswith("cupy"):
                    mod = mod.replace("cupy._core.core","numpy").replace("cupy","numpy")
                elif mod == "core" or mod.startswith("core."):
                    mod = "numpy." + mod
                return super().find_class(mod, name)
        with open(path, "rb") as f:
            return _U(f).load()


def _topo_similarity_panel(ax, A6: dict) -> None:
    """Panel A: topology cosine similarity gen vs spec -- null result is the claim."""
    gen_sims  = np.array(A6["gen_topo_sims"])
    spec_sims = np.array(A6["spec_topo_sims"])
    mw_p      = float(A6["mw_p_gen_spec_topo"])

    data   = [spec_sims, gen_sims]
    labels = ["Specialists", "Generalists"]
    colors = [EVOLVED_COL, GEN_COL]

    rng = np.random.default_rng(42)
    for i, (d, label, color) in enumerate(zip(data, labels, colors)):
        bp = ax.boxplot(d, positions=[i], widths=0.4, patch_artist=True,
                        medianprops=dict(color="black", lw=1.5 * LW_SCALE),
                        flierprops=dict(marker="", markersize=0 * MARKER_SCALE),
                        showfliers=False)
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.5)
        jitter = rng.uniform(-0.12, 0.12, len(d))
        ec = EVOLVED_COL if color == EVOLVED_COL else "black"
        ax.scatter(np.full(len(d), i) + jitter, d,
                   s=15 * MARKER_SCALE, color=color, alpha=0.5, zorder=3, edgecolor=ec)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"Specialists\n(n={len(spec_sims)})", f"Generalists\n(n={len(gen_sims)})"],
        fontsize=FS_TICK,
    )
    ax.set_ylabel("Topology cosine similarity", fontsize=FS_LABEL)
    ax.tick_params(axis="y", labelsize=FS_TICK)

    # Annotate MW p -- the null IS the result
    y_max = max(np.max(spec_sims), np.max(gen_sims))
    ax.plot([0, 1], [y_max + 0.03, y_max + 0.03], color="black", lw=0.8 * LW_SCALE)
    ax.text(0.5, y_max + 0.035, f"MW p = {mw_p:.3f} (n.s.)",
            ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold")

    pub_despine(ax)


def _fitness_cost_panel(ax, A6: dict) -> None:
    """Panel B: fitness cost of generalism per mouse (% worse than specialist)."""
    gen_cost = np.array(A6["gen_cost_pct"])   # (9,) per mouse
    order    = np.argsort(gen_cost)[::-1]
    s_mice   = [MICE[i] for i in order]
    s_cost   = [gen_cost[i] for i in order]
    colors   = [MOUSE_COLORS.get(m, "#888") for m in s_mice]

    y_pos = np.arange(len(s_mice))
    ax.barh(y_pos, s_cost, color=colors, height=0.6, edgecolor="white")

    mean_cost = float(np.mean(gen_cost))
    ax.axvline(mean_cost, color="black", lw=1.0 * LW_SCALE, ls="--",
               label=f"Mean = {mean_cost:.1f}%")

    for y, v in zip(y_pos, s_cost):
        ax.text(v + 0.5, y, f"{v:.1f}%", va="center", fontsize=FS_ANNOT)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(s_mice, fontsize=FS_TICK)
    ax.set_xlabel("Fitness cost of generalism (%)", fontsize=FS_LABEL)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.legend(frameon=False, fontsize=FS_LEGEND)
    pub_despine(ax)


def _sensitivity_variance_panel(ax, A6: dict) -> None:
    """Panel C: sensitivity variance per neuron, LOG scale.
    Uses normalized generalist variance (gen_sens_var_norm) for apples-to-apples comparison.
    Mean ratio ~2.6x. Log scale appropriate: per-neuron values span ~4 orders of magnitude.
    Individual neuron points shown. Ref: seaborn stripplot.
    """
    spec_var = np.array(A6["spec_sens_var"])       # (14,) normalized sensitivity variance
    gen_var  = np.array(A6["gen_sens_var_norm"])   # (14,) normalized — apples-to-apples

    neuron_labels = ["S0","S1","S2","S3","S4","S5",
                     "I0","I1","I2","I3","I4","I5","Spd","Trn"]
    x_spec = np.zeros(14)
    x_gen  = np.ones(14)

    rng = np.random.default_rng(42)
    jit = rng.uniform(-0.12, 0.12, 14)

    ax.scatter(x_spec + jit, spec_var, s=35 * MARKER_SCALE, color=EVOLVED_COL,
               alpha=0.8, zorder=3, edgecolor=EVOLVED_COL, lw=0.3 * LW_SCALE,
               label=f"Specialists (mean={np.mean(spec_var):.3f})")
    ax.scatter(x_gen  + jit, gen_var,  s=35 * MARKER_SCALE, color=GEN_COL,
               alpha=0.8, zorder=3, edgecolor="black", lw=0.3 * LW_SCALE,
               label=f"Generalists (mean={np.mean(gen_var):.3f})")

    # Mean lines
    ax.hlines(np.mean(spec_var), -0.25, 0.25, color=EVOLVED_COL, lw=2.0 * LW_SCALE, zorder=4)
    ax.hlines(np.mean(gen_var),   0.75, 1.25, color=GEN_COL,     lw=2.0 * LW_SCALE, zorder=4)

    # Annotate ratio + MW p-value. Report to one decimal (the ':.0f' rounding
    # mis-displayed 2.58x as "3x"); the caption gives the 2.2-2.6x range and the
    # mouse-level primary test.
    ratio = np.mean(spec_var) / np.mean(gen_var)
    _, p_mw = mannwhitneyu(spec_var, gen_var, alternative="greater")
    ax.text(0.5, 0.97,
            f"Ratio: {ratio:.1f}x  ({sig_label(p_mw)})",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FS_ANNOT, fontweight="bold", color="#CC0000")

    ax.set_yscale("log")
    # Floor at 1e-4: smallest non-zero normalized value is ~1e-4; clips only true zeros.
    ax.set_ylim(bottom=1e-4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Specialists", "Generalists"], fontsize=FS_TICK)
    ax.set_ylabel("Sensitivity variance (log scale)", fontsize=FS_LABEL)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.legend(frameon=False, fontsize=FS_LEGEND, loc="lower left")
    pub_despine(ax)


def _sens_var_traj_panel(ax, sce: dict) -> None:
    """Panel D: sensitivity variance trajectory over generations.

    Shows spec_var_mean_traj (specialists, blue) vs gen_var_mean_traj (generalists, orange)
    on a log scale. Panel C shows the gen-150 endpoint; Panel D shows the process.
    Specialist variance diverges from generalist level around generation 20 and remains
    elevated throughout, demonstrating that selection progressively builds commitment.
    """
    gens     = np.array(sce["sample_gens"])           # (16,)
    spec_var = np.array(sce["spec_var_mean_traj"])   # (16,) mean across mice
    gen_var  = np.array(sce["gen_var_mean_traj"])    # (16,) mean across mice

    ax.plot(gens, spec_var, color=EVOLVED_COL, lw=2.2 * LW_SCALE, label="Specialists")
    ax.plot(gens, gen_var,  color=GEN_COL,     lw=2.2 * LW_SCALE, label="Generalists", ls="--")

    gen20_idx = int(np.searchsorted(gens, 20))

    ax.set_yscale("log")
    y_ann = gen_var.min() * 1.3
    ax.axvline(gens[min(gen20_idx, len(gens) - 1)], ls="--",
               color="lightgray", lw=1.0 * LW_SCALE, zorder=0)
    ax.text(gens[min(gen20_idx, len(gens) - 1)] + 2, y_ann,
            "Gen ~20", fontsize=FS_ANNOT, color="gray", va="bottom")
    ax.set_xlabel("Generation", fontsize=FS_LABEL)
    ax.set_ylabel("Mean sensitivity variance (log)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="upper left")
    pub_despine(ax)


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style(font_scale=1.6)
    analysis_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "analysis", "degeneracy_analyses")
    A6 = _load_pkl(os.path.join(analysis_dir, "A6_results.pkl"))

    sce = store.sens_commitment_evo()

    # ── 4-panel 2×2 layout ────────────────────────────────────────────────
    fig = plt.figure(figsize=FIGSIZE['figC'])
    gs  = gridspec.GridSpec(
        2, 2,
        hspace=0.42, wspace=0.38,
        left=0.07, right=0.97, top=0.92, bottom=0.09,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _topo_similarity_panel(ax_a, A6)
    _fitness_cost_panel(ax_b, A6)
    _sensitivity_variance_panel(ax_c, A6)
    _sens_var_traj_panel(ax_d, sce)

    for ax, lbl in [(ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D")]:
        label_panel(ax, lbl)

    out = os.path.join(figures_dir, "figC_sensitivity_v2.pdf")
    return save_figure(fig, out)
