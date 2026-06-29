"""
paper_v2 Fig 2 -- The degeneracy paradox (Claims C4-C5).
5-panel layout (3 top + 2 bottom):
  A: 54x54 weight cosine similarity heatmap (structural compression, no mouse block structure)
  B: Raw vs BH-FDR p-values connected dotplot, 18 circuit features (0/18 null)
  E: Specialisation index over generations -- coloured per-mouse traces + pop mean (rises 0→0.27)
     vs generalist mean |bias| (flat near 0). Establishes temporal causality.
  C: Per-mouse behavioural specialisation index bars (behavioral individuation non-zero)
  D: 9x9 cross-mouse generalisation fitness matrix (specialisation structure)

Output: fig2_paradox_v2.pdf
"""

import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.metrics.pairwise import cosine_similarity

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)


def _load_pkl(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (ModuleNotFoundError, AttributeError):
        class _U(pickle.Unpickler):
            def find_class(self, mod, name):
                if mod.startswith("cupy"):
                    mod = mod.replace("cupy._core.core", "numpy").replace("cupy", "numpy")
                return super().find_class(mod, name)
        with open(path, "rb") as f:
            return _U(f).load()

from scripts.figures_logic import _panel_weight_similarity
from scripts.figure_modules._style import (
    apply_pub_style, pub_despine, save_figure, label_panel,
    FIGSIZE,
    FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    EVOLVED_COL, MOUSE_COLORS, RANDOM_COL,
    LW_SCALE, MARKER_SCALE,
)
from scripts.figure_modules.main_fig7_paradox import _pval_panel


def _gen_matrix_panel(ax, gen_matrix: np.ndarray, mice: list) -> None:
    """Panel D: 9x9 generalisation matrix, diverging palette centred on off-diagonal mean."""
    n = len(mice)
    off_diag_vals = [gen_matrix[i, j] for i in range(n) for j in range(n) if i != j]
    centre = float(np.mean(off_diag_vals))
    vmin = centre - 0.45
    vmax = centre + 0.45

    im = ax.imshow(gen_matrix, aspect="auto", cmap="RdYlGn_r",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.85, label="Fitness (lower = better)")

    for i in range(n):
        diag = gen_matrix[i, i]
        off = float(np.mean([gen_matrix[i, j] for j in range(n) if j != i]))
        idx = 1.0 - diag / off
        ax.text(i, i, f"{idx:.2f}", ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color="black")
        ax.add_patch(mpatches.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="black", lw=1.8 * LW_SCALE, zorder=5))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(mice, fontsize=FS_TICK, rotation=45, ha="right")
    ax.set_yticklabels(mice, fontsize=FS_TICK)
    ax.set_xlabel("Evaluated on", fontsize=FS_LABEL)
    ax.set_ylabel("Trained on", fontsize=FS_LABEL)
    ax.axhline(2.5, color="white", lw=1.2 * LW_SCALE, zorder=6)
    ax.axvline(2.5, color="white", lw=1.2 * LW_SCALE, zorder=6)


def _spec_index_panel(ax, gen_matrix: np.ndarray, mice: list,
                      rand_mean_index: float = 0.0,
                      rand_std_ratio: float = 0.0) -> None:
    """Panel C: per-mouse specialisation index, horizontal bars sorted descending.

    rand_mean_index: random-null mean specialisation index (1 - mean_ratio).
    rand_std_ratio:  SD of per-agent ratios in the random null (used as ± band).
    """
    n = len(mice)
    indices = []
    for i in range(n):
        diag = gen_matrix[i, i]
        off = float(np.mean([gen_matrix[i, j] for j in range(n) if j != i]))
        indices.append(1.0 - diag / off)

    order = np.argsort(indices)[::-1]
    s_mice = [mice[i] for i in order]
    s_idx = [indices[i] for i in order]
    colors = [MOUSE_COLORS.get(m, "#888") for m in s_mice]

    y_pos = np.arange(n)
    ax.barh(y_pos, s_idx, color=colors, height=0.6, edgecolor="white")
    ax.axvline(0.0, ls="--", color="gray", lw=0.8 * LW_SCALE, label="No specialisation")

    # Random null reference band (mean ± 1 SD of per-agent ratios)
    if rand_std_ratio > 0:
        lo = rand_mean_index - rand_std_ratio
        hi = rand_mean_index + rand_std_ratio
        ax.axvspan(lo, hi, alpha=0.12, color=RANDOM_COL, zorder=0)
    ax.axvline(rand_mean_index, color=RANDOM_COL, lw=1.0 * LW_SCALE, ls=":",
               label=f"Random null ({rand_mean_index:.2f} ± {rand_std_ratio:.2f})")

    for y, v in zip(y_pos, s_idx):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=FS_ANNOT)
    ax.set_xlim(-0.05, 0.60)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(s_mice, fontsize=FS_TICK)
    ax.set_xlabel("Specialisation index\n(1 − own / cross-mouse fitness)", fontsize=FS_LABEL)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.legend(frameon=False, fontsize=FS_LEGEND, loc=[0.5, 0.9])
    pub_despine(ax)


def _spec_evol_panel(ax, se: dict, sg: dict) -> None:
    """Panel E: specialisation index rises over 150 gens; generalists stay flat.

    Specialists: coloured per-mouse traces + bold population mean ± SD band.
    Generalists: dashed black line (mean |bias| across all 9 per-mouse targets).
    """
    gens     = np.array(se["SAMPLE_GENS"])    # (16,)
    sp_mean  = se["spec_mean"]                # (16, 9)  per-gen, per-mouse mean
    sp_mice  = se["MICE"]

    gen_gens = np.array(sg["SAMPLE_GENS"])    # (16,)
    mean_abs = sg["mean_abs_bias"]            # (16,)  generalist mean |bias|

    pop_mean = sp_mean.mean(axis=1)
    pop_std  = sp_mean.std(axis=1)

    # Per-mouse specialist traces (thin, background)
    for mi, mouse in enumerate(sp_mice):
        ax.plot(gens, sp_mean[:, mi],
                color=MOUSE_COLORS.get(mouse, "#888"),
                lw=1.0 * LW_SCALE, alpha=0.55)

    # Population mean ± SD band
    ax.fill_between(gens, pop_mean - pop_std, pop_mean + pop_std,
                    color=EVOLVED_COL, alpha=0.18, zorder=2)
    ax.plot(gens, pop_mean, color=EVOLVED_COL, lw=2.2 * LW_SCALE, zorder=3,
            label="Specialists (pop. mean)")

    # Generalist flat line
    ax.plot(gen_gens, mean_abs, color="black", lw=1.8 * LW_SCALE, ls="--",
            zorder=4, label="Generalists (mean |bias|)")

    ax.axhline(0, ls=":", color="lightgray", lw=0.8 * LW_SCALE, zorder=0)
    ax.set_xlabel("Generation", fontsize=FS_LABEL)
    ax.set_ylabel("Specialisation index", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc=(0.1, 0.9))
    pub_despine(ax)


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style(font_scale=1.55)
    stats_results = store.stats_results()
    anova_results = stats_results["anova"]
    gen_matrix = store.generalization_matrix()
    gen_meta = store.generalization_meta()
    mice = gen_meta["mice"]

    wd = store.weight_data()
    weight_vectors = np.asarray(wd["weight_vectors"])
    mice_list = wd["mice"]
    sim_matrix = cosine_similarity(weight_vectors)
    ws_data = {"matrix": sim_matrix, "mice": mice_list}

    _a5_path = os.path.join(_PROJECT, "analysis", "A5_random_null.pkl")
    _a5 = _load_pkl(_a5_path)
    rand_mean_index = float(1.0 - float(_a5["mean_ratio"]))
    rand_std_ratio  = float(_a5["std_ratio"])

    se = store.spec_evol()
    sg = store.spec_evol_gen()

    # ── Layout: top row [A | B(spans 2)], bottom row [C | E | D]
    # 3-col grid; A sits above C so their horizontal centres align.
    fig = plt.figure(figsize=FIGSIZE['fig2'])
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.50, wspace=0.65,
        left=0.08, right=0.97, top=0.95, bottom=0.09,
    )

    ax_a = fig.add_subplot(gs[0, :2])      # cosine similarity heatmap
    ax_b = fig.add_subplot(gs[0, -1])     # p-value dotplot (spans 2 cols)
    ax_c = fig.add_subplot(gs[1, 0])      # per-mouse specialisation bars
    ax_e = fig.add_subplot(gs[1, 1])      # specialisation index over generations
    ax_d = fig.add_subplot(gs[1, 2])      # 9x9 generalisation matrix

    _panel_weight_similarity(ax_a, ws_data)
    ax_a.set_title("")

    _pval_panel(ax_b, anova_results)
    ax_b.set_title("")

    _spec_evol_panel(ax_e, se, sg)

    _spec_index_panel(ax_c, gen_matrix, mice,
                      rand_mean_index=rand_mean_index,
                      rand_std_ratio=rand_std_ratio)

    _gen_matrix_panel(ax_d, gen_matrix, mice)

    for ax, lbl in [(ax_a, "A"), (ax_b, "B"), (ax_e, "E"), (ax_c, "C"), (ax_d, "D")]:
        label_panel(ax, lbl)

    out = os.path.join(figures_dir, "fig2_paradox_v2.pdf")
    return save_figure(fig, out)
