"""
paper_v2 Fig 1 -- System and convergence (Claims C1-C3).

8-panel layout  (3 rows):
  Row 1:  A: Maze + trajectories  |  B: Network architecture  |  C: Pop mean fitness
  Row 2:  D: Best individual fitness  |  E: Speed KDE (agent vs mouse)  |  F: Turn rate KDE
  Row 3:  G: EVR forest plot (all 18 features)  |  H: E/I heatmap (spans 2 cols)

Caption (.tex):
  \\textbf{The constrained neuroevolution system.}
  (A)~Hierarchical binary Y-maze with representative mouse trajectories (5 bouts, B5).
  (B)~14-neuron recurrent agent architecture: 6 sensory inputs (squares), 6 interneurons
  (circles), 2 motor outputs (triangles). Excitatory connections in red, inhibitory in blue.
  Dale's Law, sparse connectivity (max 3 in/3 out), and quantised weights \\{0.25, 1.0\\}
  are enforced throughout evolution.
  (C)~Population mean fitness $\\pm$ SD across 6 replicates per mouse over 150 generations.
  Fitness decreases steeply in generations 1--50, with slow improvement continuing to
  generation 100--120 before plateauing.
  (D)~Best individual fitness per mouse. The best agent in each replicate plateaus earlier
  and more cleanly than the population mean, confirming that evolutionary improvement
  concentrates in the leading individuals.
  (E)~Speed distributions for evolved agents (solid) and real mice (dashed), per mouse.
  Agents reproduce the characteristic speed profile of their target mouse without
  explicit selection for speed.
  (F)~Wall contact fraction (thigmotaxis) for evolved agents (filled circles, per mouse)
  and real mouse baselines (dashed lines). Agents exhibit wall-following behaviour that
  was absent from the fitness function.
  (G)~Evolved vs random circuit features: Cohen's $d$ with 95\\% bootstrap CI for all
  18 aggregate structural features. Navy~$p < 0.05$; grey~$p \\geq 0.05$. Evolved
  circuits are sparser, more feedforward, and use stronger connections than random agents.
  (H)~E/I balance at generation 150 for all 8 non-sensory neurons ($54 \\times 8$
  heatmap, agents ordered by mouse; white lines separate mouse groups). Speed motor (Spd)
  converges uniformly to $\\approx$0.9 excitatory across all mice; turn motor (Trn) shows
  greater variability, consistent with closed-loop inhibitory modulation of turning.

Output: figures/fig1_system_v2.pdf
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figure_modules._style import (
    apply_pub_style, pub_despine, save_figure, label_panel,
    FIGSIZE,
    FS_PANEL, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    MICE, MOUSE_COLORS,
    LW_SCALE, MARKER_SCALE,
)
from scripts.sim_setup_figure import draw_maze_panel, draw_network_panel, load_mouse_trajectories
from scripts.figure_modules.main_fig_evr import _forest_panel
from scripts.stats import cohens_d_bootstrap_ci
from config import load_config
from utils.maze import create_maze
from core.agent import Agent


# ── Panel C: population mean fitness ────────────────────────────────────────

def _pop_mean_panel(ax: plt.Axes, mouse_data: dict) -> None:
    """Per-mouse mean population fitness ± SD + grand mean (black)."""
    all_runs = [r for runs in mouse_data.values() for r in runs]
    min_len  = min(len(r["generations"]) for r in all_runs)
    gens     = np.array(all_runs[0]["generations"][:min_len])

    for mouse in MICE:
        if mouse not in mouse_data:
            continue
        mat   = np.stack([r["mean_fitness"][:min_len] for r in mouse_data[mouse]])
        mean  = mat.mean(axis=0)
        std   = mat.std(axis=0)
        color = MOUSE_COLORS[mouse]
        ax.fill_between(gens, mean - std, mean + std, color=color, alpha=0.12, lw=0 * LW_SCALE)
        ax.plot(gens, mean, color=color, lw=1.0 * LW_SCALE, label=mouse)

    grand = np.stack([r["mean_fitness"][:min_len] for r in all_runs]).mean(axis=0)
    ax.plot(gens, grand, color="black", lw=1.8 * LW_SCALE, alpha=0.85, zorder=10, label="Grand mean")

    ax.set_ylabel("Mean population fitness", fontsize=FS_LABEL)
    ax.set_xlabel("Generation", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(gens[0], gens[-1])
    ax.set_ylim(bottom=0)
    # Compact 2-entry legend (per-mouse colours are explained in Panel H footer)
    legend_els = [
        Line2D([0], [0], color="grey", lw=1.0 * LW_SCALE, alpha=0.6,
               label="Per-mouse mean ± SD"),
        Line2D([0], [0], color="black", lw=1.8 * LW_SCALE, label="Grand mean"),
    ]
    ax.legend(handles=legend_els, loc="upper right", frameon=False,
              fontsize=FS_LEGEND, handlelength=1.4,
              labelspacing=0.25, borderpad=0.2)
    pub_despine(ax)


# ── Panel D: best individual fitness ────────────────────────────────────────

def _best_ind_panel(ax: plt.Axes, mouse_data: dict) -> None:
    """Per-mouse best individual fitness ± SD + grand mean (black)."""
    all_runs = [r for runs in mouse_data.values() for r in runs]
    min_len  = min(len(r["generations"]) for r in all_runs)
    gens     = np.array(all_runs[0]["generations"][:min_len])

    for mouse in MICE:
        if mouse not in mouse_data:
            continue
        mat   = np.stack([r["best_fitness"][:min_len] for r in mouse_data[mouse]])
        mean  = mat.mean(axis=0)
        std   = mat.std(axis=0)
        color = MOUSE_COLORS[mouse]
        ax.fill_between(gens, mean - std, mean + std, color=color, alpha=0.12, lw=0 * LW_SCALE)
        ax.plot(gens, mean, color=color, lw=1.0 * LW_SCALE, label=mouse)

    grand = np.stack([r["best_fitness"][:min_len] for r in all_runs]).mean(axis=0)
    ax.plot(gens, grand, color="black", lw=1.8 * LW_SCALE, alpha=0.85, zorder=10, label="Grand mean")

    ax.set_ylabel("Best individual fitness", fontsize=FS_LABEL)
    ax.set_xlabel("Generation", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(gens[0], gens[-1])
    ax.set_ylim(bottom=0)
    pub_despine(ax)


# ── Panel E: speed KDE ───────────────────────────────────────────────────────

def _speed_kde_panel(ax: plt.Axes, emdata: dict) -> None:
    """KDE of agent (solid) and real mouse (dashed) speeds per mouse.
    x-axis starts at 0 — corrects prior truncation at 0.1.
    """
    mice_em = emdata.get("mice", {})
    x_max_global = 0.0

    # First pass: determine global x_max
    for mouse in MICE:
        if mouse not in mice_em:
            continue
        a = np.array(mice_em[mouse]["agent"]["speeds"])
        m = np.array(mice_em[mouse]["mouse"]["speeds"])
        for arr in (a, m):
            arr = arr[arr > 0]
            if len(arr):
                x_max_global = max(x_max_global, arr.max())

    xs = np.linspace(0, x_max_global * 1.05, 500)

    for mouse in MICE:
        if mouse not in mice_em:
            continue
        color = MOUSE_COLORS[mouse]
        entry = mice_em[mouse]

        for speeds, ls, alpha in [
            (np.array(entry["agent"]["speeds"]), "-",  0.85),
            (np.array(entry["mouse"]["speeds"]), "--", 0.50),
        ]:
            speeds = speeds[speeds > 0]
            if len(speeds) < 2:
                continue
            try:
                kde = gaussian_kde(speeds, bw_method="scott")
                ax.plot(xs, kde(xs), color=color, ls=ls, lw=1.0 * LW_SCALE, alpha=alpha)
            except np.linalg.LinAlgError:
                pass

    ax.set_xlabel("|Speed| (units/frame)", fontsize=FS_LABEL)
    ax.set_ylabel("Density", fontsize=FS_LABEL)
    ax.set_xlim(left=0, right=0.5)
    ax.tick_params(labelsize=FS_TICK)
    legend_els = [
        Line2D([0], [0], color="grey", ls="-",  lw=1.5 * LW_SCALE, label="Evolved agent"),
        Line2D([0], [0], color="grey", ls="--", lw=1.5 * LW_SCALE, label="Real mouse"),
    ]
    ax.legend(handles=legend_els, frameon=False, fontsize=FS_LEGEND, loc="upper right")
    pub_despine(ax)


# ── Panel F: turn rate KDE ───────────────────────────────────────────────────

def _turnrate_kde_panel(ax: plt.Axes, emdata: dict) -> None:
    """KDE of agent (solid) and real mouse (dashed) turn rates per mouse."""
    mice_em = emdata.get("mice", {})

    # First pass: determine symmetric x range from 99th-percentile absolute value
    all_turns: list = []
    for mouse in MICE:
        if mouse not in mice_em:
            continue
        entry = mice_em[mouse]
        for arr in (np.array(entry["agent"].get("turns", [])),
                    np.array(entry.get("mouse", {}).get("turns", []))):
            if len(arr):
                all_turns.extend(arr.tolist())

    if all_turns:
        t_bound = float(np.percentile(np.abs(all_turns), 99))
    else:
        t_bound = 1.0
    xs = np.linspace(-t_bound, t_bound, 500)

    for mouse in MICE:
        if mouse not in mice_em:
            continue
        color = MOUSE_COLORS[mouse]
        entry = mice_em[mouse]

        for turns, ls, alpha in [
            (np.array(entry["agent"].get("turns", [])),              "-",  0.85),
            (np.array(entry.get("mouse", {}).get("turns", [])), "--", 0.50),
        ]:
            if len(turns) < 2:
                continue
            try:
                kde = gaussian_kde(turns, bw_method="scott")
                ax.plot(xs, kde(xs), color=color, ls=ls, lw=1.0 * LW_SCALE, alpha=alpha)
            except np.linalg.LinAlgError:
                pass

    ax.set_xlabel("Turn rate (rad/frame)", fontsize=FS_LABEL)
    ax.set_ylabel("Density", fontsize=FS_LABEL)
    ax.set_xlim(-t_bound, t_bound)
    ax.tick_params(labelsize=FS_TICK)
    legend_els = [
        Line2D([0], [0], color="grey", ls="-",  lw=1.5 * LW_SCALE, label="Evolved agent"),
        Line2D([0], [0], color="grey", ls="--", lw=1.5 * LW_SCALE, label="Real mouse"),
    ]
    ax.legend(handles=legend_els, frameon=False, fontsize=FS_LEGEND, loc="upper right")
    pub_despine(ax)


# ── Panel G: EVR forest ───────────────────────────────────────────────────────

FEAT_KEYS = [
    "n_connections", "density", "ei_ratio", "n_exc", "n_inh",
    "si_count", "sm_count", "ii_count", "im_count", "mi_count", "mm_count",
    "si_exc_frac", "ii_exc_frac", "im_exc_frac",
    "inter_in_mean", "inter_out_mean", "w_mean_mag", "frac_strong",
]


def _evr_forest_panel(ax: plt.Axes,
                      circ_list: list,
                      rand_list: list) -> None:
    """Evolved vs random effect sizes — all 18 features, sorted by |d|.

    Parameters
    ----------
    circ_list : list of per-agent feature dicts (54 evolved agents)
    rand_list : list of per-agent feature dicts (random baseline agents)
    """
    from scipy.stats import ttest_ind

    evr_results: dict = {}
    d_cis:       dict = {}

    for feat in FEAT_KEYS:
        ev = np.array([r[feat] for r in circ_list if feat in r], dtype=float)
        rn = np.array([r[feat] for r in rand_list  if feat in r], dtype=float)
        if len(ev) < 2 or len(rn) < 2:
            continue
        _, p  = ttest_ind(ev, rn)
        d, d_lo, d_hi = cohens_d_bootstrap_ci(ev, rn)
        evr_results[feat] = {"cohens_d": d, "p": p}
        d_cis[feat]       = (d, d_lo, d_hi)

    _forest_panel(ax, evr_results, d_cis)
    ax.set_title("")   # panel label handles title


# ── Panel H: E/I heatmap ─────────────────────────────────────────────────────

def _ei_heatmap_panel(ax: plt.Axes,
                      weight_vectors: np.ndarray,
                      mice_list: list) -> None:
    """54×8 E/I heatmap at gen 150 with coloured mouse group x-axis labels."""
    n_agents = len(mice_list)
    W_all    = weight_vectors.reshape(n_agents, 14, 14)
    ei_mat   = np.zeros((n_agents, 8))

    for a in range(n_agents):
        for n, target in enumerate(range(6, 14)):
            inc   = W_all[a, :, target]
            exc   = float(inc[inc > 0].sum())
            inh   = float(abs(inc[inc < 0].sum()))
            total = exc + inh
            ei_mat[a, n] = exc / total if total > 0 else 0.5

    order      = np.argsort([MICE.index(m) if m in MICE else 99 for m in mice_list])
    ei_ordered = ei_mat[order]
    mice_ord   = [mice_list[i] for i in order]

    im = ax.imshow(ei_ordered.T, aspect="auto", cmap="RdBu_r",
                   vmin=0.0, vmax=1.0, interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("E/(E+I)", fontsize=FS_ANNOT)
    cb.ax.tick_params(labelsize=FS_ANNOT)

    for sep in range(6, n_agents, 6):
        ax.axvline(sep - 0.5, color="white", lw=1.0 * LW_SCALE)

    neuron_labels = ["I0", "I1", "I2", "I3", "I4", "I5", "Spd", "Trn"]
    ax.set_yticks(range(8))
    ax.set_yticklabels(neuron_labels, fontsize=FS_TICK)
    ax.set_xlabel("Agent (ordered by mouse)", fontsize=FS_LABEL)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)

    # Coloured mouse group labels below x-axis
    counts = Counter(mice_ord)
    pos    = 0
    for m in MICE:
        if m not in counts:
            continue
        centre = pos + counts[m] / 2 - 0.5
        ax.text(centre, 8.5, m, ha="center", va="bottom",
                fontsize=FS_ANNOT, color=MOUSE_COLORS.get(m, "#333"),
                fontweight="bold")
        pos += counts[m]
    ax.set_xlim(-0.5, n_agents - 0.5)


# ── Assembly ─────────────────────────────────────────────────────────────────

def generate(store, figures_dir: str, manual=False) -> list[str]:
    """Build and save Fig 1 (7-panel system overview)."""
    apply_pub_style(font_scale=1.45)

    mouse_data   = store.mouse_data()
    try:
        emdata = store.emergent_data_permouse()
    except FileNotFoundError as exc:
        print(f"  Warning: emergent data missing ({exc}); Panels E and F will be blank")
        emdata = {}
    circ_list    = store.circuit_features()          # list of 54 per-agent dicts
    rand_base    = store.random_baseline()           # dict with 'all' key -> list
    rand_list    = rand_base["all"]
    wd           = store.weight_data()
    weight_vecs  = np.asarray(wd["weight_vectors"])
    mice_list    = wd["mice"]

    # Panels A + B require the maze and a representative agent
    config = load_config()
    maze   = create_maze(6)
    try:
        trajs = load_mouse_trajectories("B5", n_bouts=5)
    except Exception as exc:
        print(f"  Warning: could not load trajectories ({exc}); Panel A will be blank")
        trajs = []
    np.random.seed(42)
    agent = Agent(config.network)

    # ── Layout ──────────────────────────────────────────────────────────────
    # Single GridSpec with a left margin wide enough to fit Panel G's long
    # y-tick labels — this keeps every panel's plot area vertically aligned
    # (otherwise G's labels push G's plot region right while A/B/C/D/E/F sit
    # flush-left, breaking column alignment).
    fig = plt.figure(figsize=FIGSIZE['fig1'])
    gs  = gridspec.GridSpec(
        3, 3,
        hspace=0.4, wspace=0.3,
        left=0.05, right=0.97, top=0.97, bottom=0.05,
        height_ratios=[1.85, 1.0, 1.5],
        width_ratios=[1.75, 1.55, 1.0],
    )

    ax_a = fig.add_subplot(gs[0, 0])        # maze
    ax_b = fig.add_subplot(gs[0, 1])        # network
    ax_c = fig.add_subplot(gs[0, 2])        # pop mean fitness
    ax_d = fig.add_subplot(gs[1, 0])        # best individual fitness
    ax_e = fig.add_subplot(gs[1, 1])        # speed KDE
    ax_f = fig.add_subplot(gs[1, 2])        # thigmotaxis
    ax_g = fig.add_subplot(gs[2, 0])        # EVR forest
    ax_h = fig.add_subplot(gs[2, 1:])       # E/I heatmap (spans cols 1-2)

    # ── Draw ────────────────────────────────────────────────────────────────
    if trajs:
        draw_maze_panel(ax_a, maze, trajs)
    else:
        ax_a.text(0.5, 0.5, "Trajectories unavailable",
                  ha="center", va="center", transform=ax_a.transAxes)
        ax_a.axis("off")

    draw_network_panel(ax_b, agent, gfx_scale=1.5)
    _speed_kde_panel(ax_c, emdata)
    _best_ind_panel(ax_d, mouse_data)
    _pop_mean_panel(ax_e, mouse_data)
    _turnrate_kde_panel(ax_f, emdata)
    _evr_forest_panel(ax_g, circ_list, rand_list)
    _ei_heatmap_panel(ax_h, weight_vecs, mice_list)

    # ── Panel labels ────────────────────────────────────────────────────────
    for ax, lbl in [
        (ax_a, "A"), (ax_b, "B"), (ax_c, "E"),
        (ax_d, "C"), (ax_e, "D"), (ax_f, "F"),
        (ax_g, "G"), (ax_h, "H"),
    ]:
        label_panel(ax, lbl)

    out = os.path.join(figures_dir, "fig1_system_v2.pdf")
    if not manual:
        return save_figure(fig, out)
    else:
        return [fig]
