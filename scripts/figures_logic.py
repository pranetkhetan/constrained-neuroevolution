"""
figures_logic.py – Plotting functions for the constrained neuroevolution paper.

All heavy lifting lives here; the notebook simply imports and calls.
"""

import os
import pickle
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import seaborn as sns

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, str(Path(_SCRIPT_DIR).parent))

from scripts.figure_modules._style import (
    FS_PANEL, FS_TITLE, FS_ANNOT, FS_SMALL, FS_MICRO, FS_SUPTITLE,
)


class _CpuUnpickler(pickle.Unpickler):
    """Unpickler that maps CuPy arrays to NumPy (for loading GPU-trained data on CPU)."""
    def find_class(self, module, name):
        if module.startswith('cupy'):
            module = module.replace('cupy._core.core', 'numpy').replace('cupy', 'numpy')
        return super().find_class(module, name)


def _load_pickle_cpu(path):
    """Load a pickle file, converting CuPy arrays to NumPy if needed."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        with open(path, 'rb') as f:
            return _CpuUnpickler(f).load()

# 10-run colour palette (perceptually distinct)
RUN_COLORS = [
    "#E64B35",  # red
    "#4DBBD5",  # cyan
    "#00A087",  # teal
    "#3C5488",  # navy
    "#F39B7F",  # salmon
    "#8491B4",  # steel
    "#91D1C2",  # mint
    "#DC9A6C",  # tan
    "#7E6148",  # brown
    "#B09C85",  # taupe
]

METRIC_LABELS = {
    "markov_score":     "Markov Penalty",
    "occupancy_score":  "Occupancy Penalty",
    "tortuosity_score": "Tortuosity Penalty",
    "turn_bias_score":  "Turn Bias Penalty",
}

METRIC_KEYS = ["markov_score", "occupancy_score", "tortuosity_score", "turn_bias_score"]


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_run(results_dir: str) -> dict:
    """
    Load a single evolutionary run.

    Returns
    -------
    dict with keys:
        'generations' : list[int]        – sorted generation numbers
        'mean_fitness': np.ndarray        – (n_gen,)
        'best_fitness': np.ndarray        – (n_gen,)
        'std_fitness' : np.ndarray        – (n_gen,)
        'mean_{metric}': np.ndarray       – (n_gen,) for each metric
        'best_{metric}': np.ndarray       – (n_gen,) for each metric
        'std_{metric}' : np.ndarray       – (n_gen,) for each metric
    """
    gen_dirs = [d for d in os.listdir(results_dir) if d.startswith("gen_")]
    gen_nums = sorted(int(d.split("_")[1]) for d in gen_dirs)

    out = {"generations": gen_nums}

    mean_f, best_f, std_f = [], [], []
    metric_mean = {k: [] for k in METRIC_KEYS}
    metric_best = {k: [] for k in METRIC_KEYS}
    metric_std  = {k: [] for k in METRIC_KEYS}

    for g in gen_nums:
        path = os.path.join(results_dir, f"gen_{g}", "summary.pkl")
        pop = _load_pickle_cpu(path)

        fitnesses = np.array([ind["fitness"] for ind in pop])
        mean_f.append(fitnesses.mean())
        best_f.append(fitnesses.min())       # lower is better
        std_f.append(fitnesses.std())

        for k in METRIC_KEYS:
            vals = np.array([ind[k] for ind in pop])
            metric_mean[k].append(vals.mean())
            # best = value belonging to the best-fitness individual
            best_idx = fitnesses.argmin()
            metric_best[k].append(pop[best_idx][k])
            metric_std[k].append(vals.std())

    out["mean_fitness"] = np.array(mean_f)
    out["best_fitness"] = np.array(best_f)
    out["std_fitness"]  = np.array(std_f)

    for k in METRIC_KEYS:
        out[f"mean_{k}"] = np.array(metric_mean[k])
        out[f"best_{k}"] = np.array(metric_best[k])
        out[f"std_{k}"]  = np.array(metric_std[k])

    return out


def load_all_runs(base_dir: str = ".", prefix: str = "results_") -> list[dict]:
    """Load all result directories matching ``prefix*`` in sorted order."""
    dirs = sorted(
        [d for d in os.listdir(base_dir) if d.startswith(prefix)],
        key=lambda x: int(x.split("_")[1]),
    )
    print(f"Found {len(dirs)} runs: {dirs}")
    runs = []
    for d in dirs:
        print(f"  Loading {d}...", end=" ", flush=True)
        runs.append(load_run(os.path.join(base_dir, d)))
        print(f"{len(runs[-1]['generations'])} generations")
    return runs


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1
# ═══════════════════════════════════════════════════════════════════════

def _plot_panel(ax, runs, value_key, ylabel, annotate=True, show_xlabel=True):
    """
    Plot one panel: 10 run lines + cross-run mean ± 1 σ shading.

    Parameters
    ----------
    value_key : str  – e.g. 'mean_fitness', 'best_markov_score'
    """
    n_runs = len(runs)

    # Determine common generation range (use shortest run)
    min_len = min(len(r["generations"]) for r in runs)
    gens = np.array(runs[0]["generations"][:min_len])

    # Stack all runs
    matrix = np.stack([r[value_key][:min_len] for r in runs])  # (n_runs, n_gen)

    # Cross-run mean and std
    cross_mean = matrix.mean(axis=0)
    cross_std  = matrix.std(axis=0)

    # Shade ±1 σ
    ax.fill_between(
        gens,
        cross_mean - cross_std,
        cross_mean + cross_std,
        color="#888888", alpha=0.15, linewidth=0,
    )

    # Individual run lines
    for i in range(n_runs):
        ax.plot(gens, matrix[i], color=RUN_COLORS[i], alpha=0.65, linewidth=0.7)

    # Staggered annotations (avoid overlapping labels)
    if annotate:
        final_vals = [(matrix[i, -1], i) for i in range(n_runs)]
        final_vals.sort(key=lambda x: x[0])
        # Compute minimum spacing in data coords
        y_range = matrix.max() - matrix.min()
        min_gap = y_range * 0.03
        placed_y = []
        for val, i in final_vals:
            y = val
            for py in placed_y:
                if abs(y - py) < min_gap:
                    y = py + min_gap
            placed_y.append(y)
            ax.annotate(
                str(i + 1),
                xy=(gens[-1], matrix[i, -1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=FS_SMALL,
                color=RUN_COLORS[i],
                fontweight="bold",
                ha="left",
                va="center",
            )

    # Cross-run mean line (on top)
    ax.plot(gens, cross_mean, color="black", linewidth=1.3, alpha=0.85, zorder=10)

    ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel("Generation")
    ax.set_xlim(gens[0], gens[-1])
    ax.set_ylim(bottom=0)


def figure1(runs: list[dict], save_path: str | None = None) -> plt.Figure:
    """
    Figure 1 — Multi-Run Evolutionary Performance and Behavioural Objective Trends.

    Layout
    ------
    Row 0 (tall) : [A] Pop Mean Fitness  |  [B] Best Fitness
    Row 1        : [C] Markov  [D] Occupancy  [E] Tortuosity  [F] Turn Bias
    Row 2        : [G] Markov  [H] Occupancy  [I] Tortuosity  [J] Turn Bias

    Rows 1-2 have "Population Mean" and "Best Individual" as row labels.
    """
    fig = plt.figure(figsize=(7.5, 7.0))

    gs = gridspec.GridSpec(
        3, 4,
        height_ratios=[1.4, 1, 1],
        hspace=0.45,
        wspace=0.38,
        left=0.08,
        right=0.95,
        top=0.94,
        bottom=0.06,
    )

    # ── Panels A & B (span 2 columns each) ──────────────────────────
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])

    _plot_panel(ax_a, runs, "mean_fitness", "Fitness (lower = better)", show_xlabel=False)
    ax_a.set_title("A   Population Mean Fitness", loc="left", fontweight="bold")

    _plot_panel(ax_b, runs, "best_fitness", "Fitness (lower = better)", show_xlabel=False)
    ax_b.set_title("B   Best Individual Fitness", loc="left", fontweight="bold")

    # ── Panels C–F (population mean for each metric) ────────────────
    panel_labels_top = ["C", "D", "E", "F"]
    for col, (key, label) in enumerate(zip(METRIC_KEYS, METRIC_LABELS.values())):
        ax = fig.add_subplot(gs[1, col])
        _plot_panel(ax, runs, f"mean_{key}", label, annotate=False, show_xlabel=False)
        ax.set_title(f"{panel_labels_top[col]}   {label}", loc="left", fontweight="bold")
        if col == 0:
            ax.annotate(
                "Population Mean",
                xy=(-0.45, 0.5),
                xycoords="axes fraction",
                fontsize=FS_TITLE,
                fontweight="bold",
                rotation=90,
                va="center",
                ha="center",
                color="#555555",
            )

    # ── Panels G–J (best individual for each metric) ────────────────
    panel_labels_bot = ["G", "H", "I", "J"]
    for col, (key, label) in enumerate(zip(METRIC_KEYS, METRIC_LABELS.values())):
        ax = fig.add_subplot(gs[2, col])
        _plot_panel(ax, runs, f"best_{key}", label, annotate=False, show_xlabel=True)
        ax.set_title(f"{panel_labels_bot[col]}", loc="left", fontweight="bold")
        if col == 0:
            ax.annotate(
                "Best Individual",
                xy=(-0.45, 0.5),
                xycoords="axes fraction",
                fontsize=FS_TITLE,
                fontweight="bold",
                rotation=90,
                va="center",
                ha="center",
                color="#555555",
            )

    # ── Suptitle ─────────────────────────────────────────────────────
    # fig.suptitle(
    #     "Figure 1 — Evolutionary Performance and Behavioural Objective Trends",
    #     fontsize=10,
    #     fontweight="bold",
    #     y=0.99,
    # )

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved -> {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Emergent Behavioural Convergence
# ═══════════════════════════════════════════════════════════════════════

def load_emergent_data(path: str = "figures/emergent_data.pkl") -> dict:
    """Load re-simulation results from resimulate_best_agents.py."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _kde(data, bw=None, n_points=300, x_range=None):
    """Simple Gaussian KDE for plotting."""
    from scipy.stats import gaussian_kde
    if len(data) < 5:
        return np.array([]), np.array([])
    kde = gaussian_kde(data, bw_method=bw)
    if x_range is None:
        lo, hi = np.percentile(data, [0.5, 99.5])
        margin = (hi - lo) * 0.1
        x_range = (lo - margin, hi + margin)
    x = np.linspace(x_range[0], x_range[1], n_points)
    return x, kde(x)


def _strip_plot(ax, values, mouse_val, ylabel, title, panel_label):
    """Jittered strip plot with mouse baseline."""
    x_jitter = np.random.RandomState(0).uniform(-0.15, 0.15, len(values))
    for i, v in enumerate(values):
        ax.scatter(x_jitter[i], v, color=RUN_COLORS[i], s=35, zorder=5,
                   edgecolors="white", linewidths=0.4)
    # Mean ± std bar
    mean_v = np.mean(values)
    std_v = np.std(values)
    ax.errorbar(0.35, mean_v, yerr=std_v, color="black", capsize=3,
                 marker="s", markersize=4, linewidth=1, zorder=6)

    if mouse_val is not None:
        ax.axhline(mouse_val, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.annotate("Mouse", xy=(0.98, mouse_val), xycoords=("axes fraction", "data"),
                     fontsize=FS_ANNOT, ha="right", va="bottom", color="#444444")

    ax.set_xlim(-0.5, 0.7)
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_title(f"{panel_label}   {title}", loc="left", fontweight="bold")



def figure2(emdata: dict, save_path: str | None = None) -> plt.Figure:
    """
    Figure 2 — Emergent Behavioural Convergence.

    Top row:  [A] Speed distribution KDEs   [B] Turn rate distribution KDEs
    Bot row:  [C] Thigmotaxis strip plot    [D] Median speed   [E] Mean |turn|
              [F] Speed distribution overlap (all agents pooled vs mouse)
    """
    runs = emdata["runs"]
    mouse = emdata.get("mouse")
    n_runs = len(runs)

    fig = plt.figure(figsize=(7.5, 5.5))
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[1.3, 1],
        hspace=0.42,
        wspace=0.40,
        left=0.08,
        right=0.96,
        top=0.92,
        bottom=0.08,
    )

    # ── Panel A: Speed distributions ─────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0:2])

    # Determine common x-range
    all_agent_speeds = np.concatenate([r["speeds"] for r in runs])
    speed_range = (0, np.percentile(np.abs(all_agent_speeds), 99.5))

    if mouse and mouse.get("speeds") is not None and len(mouse["speeds"]) > 0:
        mouse_abs_speeds = np.abs(mouse["speeds"])
        speed_range = (0, max(speed_range[1], np.percentile(mouse_abs_speeds, 99.5)))

    for i, r in enumerate(runs):
        x, y = _kde(np.abs(r["speeds"]), n_points=200, x_range=speed_range)
        ax_a.plot(x, y, color=RUN_COLORS[i], alpha=0.55, linewidth=0.8)

    if mouse and mouse.get("speeds") is not None and len(mouse["speeds"]) > 0:
        x, y = _kde(mouse_abs_speeds, n_points=200, x_range=speed_range)
        ax_a.plot(x, y, color="black", linewidth=2.0, linestyle="--", label="Mouse", zorder=10)
        ax_a.legend(loc="upper right", frameon=False)

    ax_a.set_xlabel("|Speed| (units/frame)")
    ax_a.set_ylabel("Density")
    ax_a.set_title("A   Speed Distribution", loc="left", fontweight="bold")
    ax_a.set_xlim(speed_range)
    ax_a.set_ylim(bottom=0)

    # ── Panel B: Turn rate distributions ─────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])

    all_agent_turns = np.concatenate([r["turns"] for r in runs])
    turn_range = (-np.percentile(np.abs(all_agent_turns), 99), np.percentile(np.abs(all_agent_turns), 99))

    if mouse and mouse.get("turns") is not None and len(mouse["turns"]) > 0:
        turn_range = (
            min(turn_range[0], -np.percentile(np.abs(mouse["turns"]), 99)),
            max(turn_range[1], np.percentile(np.abs(mouse["turns"]), 99)),
        )

    for i, r in enumerate(runs):
        x, y = _kde(r["turns"], n_points=200, x_range=turn_range)
        ax_b.plot(x, y, color=RUN_COLORS[i], alpha=0.55, linewidth=0.8)

    if mouse and mouse.get("turns") is not None and len(mouse["turns"]) > 0:
        x, y = _kde(mouse["turns"], n_points=200, x_range=turn_range)
        ax_b.plot(x, y, color="black", linewidth=2.0, linestyle="--", label="Mouse", zorder=10)
        ax_b.legend(loc="upper right", frameon=False)

    ax_b.set_xlabel("Turn Rate (rad/frame)")
    ax_b.set_ylabel("Density")
    ax_b.set_ylim(bottom=0)
    ax_b.set_title("B   Turn Rate Distribution", loc="left", fontweight="bold")

    # ── Panel C: Thigmotaxis ─────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    thigmo_vals = [r["thigmotaxis"] for r in runs]
    mouse_thigmo = mouse.get("thigmotaxis") if mouse else None
    _strip_plot(ax_c, thigmo_vals, mouse_thigmo, "Wall Contact Fraction",
                "Thigmotaxis", "C")

    # ── Panel D: Median speed ────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    med_speed_vals = [r["median_speed"] for r in runs]
    mouse_med_speed = float(np.median(np.abs(mouse["speeds"]))) if mouse and len(mouse.get("speeds", [])) > 0 else None
    _strip_plot(ax_d, med_speed_vals, mouse_med_speed, "Median |Speed|",
                "Median Speed", "D")

    # ── Panel E: Mean |turn rate| ────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    mean_turn_vals = [r["mean_abs_turn"] for r in runs]
    mouse_mean_turn = float(np.mean(np.abs(mouse["turns"]))) if mouse and len(mouse.get("turns", [])) > 0 else None
    _strip_plot(ax_e, mean_turn_vals, mouse_mean_turn, "Mean |Turn Rate|",
                "Mean Turn Rate", "E")

    # ── Suptitle ─────────────────────────────────────────────────────
    # fig.suptitle(
    #     "Figure 2 — Emergent Behavioural Convergence (Not in Fitness Function)",
    #     fontsize=10,
    #     fontweight="bold",
    #     y=0.98,
    # )

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved -> {save_path}")

    return fig


# ── 9-mouse colour palette ─────────────────────────────────────────
MOUSE_COLORS = {
    "B5": "#E64B35", "B6": "#4DBBD5", "B7": "#00A087",
    "D3": "#3C5488", "D4": "#F39B7F", "D5": "#8491B4",
    "D7": "#91D1C2", "D8": "#DC9A6C", "D9": "#7E6148",
}


def figure3(circuit_data, random_baseline, anova_results=None,
            generalization_matrix=None, mice=None,
            save_path=None):
    """
    Figure 3 — Circuit Properties and Per-Mouse Comparison.

    Parameters
    ----------
    circuit_data : list of dict
        Output from analyze_circuits.py (each dict has 'mouse', 'rep', features).
    random_baseline : dict
        {'mean': {feature: value}, 'all': [list of dicts]}.
    anova_results : dict or None
        Output from stats.anova_with_fdr(). If None, no significance stars.
    generalization_matrix : np.ndarray or None
        (9, 9) cross-mouse fitness matrix. If None, Panel E is skipped.
    mice : list of str or None
        Ordered mouse IDs for the generalization matrix.
    save_path : str or None
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from stats import bootstrap_ci

    fig = plt.figure(figsize=(7.5, 7.0))
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[1, 1],
        hspace=0.45,
        wspace=0.42,
        left=0.08,
        right=0.96,
        top=0.92,
        bottom=0.08,
    )

    # Key features for bar charts
    key_features = ['density', 'ii_count', 'ei_ratio', 'sm_count']
    key_labels = ['Connection\nDensity', 'I-I\nRecurrence', 'E/I\nRatio', 'S-M\nShortcuts']

    pathway_features = ['si_count', 'ii_count', 'im_count', 'mi_count', 'sm_count', 'mm_count']
    pathway_labels = ['S->I', 'I->I', 'I->M', 'M->I', 'S->M', 'M->M']

    ei_features = ['si_exc_frac', 'ii_exc_frac', 'im_exc_frac']
    ei_labels = ['S->I', 'I->I', 'I->M']

    # ── Panel A: Key features bar chart with CIs ──────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    x = np.arange(len(key_features))
    means = []
    ci_lo = []
    ci_hi = []
    for feat in key_features:
        vals = [r[feat] for r in circuit_data]
        m, lo, hi = bootstrap_ci(vals)
        means.append(m)
        ci_lo.append(m - lo)
        ci_hi.append(hi - m)

    ax_a.bar(x, means, color="#4DBBD5", edgecolor="white", linewidth=0.5, zorder=3)
    ax_a.errorbar(x, means, yerr=[ci_lo, ci_hi], fmt='none', color='black',
                  capsize=3, linewidth=0.8, zorder=4)

    # Random baseline dashed lines
    for i, feat in enumerate(key_features):
        ax_a.plot([i - 0.35, i + 0.35], [random_baseline['mean'][feat]] * 2,
                  color='red', linestyle='--', linewidth=0.8, zorder=5)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(key_labels)
    ax_a.set_ylabel("Value")
    ax_a.set_title("A   Key Circuit Features", loc="left", fontweight="bold")

    # ── Panel B: Pathway composition stacked bar ──────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    pathway_means = [np.mean([r[f] for r in circuit_data]) for f in pathway_features]
    pathway_colors = ["#3C5488", "#E64B35", "#00A087", "#F39B7F", "#8491B4", "#DC9A6C"]

    bottom = 0
    for val, label, color in zip(pathway_means, pathway_labels, pathway_colors):
        ax_b.bar(0, val, bottom=bottom, color=color, edgecolor='white',
                 linewidth=0.5, label=label, width=0.6)
        bottom += val

    ax_b.set_xticks([])
    ax_b.set_ylabel("Connection Count")
    ax_b.legend(loc="upper right", frameon=False)
    ax_b.set_title("B   Pathway Composition", loc="left", fontweight="bold")

    # ── Panel C: E/I balance by pathway ───────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])

    x_ei = np.arange(len(ei_features))
    ei_means = []
    ei_ci_lo = []
    ei_ci_hi = []
    for feat in ei_features:
        vals = [r[feat] for r in circuit_data]
        m, lo, hi = bootstrap_ci(vals)
        ei_means.append(m)
        ei_ci_lo.append(m - lo)
        ei_ci_hi.append(hi - m)

    ax_c.bar(x_ei, ei_means, color="#00A087", edgecolor="white", linewidth=0.5, zorder=3)
    ax_c.errorbar(x_ei, ei_means, yerr=[ei_ci_lo, ei_ci_hi], fmt='none',
                  color='black', capsize=3, linewidth=0.8, zorder=4)
    ax_c.axhline(0.5, color='gray', linestyle=':', linewidth=0.6)
    ax_c.set_xticks(x_ei)
    ax_c.set_xticklabels(ei_labels)
    ax_c.set_ylabel("Excitatory Fraction")
    ax_c.set_ylim(0, 1)
    ax_c.set_title("C   E/I Balance by Pathway", loc="left", fontweight="bold")

    # ── Panel D: Key features grouped by mouse ────────────────────────
    ax_d = fig.add_subplot(gs[1, 0:2])

    if mice is None:
        mice = sorted(set(r['mouse'] for r in circuit_data if r.get('mouse') != 'aggregate'))

    n_mice = len(mice)
    n_feats = len(key_features)
    bar_width = 0.8 / n_feats

    for j, feat in enumerate(key_features):
        for i, m in enumerate(mice):
            vals = [r[feat] for r in circuit_data if r.get('mouse') == m]
            if not vals:
                continue
            x_pos = i + j * bar_width - 0.4 + bar_width / 2
            mean_val = np.mean(vals)
            color = MOUSE_COLORS.get(m, '#888888')
            # Individual points
            ax_d.scatter([x_pos] * len(vals), vals, s=8, color=color, alpha=0.5, zorder=3)
            # Mean marker
            ax_d.scatter(x_pos, mean_val, s=25, color=color, edgecolor='black',
                         linewidth=0.5, zorder=4, marker='D')

    ax_d.set_xticks(np.arange(n_mice))
    ax_d.set_xticklabels(mice)
    ax_d.set_ylabel("Feature Value")
    ax_d.set_title("D   Circuit Features by Mouse", loc="left", fontweight="bold")

    # Add ANOVA significance
    if anova_results:
        sig_feats = [f for f in key_features if anova_results.get(f, {}).get('significant')]
        if sig_feats:
            ax_d.annotate(f"ANOVA sig: {', '.join(sig_feats)}",
                          xy=(0.98, 0.98), xycoords='axes fraction',
                          fontsize=FS_ANNOT, ha='right', va='top', color='red')

    # ── Panel E: Generalization heatmap ───────────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])

    if generalization_matrix is not None and mice is not None:
        im = ax_e.imshow(generalization_matrix, cmap='RdYlGn_r', aspect='auto')
        ax_e.set_xticks(np.arange(len(mice)))
        ax_e.set_yticks(np.arange(len(mice)))
        ax_e.set_xticklabels(mice, rotation=45)
        ax_e.set_yticklabels(mice)
        ax_e.set_xlabel("Tested On")
        ax_e.set_ylabel("Trained On")
        plt.colorbar(im, ax=ax_e, shrink=0.8, label="Fitness (lower=better)")
    else:
        ax_e.text(0.5, 0.5, "Generalization matrix\nnot yet computed",
                  ha='center', va='center', fontsize=FS_ANNOT, color='gray',
                  transform=ax_e.transAxes)
        ax_e.set_xticks([])
        ax_e.set_yticks([])

    ax_e.set_title("E   Cross-Mouse Generalization", loc="left", fontweight="bold")

    # ── Suptitle ──────────────────────────────────────────────────────
    # fig.suptitle(
    #     "Figure 3 — Circuit Properties and Per-Mouse Comparison",
    #     fontsize=10,
    #     fontweight="bold",
    #     y=0.98,
    # )

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved -> {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# PER-MOUSE DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def discover_mice(base_dir: str = "results") -> dict:
    """
    Auto-discover per-mouse run directories.

    Returns dict: {mouse_id: [sorted list of (rep_num, full_path) tuples]}
    """
    import re
    pattern = re.compile(r'^results_([A-Z]\d+)_r(\d+)$')
    mice = {}
    for name in sorted(os.listdir(base_dir)):
        m = pattern.match(name)
        if m:
            mouse_id, rep = m.group(1), int(m.group(2))
            full_path = os.path.join(base_dir, name)
            if os.path.isdir(full_path):
                mice.setdefault(mouse_id, []).append((rep, full_path))
    for mouse_id in mice:
        mice[mouse_id].sort()
    return mice


def load_all_runs_permouse(base_dir: str = "data/agents") -> dict:
    """
    Load all per-mouse runs.

    Returns dict: {mouse_id: [list of load_run() result dicts]}
    """
    mice = discover_mice(base_dir)
    result = {}
    for mouse_id in sorted(mice):
        reps = mice[mouse_id]
        result[mouse_id] = []
        for rep_num, full_path in reps:
            try:
                run = load_run(full_path)
                result[mouse_id].append(run)
                n_gen = len(run['generations'])
                print(f"  {mouse_id}/r{rep_num}: {n_gen} generations")
            except Exception as e:
                print(f"  {mouse_id}/r{rep_num}: FAILED - {e}")
    total = sum(len(v) for v in result.values())
    print(f"Loaded {total} runs across {len(result)} mice")
    return result


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1 — PER-MOUSE VERSION
# ═══════════════════════════════════════════════════════════════════════

def _plot_panel_permouse(ax, mouse_data, value_key, ylabel,
                         show_xlabel=True, show_legend=False,
                         kde_inset_pos=None):
    """
    Plot one panel with per-mouse mean ± std.

    Parameters
    ----------
    mouse_data : dict of mouse_id -> list[dict] (from load_all_runs_permouse)
    value_key : str — e.g. 'mean_fitness', 'best_markov_score'
    kde_inset_pos : list [x0, y0, w, h] in axes-fraction coordinates, or None.
                    When given, draws a per-mouse KDE inset of final-generation
                    values (9 color-coded KDEs, no axes/ticks/labels).
    """
    # Find common generation range
    all_runs = [r for runs in mouse_data.values() for r in runs]
    min_len = min(len(r["generations"]) for r in all_runs)
    gens = np.array(all_runs[0]["generations"][:min_len])

    for mouse_id in sorted(mouse_data):
        runs = mouse_data[mouse_id]
        if not runs:
            continue
        color = MOUSE_COLORS.get(mouse_id, '#888888')
        matrix = np.stack([r[value_key][:min_len] for r in runs])
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)

        ax.fill_between(gens, mean - std, mean + std,
                         color=color, alpha=0.12, linewidth=0)
        ax.plot(gens, mean, color=color, linewidth=1.0, label=mouse_id)

    # Grand mean across all runs
    all_matrix = np.stack([r[value_key][:min_len] for r in all_runs])
    grand_mean = all_matrix.mean(axis=0)
    ax.plot(gens, grand_mean, color='black', linewidth=1.5, alpha=0.8,
            zorder=10, label='Grand mean')

    ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel("Generation")
    ax.set_xlim(gens[0], gens[-1])
    ax.set_ylim(bottom=0)
    if show_legend:
        ax.legend(ncol=2, loc='upper right', frameon=False)

    if kde_inset_pos is not None:
        ax_ins = ax.inset_axes(kde_inset_pos)

        # Collect per-mouse final-generation values to set a shared x range
        all_final = [r[value_key][min_len - 1] for r in all_runs]
        v_min, v_max = min(all_final), max(all_final)
        pad = max((v_max - v_min) * 0.3, 1e-6)
        xs = np.linspace(v_min - pad, v_max + pad, 300)

        for mouse_id in sorted(mouse_data):
            runs = mouse_data[mouse_id]
            if not runs:
                continue
            mouse_finals = np.array([r[value_key][min_len - 1] for r in runs])
            color = MOUSE_COLORS.get(mouse_id, '#888888')
            try:
                kde = gaussian_kde(mouse_finals)
                ys = kde(xs)
                ax_ins.fill_between(xs, ys, alpha=0.35, color=color, linewidth=0)
                ax_ins.plot(xs, ys, color=color, linewidth=0.7, alpha=0.85)
            except np.linalg.LinAlgError:
                pass  # all runs identical — skip

        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        for sp in ax_ins.spines.values():
            sp.set_visible(False)
        ax_ins.patch.set_alpha(0.0)


def figure1_permouse(mouse_data: dict, save_path: str | None = None) -> plt.Figure:
    """
    Figure 1 — Per-Mouse Evolutionary Performance.

    Same 3-row layout as figure1(), but lines grouped by mouse.
    """
    fig = plt.figure(figsize=(7.5, 7.0))
    gs = gridspec.GridSpec(
        3, 4,
        # height_ratios=[1.4, 1, 1],
        hspace=0.4, wspace=0.5,
        left=0.05, right=0.95, top=0.95, bottom=0.05,
    )

    # Row 0: Fitness
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    _plot_panel_permouse(ax_a, mouse_data, "mean_fitness",
                         "Fitness (lower = better)", show_xlabel=False,
                         show_legend=False)
    ax_a.set_title("A. Population Mean Fitness", loc="left", fontweight="bold")
    ax_a.set_ylim(0, 4.0)
    _plot_panel_permouse(ax_b, mouse_data, "best_fitness",
                         "Fitness (lower = better)", show_xlabel=False, show_legend=True)
    ax_b.set_ylim(0, 4.0)
    ax_b.set_title("B. Best Individual Fitness", loc="left", fontweight="bold")
    sns.despine(ax=ax_a)
    sns.despine(ax=ax_b)

    # Row 1: Pop mean metrics  — KDE inset at bottom-left
    _KDE_BL = [0.02, 0.02, 0.5, 0.2]   # bottom-left: short, wide-ish
    up_lim = [1.5, 1.5, 0.3, 0.25]
    labels_top = ["C", "D", "E", "F"]
    for col, (key, label) in enumerate(zip(METRIC_KEYS, METRIC_LABELS.values())):
        ax = fig.add_subplot(gs[1, col])
        _plot_panel_permouse(ax, mouse_data, f"mean_{key}", label,
                             show_xlabel=False, kde_inset_pos=_KDE_BL)
        ax.set_title(f"{labels_top[col]}. {label}", loc="left",
                     fontweight="bold", fontsize=FS_TITLE)
        if col == 0:
            ax.annotate("Population Mean", xy=(-0.5, 0.5),
                         xycoords="axes fraction", fontsize=FS_TITLE,
                         fontweight="bold", rotation=90, va="center",
                         ha="center", color="#555555")
        ax.set_ylim(0, up_lim[col])
        sns.despine(ax=ax)


    # Row 2: Best individual metrics — KDE inset at top-right
    _KDE_TR = [0.02, 0.75, 0.5, 0.2]   # top-right: short, wide-ish
    # labels_bot = ["G", "H", "I", "J"]
    for col, (key, label) in enumerate(zip(METRIC_KEYS, METRIC_LABELS.values())):
        ax = fig.add_subplot(gs[2, col])
        _plot_panel_permouse(ax, mouse_data, f"best_{key}", label,
                             show_xlabel=True, kde_inset_pos=_KDE_TR)
        # ax.set_title(f"{labels_bot[col]}", loc="left",
        #              fontweight="bold", fontsize=FS_TITLE)
        ax.set_ylim(0, up_lim[col])
        if col == 0:
            ax.annotate("Best Individual", xy=(-0.5, 0.5),
                         xycoords="axes fraction", fontsize=FS_TITLE,
                         fontweight="bold", rotation=90, va="center",
                         ha="center", color="#555555")
        sns.despine(ax=ax)

    # fig.suptitle(
    #     "Figure 1 — Per-Mouse Evolutionary Performance",
    #     fontsize=10, fontweight="bold", y=0.99,
    # )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        # save svg as well
        fig.savefig(save_path.replace(".png", ".svg"), dpi=300)
        print(f"Saved -> {save_path}")
    return fig


def figure1_supplement(mouse_data: dict,
                       save_path_prefix: str | None = None) -> list:
    """
    Supplementary figures: one figure1-style plot per mouse (6 rep lines each).

    Returns list of Figure objects.
    """
    figs = []
    for mouse_id in sorted(mouse_data):
        runs = mouse_data[mouse_id]
        if not runs:
            continue
        fig = figure1(runs, save_path=None)
        fig.suptitle(
            f"Figure S1 — Mouse {mouse_id} ({len(runs)} replicates)",
            fontsize=FS_SUPTITLE, fontweight="bold", y=0.99,
        )
        if save_path_prefix:
            path = f"{save_path_prefix}_{mouse_id}.png"
            fig.savefig(path, dpi=300)
            print(f"Saved -> {path}")
        figs.append(fig)
    return figs


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — PER-MOUSE VERSION
# ═══════════════════════════════════════════════════════════════════════

def load_emergent_data_permouse(path: str = "figures/emergent_data_permouse.pkl") -> dict:
    """Load per-mouse re-simulation results."""
    with open(path, "rb") as f:
        return pickle.load(f)


def figure2_permouse(emdata: dict, save_path: str | None = None) -> plt.Figure:
    """
    Figure 2 — Emergent Behavioural Convergence (per-mouse).

    Layout: same 2-row, 3-column as figure2().
    emdata['mice'] = {mouse_id: {'agent': {...}, 'mouse': {...}}}
    """
    mice_data = emdata["mice"]
    mice_ids = sorted(mice_data.keys())

    fig = plt.figure(figsize=(7.5, 5.5))
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[1.3, 1],
        hspace=0.42, wspace=0.40,
        left=0.08, right=0.96, top=0.92, bottom=0.08,
    )

    # ── Panel A: Speed distributions ─────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0:2])

    # Compute common x-range from all mice
    all_speeds = []
    for m in mice_ids:
        agent_data = mice_data[m].get('agent', {})
        if 'speeds' in agent_data and len(agent_data['speeds']) > 0:
            all_speeds.extend(np.abs(agent_data['speeds']))
    speed_range = (0, np.percentile(all_speeds, 99.5)) if all_speeds else (0, 1)

    for m in mice_ids:
        color = MOUSE_COLORS.get(m, '#888888')
        agent_data = mice_data[m].get('agent', {})
        if 'speeds' in agent_data and len(agent_data['speeds']) > 0:
            x, y = _kde(np.abs(agent_data['speeds']), n_points=200,
                        x_range=speed_range)
            ax_a.plot(x, y, color=color, alpha=0.7, linewidth=1.5, label=m)

        mouse_dist = mice_data[m].get('mouse')
        if mouse_dist and 'speeds' in mouse_dist and len(mouse_dist['speeds']) > 0:
            x, y = _kde(np.abs(mouse_dist['speeds']), n_points=200,
                        x_range=speed_range)
            ax_a.plot(x, y, color=color, linewidth=0.9, linestyle='--', alpha=0.5)

    ax_a.set_xlabel("|Speed| (units/frame)")
    ax_a.set_ylabel("Density")
    ax_a.set_title("A. Speed Distribution", loc="left", fontweight="bold")
    ax_a.set_xlim(speed_range)
    ax_a.set_ylim(0,50)
    # Build a two-part legend:
    #   Part 1 — one coloured entry per mouse (solid line = agent)
    #   Part 2 — encoding legend (solid = evolved agent, dashed = real mouse)
    handles, labels = ax_a.get_legend_handles_labels()
    encoding_handles = [
        Line2D([0], [0], color='black', linewidth=1.5, linestyle='-',  label='Evolved agent'),
        Line2D([0], [0], color='black', linewidth=0.9, linestyle='--', alpha=0.5, label='Real mouse'),
    ]
    ax_a.legend(
        handles=handles + encoding_handles,
        labels=labels + ['Evolved agent', 'Real mouse'],
        ncol=3, loc='upper right', frameon=False,
        fontsize=FS_ANNOT,
    )
    sns.despine(ax=ax_a)

    # ── Panel B: Turn rate distributions ─────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])

    all_turns = []
    for m in mice_ids:
        agent_data = mice_data[m].get('agent', {})
        if 'turns' in agent_data and len(agent_data['turns']) > 0:
            all_turns.extend(agent_data['turns'])
    if all_turns:
        t99 = np.percentile(np.abs(all_turns), 99)
        turn_range = (-t99, t99)
    else:
        turn_range = (-1, 1)

    for m in mice_ids:
        color = MOUSE_COLORS.get(m, '#888888')
        agent_data = mice_data[m].get('agent', {})
        if 'turns' in agent_data and len(agent_data['turns']) > 0:
            x, y = _kde(agent_data['turns'], n_points=200, x_range=turn_range)
            ax_b.plot(x, y, color=color, alpha=0.7, linewidth=1.5)

        mouse_dist = mice_data[m].get('mouse')
        if mouse_dist and 'turns' in mouse_dist and len(mouse_dist['turns']) > 0:
            x, y = _kde(mouse_dist['turns'], n_points=200, x_range=turn_range)
            ax_b.plot(x, y, color=color, linewidth=0.9, linestyle='--', alpha=0.5)

    ax_b.set_xlabel("Turn Rate (rad/frame)")
    ax_b.set_ylabel("Density")
    ax_b.set_ylim(bottom=0)
    ax_b.set_title("B. Turn Rate Distribution", loc="left", fontweight="bold")
    # Encoding legend: solid = evolved agent, dashed = real mouse
    ax_b.legend(
        handles=[
            Line2D([0], [0], color='black', linewidth=1.5, linestyle='-',  label='Evolved agent'),
            Line2D([0], [0], color='black', linewidth=0.9, linestyle='--', alpha=0.5, label='Real mouse'),
        ],
        loc='upper left', frameon=False, fontsize=FS_ANNOT,
    )
    sns.despine(ax=ax_b)

    # ── Panels C-E: Strip plots ──────────────────────────────────────
    strip_configs = [
        ('thigmotaxis', 'Wall Contact Fraction', 'Thigmotaxis', 'C'),
        ('median_speed', 'Median |Speed|', 'Median Speed', 'D'),
        ('mean_abs_turn', 'Mean |Turn Rate|', 'Mean Turn Rate', 'E'),
    ]

    up_lims = [0.8, 0.2, 0.3]
    for col, (key, ylabel, title, panel_label) in enumerate(strip_configs):
        ax = fig.add_subplot(gs[1, col])
        rng = np.random.RandomState(0)

        for i, m in enumerate(mice_ids):
            color = MOUSE_COLORS.get(m, '#888888')
            agent_data = mice_data[m].get('agent', {})
            val = agent_data.get(key)
            if val is not None:
                x_jitter = rng.uniform(-0.15, 0.15)
                ax.scatter(i + x_jitter, val, color=color, s=35, zorder=5,
                           edgecolors="white", linewidths=0.4)

            # Mouse baseline as horizontal tick
            mouse_dist = mice_data[m].get('mouse')
            if mouse_dist and key in mouse_dist and mouse_dist[key] is not None:
                ax.plot([i - 0.25, i + 0.25], [mouse_dist[key]] * 2,
                        color=color, linestyle='-', linewidth=1, alpha=1.0)

        # Grand mean
        vals = [mice_data[m]['agent'].get(key) for m in mice_ids
                if mice_data[m].get('agent', {}).get(key) is not None]
        if vals:
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            n = len(mice_ids)
            ax.errorbar(n + 0.5, mean_v, yerr=std_v, color="black",
                        capsize=3, marker="s", markersize=4, linewidth=1,
                        zorder=6)

        ax.set_xticks(list(range(len(mice_ids))) + [len(mice_ids) + 0.5])
        ax.set_xticklabels(mice_ids + ['Avg'], rotation=45)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0, top=up_lims[col])
        ax.set_title(f"{panel_label}. {title}", loc="left",
                     fontweight="bold", fontsize=FS_TITLE)
        # Add encoding legend on first strip panel only
        if col == 0:
            ax.legend(
                handles=[
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                           markersize=5, label='Evolved agent'),
                    Line2D([0], [0], color='black', linewidth=1.2, linestyle='-',
                           label='Real mouse baseline'),
                ],
                loc='lower right', frameon=False, fontsize=FS_ANNOT,
            )
        sns.despine(ax=ax)
    # fig.suptitle(
    #     "Figure 2 — Emergent Behavioural Convergence (Per Mouse)",
    #     fontsize=10, fontweight="bold", y=0.98,
    # )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        fig.savefig(save_path.replace(".png", ".svg"), dpi=300)
        print(f"Saved -> {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3 — NETWORK AND E/I ANALYSIS (NEW)
# ═══════════════════════════════════════════════════════════════════════

# Node labels for publication
_NODE_LABELS = {
    'S0': 'F', 'S1': 'L', 'S2': 'R',
    'S3': 'PSpd', 'S4': 'PTrn', 'S5': 'N',
    'M0': 'Speed', 'M1': 'Turn',
}

# Layout constants (scaled down from visualize.py)
_X_SPREAD = 2.5
_Y_SPREAD_S = 2.0
_Y_SPREAD_I = 2.5
_Y_SPREAD_M = 1.0


def _panel_network_graph(ax, agent):
    """
    Draw publication-quality network graph of a single agent.

    Nodes: squares=sensory, circles=inter, triangles=motor.
    Colors: red=excitatory, blue=inhibitory.
    Edges: red=excitatory, blue=inhibitory, width~|weight|.
    """
    import networkx as nx

    G = agent.to_networkx()

    # Deterministic layout (same as visualize.py)
    S = sorted([n for n in G.nodes if G.nodes[n]['type'] == 'sensory'],
               key=lambda x: G.nodes[x]['idx'])
    I = sorted([n for n in G.nodes if G.nodes[n]['type'] == 'inter'],
               key=lambda x: G.nodes[x]['idx'])
    M = sorted([n for n in G.nodes if G.nodes[n]['type'] == 'motor'],
               key=lambda x: G.nodes[x]['idx'])

    pos = {}
    ys = np.linspace(_Y_SPREAD_S/2, -_Y_SPREAD_S/2, len(S)) if len(S) > 1 else [0.0]
    for i, n in enumerate(S):
        pos[n] = np.array([-_X_SPREAD, ys[i]])
    ys = np.linspace(_Y_SPREAD_I/2, -_Y_SPREAD_I/2, len(I)) if len(I) > 1 else [0.0]
    for i, n in enumerate(I):
        pos[n] = np.array([0.0, ys[i]])
    ys = np.linspace(_Y_SPREAD_M/2, -_Y_SPREAD_M/2, len(M)) if len(M) > 1 else [0.0]
    for i, n in enumerate(M):
        pos[n] = np.array([_X_SPREAD, ys[i]])

    edges = G.edges(data=True)
    exc_edges = [(u, v) for u, v, d in edges if d['weight'] > 0]
    inh_edges = [(u, v) for u, v, d in edges if d['weight'] < 0]
    exc_w = [abs(d['weight']) for u, v, d in edges if d['weight'] > 0]
    inh_w = [abs(d['weight']) for u, v, d in edges if d['weight'] < 0]

    def scale_w(ws):
        return [0.3 + 1.5 * w for w in ws]

    if exc_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=exc_edges,
                               width=scale_w(exc_w), edge_color='#ff4444',
                               alpha=0.35, arrowstyle='-|>', arrowsize=8,
                               connectionstyle='arc3,rad=0.25')
    if inh_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=inh_edges,
                               width=scale_w(inh_w), edge_color='#4444ff',
                               alpha=0.35, arrowstyle='-[', arrowsize=8,
                               connectionstyle='arc3,rad=0.25')

    def get_color(nlist):
        return ['#ff8888' if G.nodes[n].get('node_type', 1) == 1
                else '#8888ff' for n in nlist]

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=S, node_color=get_color(S),
                           node_shape='s', node_size=200,
                           edgecolors='#333333', linewidths=1.0)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=I, node_color=get_color(I),
                           node_shape='o', node_size=170,
                           edgecolors='#333333', linewidths=1.0)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=M, node_color=get_color(M),
                           node_shape='^', node_size=200,
                           edgecolors='#333333', linewidths=1.0)

    # Labels
    for n in G.nodes:
        label = _NODE_LABELS.get(n, n)
        ntype = "E" if G.nodes[n].get('node_type', 1) == 1 else "I"
        ax.text(pos[n][0], pos[n][1] + 0.28, f"{label}",
                ha='center', va='bottom', fontsize=FS_MICRO, color='#333333',
                fontweight='bold')

    # Column headers
    font_props = {'color': '#333333', 'weight': 'bold', 'size': FS_ANNOT}
    ax.text(-_X_SPREAD, 1.6, "SENSORY", ha='center', **font_props)
    ax.text(0.0, 1.6, "INTER", ha='center', **font_props)
    ax.text(_X_SPREAD, 1.6, "MOTOR", ha='center', **font_props)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-2.0, 2.0)
    ax.axis('off')


def _panel_ei_timeseries(ax, ei_data, ylabel="E/I Ratio",
                         show_xlabel=True):
    """
    Plot E/I ratio timeseries across generations, grouped by mouse.

    Parameters
    ----------
    ei_data : dict {mouse_id: {rep: {'generations': list, 'ei_ratios': list}}}
    """
    all_curves = []
    gens = None
    for mouse_id in sorted(ei_data):
        color = MOUSE_COLORS.get(mouse_id, '#888888')
        reps = ei_data[mouse_id]
        if not reps:
            continue

        # Stack reps into matrix
        rep_keys = sorted(reps.keys())
        min_len = min(len(reps[r]['generations']) for r in rep_keys)
        if min_len == 0:
            continue

        gens = np.array(reps[rep_keys[0]]['generations'][:min_len])
        matrix = np.stack([
            np.array(reps[r]['ei_ratios'][:min_len]) for r in rep_keys
        ])
        all_curves.append(matrix)

        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        ax.fill_between(gens, mean - std, mean + std,
                         color=color, alpha=0.12, linewidth=0)
        ax.plot(gens, mean, color=color, linewidth=1.5, label=mouse_id)

    if gens is None:
        ax.text(0.5, 0.5, "Cache not available\n(run ei_analysis.py on Colab)",
                ha='center', va='center', fontsize=FS_ANNOT, color='gray',
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Grand mean
    if all_curves:
        grand = np.concatenate(all_curves, axis=0)
        grand_mean = grand.mean(axis=0)
        ax.plot(gens, grand_mean, color='black', linewidth=2.0,
                alpha=0.8, zorder=10)

    # ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel("Generation")
    ax.set_xlim(gens[0], gens[-1])


def _panel_trajectories(ax, trajectories, maze_walls=None):
    """
    Plot representative agent trajectories in the maze.

    Parameters
    ----------
    trajectories : list of (label, color, xy_array) tuples
    maze_walls : (N, 2) array of wall segment endpoints, or None
    """
    if maze_walls is not None:
        ax.plot(maze_walls[:, 0], maze_walls[:, 1], 'k-',
                linewidth=0.4, alpha=0.3)

    for label, color, traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=0.5,
                alpha=0.7, label=label)

    ax.set_aspect('equal')
    ax.legend(loc='upper right', frameon=False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def figure3_new(ei_inter, ei_speed, ei_turn,
                ei_neuron_matrix=None,
                weight_similarity=None,
                mice_list=None,
                save_path=None):
    """
    Figure 4 — E/I Balance Dynamics.

    Layout (2×2):
        (A) Interneurons E/I  |  (B) Speed motor E/I
        (C) Turn motor E/I    |  (D) Per-neuron E/I heatmap

    weight_similarity is accepted for backward compatibility but not plotted;
    the cosine heatmap has moved to Figure 5A.
    """
    fig = plt.figure(figsize=(9, 6))
    gs = gridspec.GridSpec(
        2, 2,
        hspace=0.35, wspace=0.30,
        left=0.08, right=0.97, top=0.95, bottom=0.08,
    )

    # Panel A: Interneurons E/I timeseries
    ax_a = fig.add_subplot(gs[0, 0])
    _panel_ei_timeseries(ax_a, ei_inter, ylabel="E/(E+I) Balance",
                         show_xlabel=True)
    ax_a.set_title("A. E/I: Interneurons", loc="left",
                   fontweight="bold", fontsize=FS_TITLE)
    ax_a.legend(ncol=3, loc='lower center', frameon=False)

    # Panel B: Speed motor E/I timeseries
    ax_b = fig.add_subplot(gs[0, 1])
    _panel_ei_timeseries(ax_b, ei_speed, ylabel="E/(E+I) Balance",
                         show_xlabel=True)
    ax_b.set_title("B. E/I: Speed Motor", loc="left",
                   fontweight="bold", fontsize=FS_TITLE)

    # Panel C: Turn motor E/I timeseries
    ax_c = fig.add_subplot(gs[1, 0])
    _panel_ei_timeseries(ax_c, ei_turn, ylabel="E/(E+I) Balance",
                         show_xlabel=True)
    ax_c.set_title("C. E/I: Turn Motor", loc="left",
                   fontweight="bold", fontsize=FS_TITLE)

    # Panel D: Per-neuron E/I heatmap
    ax_d = fig.add_subplot(gs[1, 1])
    if ei_neuron_matrix is not None and mice_list is not None:
        _panel_ei_neuron_heatmap(ax_d, ei_neuron_matrix, mice_list)
    else:
        ax_d.axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        fig.savefig(save_path.replace(".png", ".svg"), dpi=300)
        print(f"Saved -> {save_path}")
    return fig


def _panel_ei_neuron_heatmap(ax, ei_matrix, mice_list):
    """
    Heatmap of per-neuron E/I ratios across all evolved agents.

    Parameters
    ----------
    ei_matrix : (N, 8) array — E/I ratio per (agent, non-sensory neuron)
    mice_list : list of N mouse IDs — determines column ordering
    """
    neuron_labels = ['I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'S', 'T']
    mouse_order = sorted(set(mice_list))

    ordered_idx = [i for m in mouse_order for i, am in enumerate(mice_list) if am == m]
    ordered = ei_matrix[ordered_idx, :].T   # (8, N)

    im = ax.imshow(ordered, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto',
                   interpolation='nearest')
    ax.set_yticks(range(8))
    ax.set_yticklabels(neuron_labels)
    ax.set_xticks([])

    # Horizontal separator between interneurons and motor neurons
    ax.axhline(5.5, color='black', linewidth=0.8, alpha=0.6)

    # Mouse group boundaries and labels
    tick_positions = []
    tick_labels = []
    tick_colors = []
    cumsum = 0
    for m in mouse_order:
        n_m = sum(1 for am in mice_list if am == m)
        if cumsum > 0:
            ax.axvline(cumsum - 0.5, color='black', linewidth=0.8, alpha=0.8)
        tick_positions.append(cumsum + n_m / 2 - 0.5)
        tick_labels.append(m)
        tick_colors.append(MOUSE_COLORS.get(m, '#333'))
        cumsum += n_m

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontweight='bold')
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    ax.tick_params(axis='x', length=0, pad=3)

    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("E/(E+I)")
    ax.set_title("D. Per-neuron E/I Balance", loc="left",
                 fontweight="bold", fontsize=FS_TITLE)


def _panel_weight_similarity(ax, ws_data):
    """
    Plot weight vector cosine similarity heatmap.

    Parameters
    ----------
    ws_data : dict with 'matrix' (N,N), 'mice' (list of mouse IDs per agent)
    """
    matrix = ws_data['matrix']
    agent_mice = ws_data['mice']

    # Order by mouse for block-diagonal
    mouse_order = sorted(set(agent_mice))
    ordered_idx = []
    for m in mouse_order:
        ordered_idx.extend([i for i, am in enumerate(agent_mice) if am == m])

    reordered = matrix[np.ix_(ordered_idx, ordered_idx)]

    # Fixed symmetric range: values cluster ~0.2 (warm) vs random near 0 (white)
    im = ax.imshow(reordered, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto',
                   interpolation='nearest')

    # Draw mouse group boundaries on both axes
    tick_positions = []
    tick_labels = []
    tick_colors = []
    cumsum = 0
    for m in mouse_order:
        n_m = sum(1 for am in agent_mice if am == m)
        if cumsum > 0:
            ax.axhline(cumsum - 0.5, color='black', linewidth=0.8, alpha=0.8)
            ax.axvline(cumsum - 0.5, color='black', linewidth=0.8, alpha=0.8)
        tick_positions.append(cumsum + n_m / 2 - 0.5)
        tick_labels.append(m)
        tick_colors.append(MOUSE_COLORS.get(m, '#333'))
        cumsum += n_m

    # Place mouse labels as x-tick labels below the image (no title overlap)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontweight='bold')
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    ax.tick_params(axis='x', length=0, pad=3)

    ax.set_yticks([])

    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Cosine similarity")


# ═══════════════════════════════════════════════════════════════════════════
# EXTRA FIGURES (colab_8)
# ═══════════════════════════════════════════════════════════════════════════

NEURON_LABELS = ['F', 'L', 'R', 'PSpd', 'PTrn', 'N',
                 'I0', 'I1', 'I2', 'I3', 'I4', 'I5',
                 'Spd', 'Trn']


def figure_fitness_boxplots(circuit_data, save_path=None):
    """Box plots of gen-150 best fitness per mouse."""
    mice = sorted(set(r['mouse'] for r in circuit_data))
    fig, ax = plt.subplots(figsize=(5, 3))

    data_per_mouse = []
    colors = []
    for m in mice:
        vals = [r['fitness'] for r in circuit_data if r['mouse'] == m]
        data_per_mouse.append(vals)
        colors.append(MOUSE_COLORS.get(m, '#888'))

    bp = ax.boxplot(data_per_mouse, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=1))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, (vals, c) in enumerate(zip(data_per_mouse, colors)):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   color=c, s=18, zorder=3, edgecolor='white', linewidth=0.3)

    grand_mean = np.mean([r['fitness'] for r in circuit_data])
    ax.axhline(grand_mean, ls='--', color='gray', lw=0.7, alpha=0.7,
               label=f'Grand mean ({grand_mean:.3f})')

    ax.set_xticklabels(mice)
    ax.set_xlabel('Mouse')
    ax.set_ylabel('Best Fitness (lower = better)')
    ax.set_title('Fitness Convergence by Mouse', fontweight='bold')
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'Saved -> {save_path}')
    return fig


def figure_forest_plot(evr_results, d_cis=None, save_path=None):
    """
    Forest plot of Cohen's d (evolved vs random) with 95% CIs.

    Args:
        evr_results: dict from evolved_vs_random()
        d_cis: dict of feature -> (d, d_lo, d_hi) from cohens_d_bootstrap_ci
        save_path: optional save path
    """
    # Sort features by absolute effect size
    feats = sorted(evr_results.keys(), key=lambda f: abs(evr_results[f]['cohens_d']))
    ds = [evr_results[f]['cohens_d'] for f in feats]
    ps = [evr_results[f]['p'] for f in feats]

    fig, ax = plt.subplots(figsize=(5, 5))
    y_pos = np.arange(len(feats))

    for i, (feat, d, p) in enumerate(zip(feats, ds, ps)):
        color = '#3C5488' if p < 0.05 else '#999999'
        ax.plot(d, i, 'o', color=color, markersize=5, zorder=3)

        if d_cis and feat in d_cis:
            _, d_lo, d_hi = d_cis[feat]
            ax.plot([d_lo, d_hi], [i, i], '-', color=color, lw=1.2, zorder=2)

    ax.axvline(0, ls='--', color='gray', lw=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats)
    ax.set_xlabel("Cohen's d (evolved − random)")
    ax.set_title('Evolved vs Random: Effect Sizes', fontweight='bold')

    # Legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker='o', color='#3C5488', lw=0, markersize=5, label='p < 0.05'),
        Line2D([0], [0], marker='o', color='#999999', lw=0, markersize=5, label='p ≥ 0.05'),
    ]
    ax.legend(handles=legend_els, loc='lower right')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'Saved -> {save_path}')
    return fig


def figure_power_curve(power_results, save_path=None):
    """Scatter of eta² vs achieved power for each feature."""
    feats = sorted(power_results.keys())
    eta2s = [power_results[f]['eta_squared'] for f in feats]
    powers = [power_results[f]['power'] for f in feats]
    min_eta2 = power_results[feats[0]].get('min_detectable_eta2', 0.247)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    for f, e, pw in zip(feats, eta2s, powers):
        color = '#E64B35' if pw < 0.8 else '#00A087'
        ax.scatter(e, pw, color=color, s=500, zorder=3, edgecolor='white', )
        ax.annotate(f, (e, pw), fontsize=10.5, ha='left', va='bottom',
                    xytext=(3, 2), textcoords='offset points')

    ax.axhline(0.8, ls='--', color='gray', lw=1.0, label='80% power threshold')
    ax.axvline(min_eta2, ls=':', color='gray', lw=1.0,
               label=f'Min detectable η²={min_eta2:.3f}')

    ax.set_xlabel('Observed η²')
    ax.set_ylabel('Achieved Power')

    ax.set_title('Post-Hoc Power Analysis', fontweight='bold')
    ax.set_xlim(-0.02, max(eta2s) + 0.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    sns.despine(ax=ax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        fig.savefig(save_path.replace(".png", ".svg"), dpi=300)
        print(f'Saved -> {save_path} + svg')
    return fig


def figure_specialization_bars(gen_matrix, mice, save_path=None):
    """Horizontal bar chart of per-mouse specialization ratios."""
    n = len(mice)
    ratios = []
    for i in range(n):
        diag = gen_matrix[i, i]
        off_diag = np.mean([gen_matrix[i, j] for j in range(n) if j != i])
        ratios.append(diag / off_diag)

    # Sort by ratio (most specialized first)
    order = np.argsort(ratios)
    sorted_mice = [mice[i] for i in order]
    sorted_ratios = [ratios[i] for i in order]
    sorted_colors = [MOUSE_COLORS.get(m, '#888') for m in sorted_mice]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    y_pos = np.arange(n)
    ax.barh(y_pos, sorted_ratios, color=sorted_colors, height=0.6, edgecolor='white')
    ax.axvline(1.0, ls='--', color='gray', lw=0.7, label='No specialization')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_mice)
    ax.set_xlabel('Specialization Ratio (diagonal / off-diagonal)')
    ax.set_title('Per-Mouse Behavioral Specialization', fontweight='bold')
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'Saved -> {save_path}')
    return fig


def figure_connection_heatmap(conn_results, save_path=None):
    """
    Two-panel figure: (A) -log10(p_fdr) heatmap, (B) top connections by mouse.
    """
    p_matrix = conn_results['p_matrix']
    presence = conn_results['presence']
    mice = conn_results['mice']
    sig_conns = conn_results['significant_connections']
    n = p_matrix.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4),
                             gridspec_kw={'width_ratios': [1, 1.2]})

    # Panel A: -log10(p) heatmap
    ax = axes[0]
    log_p = -np.log10(np.clip(p_matrix, 1e-10, 1.0))
    log_p = np.nan_to_num(log_p, nan=0.0)

    im = ax.imshow(log_p, cmap='YlOrRd', aspect='equal', origin='upper')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(NEURON_LABELS, rotation=45, ha='right')
    ax.set_yticklabels(NEURON_LABELS)
    ax.set_xlabel('Target neuron')
    ax.set_ylabel('Source neuron')
    ax.set_title('A   Connection-Level ANOVA\n(-log10 p_FDR)', loc='left',
                 fontweight='bold', fontsize=FS_TITLE)

    # Mark significant cells
    thresh = -np.log10(0.05)
    for i in range(n):
        for j in range(n):
            if log_p[i, j] > thresh:
                ax.plot(j, i, 'k*', markersize=4)

    # Boundary lines between neuron types
    for boundary in [5.5, 11.5]:
        ax.axhline(boundary, color='white', lw=0.5)
        ax.axvline(boundary, color='white', lw=0.5)

    cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.ax.tick_params(labelsize=FS_MICRO)

    # Panel B: top connections presence by mouse
    ax2 = axes[1]
    if len(sig_conns) > 0:
        n_show = min(10, len(sig_conns))
        top_conns = sig_conns[:n_show]
        conn_labels = [f'{NEURON_LABELS[i]}→{NEURON_LABELS[j]}'
                       for i, j, _, _ in top_conns]

        x = np.arange(len(mice))
        width = 0.8 / n_show
        for k, (i, j, F, p_fdr) in enumerate(top_conns):
            fracs = [presence[m][i, j] for m in mice]
            offset = (k - n_show / 2 + 0.5) * width
            bars = ax2.bar(x + offset, fracs, width, label=conn_labels[k],
                           alpha=0.8)

        ax2.set_xticks(x)
        ax2.set_xticklabels(mice)
        ax2.set_ylabel('Fraction of agents with connection')
        ax2.set_title('B   Significant Connections by Mouse', loc='left',
                       fontweight='bold', fontsize=FS_TITLE)
        ax2.legend(ncol=2, loc='upper right')
    else:
        ax2.text(0.5, 0.5, 'No significant connections\n(p_FDR < 0.05)',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=FS_ANNOT)
        ax2.set_title('B   Significant Connections by Mouse', loc='left',
                       fontweight='bold', fontsize=FS_TITLE)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'Saved -> {save_path}')
    return fig


def figure_required_reps(reps_data, anova_results, save_path=None):
    """
    Line curve: eta² vs required replicates, with features as labeled points.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Smooth curve
    eta2_range = np.linspace(0.02, 0.45, 200)
    from scipy.stats import ncf, f as f_dist
    k = 9
    req_ns = []
    for e2 in eta2_range:
        f_sq = e2 / (1 - e2)
        lo, hi = 2, 500
        for _ in range(50):
            mid = (lo + hi) // 2
            n_total = mid * k
            df1, df2 = k - 1, n_total - k
            ncp = f_sq * n_total
            f_crit = f_dist.ppf(0.95, df1, df2)
            pw = 1 - ncf.cdf(f_crit, df1, df2, ncp)
            if pw < 0.8:
                lo = mid + 1
            else:
                hi = mid
        req_ns.append(hi)

    ax.plot(eta2_range, req_ns, '-', color='#3C5488', lw=1.2)
    ax.axhline(6, ls='--', color='#E64B35', lw=0.8, label='Current n=6 per mouse')

    # Plot each feature
    for feat, r in reps_data.items():
        eta2 = r['eta_squared']
        req_n = r['required_n_per_group']
        if np.isinf(req_n) or eta2 <= 0:
            continue
        color = '#E64B35' if req_n > 6 else '#00A087'
        ax.scatter(eta2, min(req_n, 500), color=color, s=20, zorder=3,
                   edgecolor='white', linewidth=0.3)
        ax.annotate(feat, (eta2, min(req_n, 500)), fontsize=FS_MICRO,
                    xytext=(3, 2), textcoords='offset points')

    ax.set_xlabel('Observed η²')
    ax.set_ylabel('Required n per mouse (for 80% power)')
    ax.set_title('Replicates Needed to Detect Effects', fontweight='bold')
    ax.set_ylim(0, 60)
    ax.set_xlim(0, 0.45)
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f'Saved -> {save_path}')
    return fig

