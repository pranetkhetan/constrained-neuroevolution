"""
Main figure — §2.8 circuit comparison (fig:circuit_comparison).

3×3 layout:
  A  B  C
  D  E  F
  G  H  I

A: Joint KDE contours (topology Jaccard × behavioural cosine distance).
   Numbered dots mark the two within-mouse pairs selected for comparison:
     1 = most topologically similar within-mouse pair  (D9 r2/r5, min Jaccard)
     2 = most topologically dissimilar within-mouse pair (B5 r1/r5, max Jaccard)
B: Circuit diagram — D9 r2  (pair 1, circuit a)
C: Circuit diagram — D9 r5  (pair 1, circuit b)
D: Lyapunov exponent strip plot (per mouse, all 54 agents; KW null)
E: Circuit diagram — B5 r1  (pair 2, circuit a)
F: Circuit diagram — B5 r5  (pair 2, circuit b)
G: Perturbation decay curves (log |δh(t)|/|δh(0)|, 4 selected agents overlaid)
H: Driven trajectory PCA (interneuron+motor activations, within-pair overlay)
I: Autonomous attractor terminal-state PCA (200 random inits × 4 agents)

Circuit diagrams use draw_network_panel from sim_setup_figure.py (same as Fig 1).
"""

import os
import sys

import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

_PROJECT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.sim_setup_figure import draw_network_panel
from scripts.figure_modules._style import (
    apply_pub_style,
    FIGSIZE,
    FS_ANNOT, FS_LABEL, FS_LEGEND, FS_PANEL, FS_TICK,
    MICE, MOUSE_COLORS, label_panel, pub_despine, save_figure,
    LW_SCALE, MARKER_SCALE,
)

# ── KDE pair-type colours (must match figA Panel C) ───────────────────────────
WITHIN_COL  = "#CC6677"   # rose   — within-mouse pairs
BETWEEN_COL = "#332288"   # indigo — between-mouse pairs

# ── Dot colour for the two selected pairs ─────────────────────────────────────
DOT_COL = "#1a237e"       # deep navy

# ── Colours for dynamics panels G / H / I ─────────────────────────────────────
# Pair 1 (D9r2, D9r5): topologically similar — warm tones
P1A_COL = "#CC6677"       # rose       — D9r2
P1B_COL = "#882255"       # wine       — D9r5
# Pair 2 (B5r1, B5r5): topologically dissimilar — cool tones
P2A_COL = "#332288"       # indigo     — B5r1
P2B_COL = "#117733"       # teal-green — B5r5

GEN = 150
MICE_ORDER = MICE   # ['B5','B6','B7','D3','D4','D5','D7','D8','D9']


# ── Helpers ──────────────────────────────────────────────────────────────────

def _agent_idx(mouse: str, rep: int) -> int:
    return MICE_ORDER.index(mouse) * 6 + (rep - 1)


def _load_best_agent(project_dir: str, mouse: str, rep: int):
    # Prefer data/best_agents.pkl (repo default); falls back to data/agents/.
    from scripts.figure_modules._loaders import load_best_agent as _lba
    return _lba(project_dir, mouse, rep, gen=GEN)


# ── Panel A: KDE landscape ────────────────────────────────────────────────────

def _panel_A(ax: plt.Axes, A1: dict, A5: dict,
              sim_jac: float, sim_beh: float,
              dis_jac: float, dis_beh: float) -> None:
    topo  = np.array(A1["topo_dists"])
    beh   = np.array(A1["beh_dists_upper"])
    ptypes = np.array(A1["pair_types"])

    within  = ptypes == "within"
    between = ptypes == "between"

    kw = dict(levels=5, thresh=0.10, bw_adjust=0.8)
    for mask, col in [(between, BETWEEN_COL), (within, WITHIN_COL)]:
        x, y = topo[mask], beh[mask]
        kd = dict(x=x, y=y, ax=ax, **kw)
        sns.kdeplot(**kd, fill=True,  color=col, alpha=0.15, zorder=1)
        sns.kdeplot(**kd, fill=False, color=col, alpha=0.70, linewidths=1.0 * LW_SCALE, zorder=2)

    rho_ev, _ = spearmanr(topo, beh)
    rho_rand  = float(A5["rho_rand"])
    ax.text(
        0.03, 0.97,
        f"Evolved:  $\\rho = {rho_ev:+.3f}$\n"
        f"Random:  $\\rho = {rho_rand:+.3f}$",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=FS_ANNOT, color="#333",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#ccc", alpha=0.85),
        zorder=6,
    )

    ax.set_xlabel("Topology distance (Jaccard)", fontsize=FS_LABEL)
    ax.set_ylabel("Behavioural distance (cosine)", fontsize=FS_LABEL)
    ax.set_xlim(0.6, 1)
    ax.set_ylim(-0.01, 0.07)
    ax.tick_params(labelsize=FS_TICK)
    pub_despine(ax)

    # Numbered dots
    dot_kw = dict(s=220 * MARKER_SCALE, color=DOT_COL, edgecolors="white", linewidths=1.4 * LW_SCALE, zorder=10)
    y_nudge = 0.0025
    for jac, bh, lbl in [(sim_jac, sim_beh, "1"), (dis_jac, dis_beh, "2")]:
        ax.scatter([jac], [bh], **dot_kw)
        ax.text(jac, bh + y_nudge, lbl, ha="center", va="bottom",
                fontsize=FS_LABEL, fontweight="bold", color=DOT_COL, zorder=11)


# ── Panel D: Lyapunov strip ───────────────────────────────────────────────────

def _panel_D(ax: plt.Axes, lya: dict,
             sim_mouse: str, sim_rep_a: int, sim_rep_b: int,
             dis_mouse: str, dis_rep_a: int, dis_rep_b: int) -> None:
    lya_by_mouse = lya["lya_by_mouse"]
    lya_results  = lya["lya_results"]
    kw_p  = float(lya["p_kruskal"])

    rng = np.random.default_rng(42)

    # Collect per-agent values for the 4 selected agents
    selected = {
        _agent_idx(sim_mouse, sim_rep_a): (P1A_COL, "★"),
        _agent_idx(sim_mouse, sim_rep_b): (P1B_COL, "★"),
        _agent_idx(dis_mouse, dis_rep_a): (P2A_COL, "★"),
        _agent_idx(dis_mouse, dis_rep_b): (P2B_COL, "★"),
    }

    for xi, mouse in enumerate(MICE_ORDER):
        vals = np.array(lya_by_mouse[mouse])
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(
            np.full(len(vals), xi) + jitter, vals,
            color=MOUSE_COLORS[mouse], s=32 * MARKER_SCALE, alpha=0.75,
            edgecolors="white", linewidths=0.4 * LW_SCALE, zorder=3,
        )
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([xi - 0.28, xi + 0.28], [med, med], color="black", lw=1.4 * LW_SCALE, zorder=4)
        ax.plot([xi, xi], [q1, q3], color="black", lw=0.9 * LW_SCALE, zorder=4)

    # Overlay selected agents with larger coloured markers
    for glob_idx, (col, _) in selected.items():
        r = lya_results[glob_idx]
        xi = MICE_ORDER.index(r["mouse"])
        ax.scatter([xi], [r["lambda1"]], color=col, s=90 * MARKER_SCALE, marker="D",
                   edgecolors="white", linewidths=0.8 * LW_SCALE, zorder=6)

    ax.axhline(0, ls="--", color="gray", lw=1.0 * LW_SCALE, alpha=0.8)
    ax.text(8.6, 0.003, "$\\lambda_1 = 0$", fontsize=FS_ANNOT,
            color="gray", va="bottom", ha="right")

    p_str = f"KW $p = {kw_p:.3f}$" if kw_p >= 0.001 else "KW $p < 0.001$"
    ax.text(0.02, 0.97, p_str + "  (n.s.)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_ANNOT, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.85))

    ax.set_xticks(range(len(MICE_ORDER)))
    ax.set_xticklabels(MICE_ORDER, fontsize=FS_TICK)
    ax.set_ylabel("Maximal Lyapunov exp. ($\\lambda_1$)", fontsize=FS_LABEL)
    pub_despine(ax)

    # Diamond legend for selected pairs
    hl = [
        mpatches.Patch(color=P1A_COL, label=f"Pair 1 ({sim_mouse} r{sim_rep_a})"),
        mpatches.Patch(color=P1B_COL, label=f"Pair 1 ({sim_mouse} r{sim_rep_b})"),
        mpatches.Patch(color=P2A_COL, label=f"Pair 2 ({dis_mouse} r{dis_rep_a})"),
        mpatches.Patch(color=P2B_COL, label=f"Pair 2 ({dis_mouse} r{dis_rep_b})"),
    ]
    ax.legend(handles=hl, frameon=False, fontsize=FS_ANNOT,
              loc="lower right", ncol=2)


# ── Panel G: perturbation decay curves ───────────────────────────────────────

def _panel_G(ax: plt.Axes, lya_results: list,
             sim_mouse: str, sim_rep_a: int, sim_rep_b: int,
             dis_mouse: str, dis_rep_a: int, dis_rep_b: int) -> None:
    steps = np.arange(400)

    # Background: per-mouse mean log_div (thin, mouse-coloured)
    for mouse in MICE_ORDER:
        mouse_curves = [
            np.array(lya_results[_agent_idx(mouse, r)]["log_div"])
            for r in range(1, 7)
        ]
        mean_curve = np.mean(mouse_curves, axis=0)
        ax.plot(steps, mean_curve, color=MOUSE_COLORS[mouse],
                lw=0.9 * LW_SCALE, alpha=0.45, zorder=2)

    # Foreground: 4 selected agents (bold, labelled)
    specs = [
        (_agent_idx(sim_mouse, sim_rep_a), P1A_COL, "-",  f"Pair 1: {sim_mouse} r{sim_rep_a}"),
        (_agent_idx(sim_mouse, sim_rep_b), P1B_COL, "--", f"Pair 1: {sim_mouse} r{sim_rep_b}"),
        (_agent_idx(dis_mouse, dis_rep_a), P2A_COL, "-",  f"Pair 2: {dis_mouse} r{dis_rep_a}"),
        (_agent_idx(dis_mouse, dis_rep_b), P2B_COL, "--", f"Pair 2: {dis_mouse} r{dis_rep_b}"),
    ]
    for idx, col, ls, label in specs:
        curve = np.array(lya_results[idx]["log_div"])
        lam   = lya_results[idx]["lambda1"]
        ax.plot(steps, curve, color=col, ls=ls, lw=2.2 * LW_SCALE, alpha=0.95, zorder=4,
                label=f"{label}  ($\\lambda_1 = {lam:.3f}$)")

    ax.axhline(0, ls=":", color="gray", lw=0.8 * LW_SCALE, alpha=0.7)
    ax.text(2, 0.3, "$\\lambda_1 = 0$", fontsize=FS_ANNOT, color="gray", va="bottom")
    ax.set_xlabel("Step", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\log\,\|\delta h(t)\| / \|\delta h(0)\|$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="upper right")
    pub_despine(ax)


# ── Panel H: driven trajectory PCA ───────────────────────────────────────────

def _panel_H(ax: plt.Axes, raw_acts: list, exit_frames: list,
             dis_mouse: str, dis_rep_a: int, dis_rep_b: int,
             dis_jac: float = 0.0) -> None:
    """
    Pair-2 driven trajectory PCA (dis_mouse r_a vs r_b) — full panel.
    Pair 2 is chosen because it has the maximum within-mouse topology distance,
    so trajectory overlap here is the strongest evidence of degeneracy.
    Interneuron+motor activations (dims 6-13) from all valid maze bouts.
    """
    INTER_DIMS = slice(6, 14)

    idx_a = _agent_idx(dis_mouse, dis_rep_a)
    idx_b = _agent_idx(dis_mouse, dis_rep_b)

    acts_a = np.array(raw_acts[idx_a])     # (20, 2000, 14)
    ef_a   = np.array(exit_frames[idx_a])  # (20,)
    acts_b = np.array(raw_acts[idx_b])
    ef_b   = np.array(exit_frames[idx_b])

    valid_a = [(b, int(ef_a[b])) for b in range(20) if ef_a[b] < 2000]
    valid_b = [(b, int(ef_b[b])) for b in range(20) if ef_b[b] < 2000]

    frames_a = [acts_a[b, :n, INTER_DIMS] for b, n in valid_a]
    frames_b = [acts_b[b, :n, INTER_DIMS] for b, n in valid_b]
    pooled   = np.vstack(frames_a + frames_b)

    pca = PCA(n_components=2, random_state=42)
    pca.fit(pooled)
    var = pca.explained_variance_ratio_ * 100

    for b, n in valid_a:
        traj = pca.transform(acts_a[b, :n, INTER_DIMS])
        ax.plot(traj[:, 0], traj[:, 1], color=P2A_COL, lw=0.9 * LW_SCALE, alpha=0.40)
    ax.plot([], [], color=P2A_COL, lw=2.0 * LW_SCALE,
            label=f"{dis_mouse} r{dis_rep_a}  (Jaccard high end)")

    for b, n in valid_b:
        traj = pca.transform(acts_b[b, :n, INTER_DIMS])
        ax.plot(traj[:, 0], traj[:, 1], color=P2B_COL, lw=0.9 * LW_SCALE, alpha=0.40, ls="--")
    ax.plot([], [], color=P2B_COL, lw=2.0 * LW_SCALE, ls="--",
            label=f"{dis_mouse} r{dis_rep_b}  (Jaccard low end)")

    n_a = sum(n for _, n in valid_a)
    n_b = sum(n for _, n in valid_b)
    ax.text(0.02, 0.98,
            f"Jaccard distance = {dis_jac:.3f} (maximum within-mouse)\n"
            f"n frames: r{dis_rep_a}={n_a}, r{dis_rep_b}={n_b}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_ANNOT, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.85))

    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var.)", fontsize=FS_LABEL)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var.)", fontsize=FS_LABEL)
    ax.set_title("")
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="lower right")
    pub_despine(ax)


# ── Panel I: attractor terminal-state PCA (all 54 agents) ────────────────────

def _panel_I(ax: plt.Axes, all_finals: list,
             sim_mouse: str, sim_rep_a: int, sim_rep_b: int,
             dis_mouse: str, dis_rep_a: int, dis_rep_b: int) -> None:
    """
    All-54-agent autonomous attractor PCA (replicates supp_d4 Panel C) with
    the four selected circuits highlighted by larger markers.
    """
    mouse_per_agent = [m for m in MICE_ORDER for _ in range(6)]
    n_agents = len(all_finals)

    # Shared PCA over all 54 × 200 terminal states
    pooled = np.vstack([np.array(all_finals[i]) for i in range(n_agents)])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(pooled)
    var = pca.explained_variance_ratio_ * 100

    # Background: all 54 agents (small, mouse colour, low alpha)
    for i in range(n_agents):
        coords = pca.transform(np.array(all_finals[i]))
        mouse  = mouse_per_agent[i]
        ax.scatter(coords[:, 0], coords[:, 1],
                   color=MOUSE_COLORS[mouse], s=20 * MARKER_SCALE, alpha=0.2,
                   edgecolors="none", zorder=2)

    # Foreground: 4 selected circuits (larger, distinct markers, labelled)
    selected = [
        (_agent_idx(sim_mouse, sim_rep_a), P1A_COL, "o", f"Pair 1: {sim_mouse} r{sim_rep_a}"),
        (_agent_idx(sim_mouse, sim_rep_b), P1B_COL, "s", f"Pair 1: {sim_mouse} r{sim_rep_b}"),
        (_agent_idx(dis_mouse, dis_rep_a), P2A_COL, "^", f"Pair 2: {dis_mouse} r{dis_rep_a}"),
        (_agent_idx(dis_mouse, dis_rep_b), P2B_COL, "D", f"Pair 2: {dis_mouse} r{dis_rep_b}"),
    ]
    for idx, col, mk, label in selected:
        coords = pca.transform(np.array(all_finals[idx]))
        ax.scatter(coords[:, 0], coords[:, 1],
                   color=col, marker=mk, s=20 * MARKER_SCALE, alpha=0.55,
                   edgecolors="none", zorder=4)
        ctr = coords.mean(axis=0)
        ax.scatter([ctr[0]], [ctr[1]], color=col, marker=mk, s=40 * MARKER_SCALE,
                   edgecolors="white", linewidths=0.8 * LW_SCALE, zorder=6, label=label)

    ax.axhline(0, ls=":", color="#aaa", lw=0.7 * LW_SCALE, zorder=0)
    ax.axvline(0, ls=":", color="#aaa", lw=0.7 * LW_SCALE, zorder=0)
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var.)", fontsize=FS_LABEL)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var.)", fontsize=FS_LABEL)
    ax.set_title("")
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(frameon=False, fontsize=FS_ANNOT, loc="best", markerscale=1.2)
    pub_despine(ax)


# ── Main generate function ────────────────────────────────────────────────────

def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style(font_scale=1.55)
    project_dir  = store._project
    analysis_dir = store._analysis

    # ── Load data ─────────────────────────────────────────────────────────────
    import pickle

    def _lpkl(path):
        class _CPU(pickle.Unpickler):
            def find_class(self, m, n):
                if m.startswith("cupy"):
                    m = m.replace("cupy._core.core", "numpy").replace("cupy", "numpy")
                return super().find_class(m, n)
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (ModuleNotFoundError, AttributeError):
            with open(path, "rb") as f:
                return _CPU(f).load()

    circ = _lpkl(os.path.join(analysis_dir, "activity_embeddings", "circuits_results.pkl"))
    A1   = _lpkl(os.path.join(analysis_dir, "degeneracy_analyses", "A1_results.pkl"))
    A5   = _lpkl(os.path.join(analysis_dir, "degeneracy_analyses", "A5_results.pkl"))

    dyn_full = store.dynamics_results_full()
    lya      = dyn_full["lyapunov"]

    B  = store.act_emb_b()
    raw_acts    = B["raw_acts"]
    exit_frames = B["exit_frames"]

    D4 = store.act_emb_d()["D4"]
    all_finals = D4["all_finals"]

    # ── Pair coordinates ──────────────────────────────────────────────────────
    sim_mouse = circ["sim_mouse"]; sim_rep_a = circ["sim_rep_a"]; sim_rep_b = circ["sim_rep_b"]
    dis_mouse = circ["dis_mouse"]; dis_rep_a = circ["dis_rep_a"]; dis_rep_b = circ["dis_rep_b"]
    sim_jac = float(circ["sim_jaccard"]); sim_beh = float(circ["sim_beh_dist"])
    dis_jac = float(circ["dis_jaccard"]); dis_beh = float(circ["dis_beh_dist"])

    # ── Load best agents ──────────────────────────────────────────────────────
    sim_ag_a = _load_best_agent(project_dir, sim_mouse, sim_rep_a)
    sim_ag_b = _load_best_agent(project_dir, sim_mouse, sim_rep_b)
    dis_ag_a = _load_best_agent(project_dir, dis_mouse, dis_rep_a)
    dis_ag_b = _load_best_agent(project_dir, dis_mouse, dis_rep_b)

    # ── Layout ────────────────────────────────────────────────────────────────
    # 3×3 grid.  Row 0 = A/B/C, Row 1 = D/E/F, Row 2 = G/H/I
    fig = plt.figure(figsize=FIGSIZE['circuit'])
    outer = mgridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.3, wspace=0.32,
        left=0.07, right=0.97, top=0.97, bottom=0.04,
    )

    ax_A = fig.add_subplot(outer[0, 0])
    ax_B = fig.add_subplot(outer[0, 1])
    ax_C = fig.add_subplot(outer[0, 2])
    ax_D = fig.add_subplot(outer[1, 0])
    ax_E = fig.add_subplot(outer[1, 1])
    ax_F = fig.add_subplot(outer[1, 2])
    ax_G = fig.add_subplot(outer[2, 0])
    ax_H = fig.add_subplot(outer[2, 1])   # will be split by _panel_H
    ax_I = fig.add_subplot(outer[2, 2])

    # ── Populate ──────────────────────────────────────────────────────────────
    _panel_A(ax_A, A1, A5, sim_jac, sim_beh, dis_jac, dis_beh)

    draw_network_panel(ax_B, sim_ag_a)
    ax_B.set_title(f"{sim_mouse} r{sim_rep_a}", fontsize=FS_TICK, pad=4)

    draw_network_panel(ax_C, sim_ag_b)
    ax_C.set_title(f"{sim_mouse} r{sim_rep_b}  (J = {sim_jac:.2f})",
                   fontsize=FS_TICK, pad=4)

    _panel_D(ax_D, lya, sim_mouse, sim_rep_a, sim_rep_b, dis_mouse, dis_rep_a, dis_rep_b)

    draw_network_panel(ax_E, dis_ag_a)
    ax_E.set_title(f"{dis_mouse} r{dis_rep_a}", fontsize=FS_TICK, pad=4)

    draw_network_panel(ax_F, dis_ag_b)
    ax_F.set_title(f"{dis_mouse} r{dis_rep_b}  (J = {dis_jac:.2f})",
                   fontsize=FS_TICK, pad=4)

    _panel_G(ax_G, lya["lya_results"], sim_mouse, sim_rep_a, sim_rep_b,
             dis_mouse, dis_rep_a, dis_rep_b)

    _panel_H(ax_H, raw_acts, exit_frames, dis_mouse, dis_rep_a, dis_rep_b, dis_jac)

    _panel_I(ax_I, all_finals, sim_mouse, sim_rep_a, sim_rep_b,
             dis_mouse, dis_rep_a, dis_rep_b)

    # ── Panel labels ─────────────────────────────────────────────────────────
    label_panel(ax_A, "A")
    label_panel(ax_B, "B")
    label_panel(ax_C, "C")
    label_panel(ax_D, "D")
    label_panel(ax_E, "E")
    label_panel(ax_F, "F")
    label_panel(ax_G, "G")
    label_panel(ax_H, "H")
    label_panel(ax_I, "I")

    out = os.path.join(figures_dir, "fig_circuit_comparison.pdf")
    return save_figure(fig, out)