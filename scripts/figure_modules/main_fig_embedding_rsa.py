"""
paper_v2 §2.7 — Representational geometry degeneracy figure.

3×3 layout (20 × 18 in):

  Row 1 (summary):
    A  B4 KDE — within vs between Procrustes distance + generalist band
    B  D1/D2/D3 Mantel ρ barplot — dissociation from all three axes
    C  54×54 Procrustes distance heatmap — no block structure on diagonal

  Row 2 (KDE jointplots, within=red / between=blue):
    D  rep vs behavioral distance
    E  rep vs topological distance
    F  rep vs sensitivity distance

  Row 3 (scatter, between=grey / within=mouse-colour):
    G  rep vs behavioral distance
    H  rep vs topological distance
    I  rep vs sensitivity distance

Data sources:
  analysis/activity_embeddings/B_results.pkl  — B4 (Procrustes), B5 (generalist)
  analysis/activity_embeddings/D_results.pkl  — D1/D2/D3 Mantel rho + p
  analysis/degeneracy_analyses/A1_results.pkl — beh_dists_upper, topo_dists
  analysis/degeneracy_analyses/A4_results.pkl — sens_RSM
  Optional .npy overrides: beh_upper, sens_upper, pair_indices, rep_upper_dist, pair_types

Note: B_results.pkl and D_results.pkl values are CORRECT after notebook rerun
(procrustes_distance bug fixed in Cell 23 of colab_20_activity_embeddings.ipynb).
No scale→distance or rho-sign corrections are applied here.

Output: figures/fig_embedding_rsa_v2.pdf + .png
"""

import os
import sys
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import gaussian_kde

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figure_modules._style import (
    apply_pub_style, save_figure, label_panel, pub_despine,
    FIGSIZE, MOUSE_COLORS, MICE, GEN_COL,
    FS_PANEL, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    LW_SCALE, MARKER_SCALE,
)

_WITHIN_COL  = "#CC6677"
_BETWEEN_COL = "#88CCEE"
_N_REPS      = 6


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_pkl(path):
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


def _sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _clean_marginal(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_xlabel("")
    ax.set_ylabel("")


def _make_joint_axes(fig, subplot_spec):
    """Subfigure with 2×2 nested grid → (main, top-marginal, right-marginal) axes."""
    sfig = fig.add_subfigure(subplot_spec)
    gs = sfig.add_gridspec(
        2, 2,
        height_ratios=[1, 5],
        width_ratios=[5, 1],
        hspace=0.04, wspace=0.04,
    )
    ax_main  = sfig.add_subplot(gs[1, 0])
    ax_top   = sfig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = sfig.add_subplot(gs[1, 1], sharey=ax_main)
    corner   = sfig.add_subplot(gs[0, 1])
    corner.axis("off")
    return ax_main, ax_top, ax_right


# ── Row 1 panels ──────────────────────────────────────────────────────────────

def _panel_a_kde(ax, within_proc, between_proc, gen_dists, p_proc):
    """B4 KDE: within / between / generalist Procrustes distances."""
    xs_min = min(within_proc.min(), between_proc.min()) - 0.15
    xs_max = max(within_proc.max(), between_proc.max()) + 0.15
    xs = np.linspace(xs_min, xs_max, 400)

    def _kde(data, color, label, ls="-"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = gaussian_kde(data, bw_method=0.25)
        ax.plot(xs, k(xs), color=color, lw=1.8 * LW_SCALE, ls=ls, label=label)
        ax.fill_between(xs, k(xs), alpha=0.12, color=color)

    _kde(within_proc,  _WITHIN_COL,  f"Within-mouse (n={len(within_proc)})")
    _kde(between_proc, _BETWEEN_COL, f"Between-mouse (n={len(between_proc)})")

    if gen_dists is not None and len(gen_dists) > 2:
        gm, gs_ = gen_dists.mean(), gen_dists.std()
        ax.axvline(gm, color=GEN_COL, lw=1.4 * LW_SCALE, ls="--",
                   label=f"Generalist mean ± SD (n={len(gen_dists)})")
        ax.axvspan(gm - gs_, gm + gs_, color=GEN_COL, alpha=0.10)

    p_str = f"MW p = {p_proc:.3f}" if p_proc >= 0.001 else "MW p < 0.001"
    ax.text(0.9, 0.93, p_str, transform=ax.transAxes,
            ha="right", va="top", fontsize=FS_ANNOT, color="0.35")

    ax.set_xlabel("Procrustes distance (activity embeddings)", fontsize=FS_TICK)
    ax.set_ylabel("Density", fontsize=FS_TICK)
    ax.legend(frameon=False, fontsize=FS_LEGEND, loc="lower left")
    pub_despine(ax)
    label_panel(ax, "A")


def _panel_b_mantel(ax, rho_D1, p_D1, rho_D2, p_D2, rho_D3, p_D3):
    """Horizontal barplot: D1/D2/D3 Mantel ρ with significance labels."""
    rows = [
        ("Behavioral",  rho_D1, p_D1, _BETWEEN_COL),
        ("Topological", rho_D2, p_D2, "#CC3333"),
        ("Sensitivity", rho_D3, p_D3, _BETWEEN_COL),
    ]
    ys = np.arange(len(rows))
    for i, (_, rho, _, color) in enumerate(rows):
        ax.barh(i, rho, color=color, alpha=0.70, height=0.50)

    ax.axvline(0, color="k", lw=0.8 * LW_SCALE, zorder=5)
    for i, (_, rho, p, _) in enumerate(rows):
        sig  = _sig_label(p)
        xoff = 0.006 if rho >= 0 else -0.006
        ha   = "left"  if rho >= 0 else "right"
        ax.text(rho + xoff, i, sig, va="center", ha=ha, fontsize=FS_ANNOT)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS_TICK)
    ax.set_xlabel("Mantel ρ (representational vs predictor)", fontsize=FS_TICK)
    pub_despine(ax)
    label_panel(ax, "B")


def _panel_c_heatmap(ax, D_proc, agent_mouse_labels):
    """54×54 Procrustes distance heatmap; agents grouped by mouse (6 reps each)."""
    # Exclude diagonal zeros from colour scale so within/between contrast is visible
    off_diag = D_proc[~np.eye(len(D_proc), dtype=bool)]
    vmin, vmax = off_diag.min(), off_diag.max()
    D_plot = D_proc.copy().astype(float)
    np.fill_diagonal(D_plot, np.nan)   # diagonal rendered as grey (masked)

    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad(color="0.85")         # grey for masked diagonal

    im = ax.imshow(D_plot, aspect="equal", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    # Mouse group boundaries (every 6 agents)
    for k in range(_N_REPS, 54, _N_REPS):
        ax.axhline(k - 0.5, color="white", lw=1.0 * LW_SCALE, zorder=5)
        ax.axvline(k - 0.5, color="white", lw=1.0 * LW_SCALE, zorder=5)

    # Tick labels at group centres
    centres = [_N_REPS * i + _N_REPS / 2 - 0.5 for i in range(len(MICE))]
    ax.set_xticks(centres)
    ax.set_yticks(centres)
    ax.set_xticklabels(MICE, fontsize=FS_TICK - 1, rotation=45, ha="right")
    ax.set_yticklabels(MICE, fontsize=FS_TICK - 1)

    plt.colorbar(im, ax=ax, label="Procrustes distance", shrink=0.80,
                 pad=0.02, fraction=0.046)
    label_panel(ax, "C")


# ── Row 2: KDE jointplots (within=red, between=blue) ─────────────────────────

def _joint_kde(ax_main, ax_top, ax_right,
               x, y, pair_types,
               x_label, y_label, rho, p_val, panel_label,
               x_lim=None):
    """KDE contour joint plot, within=_WITHIN_COL, between=_BETWEEN_COL."""
    within_mask  = pair_types == "within"
    between_mask = ~within_mask

    for mask, color, label in [
        (between_mask, _BETWEEN_COL, f"Between-mouse (n={between_mask.sum()})"),
        (within_mask,  _WITHIN_COL,  f"Within-mouse (n={within_mask.sum()})"),
    ]:
        kw = dict(x=x[mask], y=y[mask], levels=5, thresh=0.10,
                  bw_adjust=0.8, ax=ax_main)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.kdeplot(**kw, fill=True,  color=color, alpha=0.15, zorder=1)
            sns.kdeplot(**kw, fill=False, color=color, alpha=0.75,
                        linewidths=1.0 * LW_SCALE, zorder=2, label=label)

    sig   = _sig_label(p_val)
    p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    ax_main.text(
        0.97, 0.05,
        f"Mantel $\\rho = {rho:+.3f}$ {sig}\n{p_str}",
        transform=ax_main.transAxes, ha="right", va="bottom",
        fontsize=FS_ANNOT, color="#333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
        zorder=6,
    )
    ax_main.set_xlabel(x_label, fontsize=FS_LABEL)
    ax_main.set_ylabel(y_label, fontsize=FS_LABEL)
    ax_main.tick_params(labelsize=FS_TICK)
    # Manual legend — seaborn kdeplot doesn't propagate labels reliably
    handles = [
        mpatches.Patch(color=_BETWEEN_COL, alpha=0.65,
                       label=f"Between-mouse (n={between_mask.sum()})"),
        mpatches.Patch(color=_WITHIN_COL,  alpha=0.65,
                       label=f"Within-mouse (n={within_mask.sum()})"),
    ]
    ax_main.legend(handles=handles, frameon=False, fontsize=FS_LEGEND, loc="upper right")
    if x_lim is not None:
        ax_main.set_xlim(x_lim)
    sns.despine(ax=ax_main, offset=1, trim=True,)
    label_panel(ax_main, panel_label)

    # Marginals
    for mask, color in [(between_mask, _BETWEEN_COL), (within_mask, _WITHIN_COL)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.kdeplot(x[mask], ax=ax_top, color=color,
                        fill=True, alpha=0.25, linewidth=1.0 * LW_SCALE, bw_adjust=0.8)
            sns.kdeplot(y=y[mask], ax=ax_right, color=color,
                        fill=True, alpha=0.25, linewidth=1.0 * LW_SCALE, bw_adjust=0.8)
    _clean_marginal(ax_top)
    _clean_marginal(ax_right)


# ── Row 3: scatter jointplots (between=grey, within=mouse-colour) ─────────────

def _joint_mouse(ax_main, ax_top, ax_right,
                 x, y, pair_types, pair_mouse_idx,
                 x_label, y_label, rho, p_val, panel_label,
                 x_lim=None):
    """Scatter joint plot: between-mouse=grey, within-mouse=per-mouse colour."""
    within_mask  = pair_types == "within"
    between_mask = ~within_mask

    # Grey background: between-mouse pairs
    ax_main.scatter(
        x[between_mask], y[between_mask],
        s=2 * MARKER_SCALE, alpha=0.07, color="#cccccc", linewidths=0 * LW_SCALE,
        rasterized=True, zorder=1,
    )

    # Within-mouse pairs coloured by mouse
    for m_idx, mouse in enumerate(MICE):
        mask = pair_mouse_idx == m_idx
        if not mask.any():
            continue
        ax_main.scatter(
            x[mask], y[mask],
            s=24 * MARKER_SCALE, alpha=0.80, color=MOUSE_COLORS[mouse],
            linewidths=0 * LW_SCALE, rasterized=True, zorder=3, label=mouse,
        )

    sig   = _sig_label(p_val)
    p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    ax_main.text(
        0.97, 0.05,
        f"Mantel $\\rho = {rho:+.3f}$ {sig}\n{p_str}",
        transform=ax_main.transAxes, ha="right", va="bottom",
        fontsize=FS_ANNOT, color="#333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
        zorder=6,
    )
    ax_main.set_xlabel(x_label, fontsize=FS_LABEL)
    ax_main.set_ylabel(y_label, fontsize=FS_LABEL)
    ax_main.tick_params(labelsize=FS_TICK)
    if x_lim is not None:
        ax_main.set_xlim(x_lim)
    sns.despine(ax=ax_main, offset=1, trim=True,)
    label_panel(ax_main, panel_label)

    # Top marginal: grey KDE for between + per-mouse lines for within
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.kdeplot(x[between_mask], ax=ax_top, color="#bbbbbb",
                    fill=True, alpha=0.30, linewidth=0.8 * LW_SCALE, bw_adjust=0.8)
        for m_idx, mouse in enumerate(MICE):
            mask = pair_mouse_idx == m_idx
            if mask.sum() < 3:
                continue
            sns.kdeplot(x[mask], ax=ax_top, color=MOUSE_COLORS[mouse],
                        linewidth=0.9 * LW_SCALE, bw_adjust=1.0, alpha=0.75)

    # Right marginal: same but y-axis orientation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.kdeplot(y=y[between_mask], ax=ax_right, color="#bbbbbb",
                    fill=True, alpha=0.30, linewidth=0.8 * LW_SCALE, bw_adjust=0.8)
        for m_idx, mouse in enumerate(MICE):
            mask = pair_mouse_idx == m_idx
            if mask.sum() < 3:
                continue
            sns.kdeplot(y=y[mask], ax=ax_right, color=MOUSE_COLORS[mouse],
                        linewidth=0.9 * LW_SCALE, bw_adjust=1.0, alpha=0.75)

    _clean_marginal(ax_top)
    _clean_marginal(ax_right)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(store, figures_dir: str) -> list[str]:
    """Generate fig_embedding_rsa_v2.pdf/.png and return list of output paths."""
    apply_pub_style(font_scale=1.55)

    act_dir = os.path.join(store._analysis, "activity_embeddings")
    deg_dir = os.path.join(store._analysis, "degeneracy_analyses")

    B  = _load_pkl(os.path.join(act_dir, "B_results.pkl"))
    D  = _load_pkl(os.path.join(act_dir, "D_results.pkl"))
    A1 = _load_pkl(os.path.join(deg_dir, "A1_results.pkl"))
    A4 = _load_pkl(os.path.join(deg_dir, "A4_results.pkl"))

    # ── B4: Procrustes distances (correct after notebook fix) ─────────────
    B4           = B["B4"]
    within_proc  = np.asarray(B4["within_proc"])
    between_proc = np.asarray(B4["between_proc"])
    p_proc       = float(B4["p_proc"])
    D_proc_54    = np.asarray(B4["D_procrustes"])

    agent_mouse_labels = np.asarray(B["agent_mouse_labels"])

    gen_dists = None
    if "B5" in B:
        gd = np.asarray(B["B5"].get("gen_dists_proc", [])).ravel()
        gd = gd[np.isfinite(gd)]
        if len(gd) > 2:
            gen_dists = gd

    # ── D1/D2/D3 Mantel (correct signs after notebook fix) ────────────────
    rho_D1 = float(D["D1"]["rho"])
    p_D1   = float(D["D1"]["p_perm"])
    rho_D2 = float(D["D2"]["rho"])
    p_D2   = float(D["D2"]["p_perm"])
    rho_D3 = float(D["D3"]["rho"])
    p_D3   = float(D["D3"]["p_perm"])

    # ── Pair indices (upper triangle of 54×54) ────────────────────────────
    n_agents = len(agent_mouse_labels)
    ii, jj   = np.triu_indices(n_agents, k=1)

    # rep_upper: load from .npy if available, else derive from D_proc_54
    rep_path = os.path.join(act_dir, "rep_upper_dist.npy")
    pt_path  = os.path.join(act_dir, "pair_types.npy")
    if os.path.exists(rep_path):
        rep_upper  = np.load(rep_path)
        pair_types = np.load(pt_path, allow_pickle=True)
    else:
        rep_upper  = D_proc_54[ii, jj]
        pair_types = np.array([
            "within" if agent_mouse_labels[i] == agent_mouse_labels[j] else "between"
            for i, j in zip(ii, jj)
        ])

    # Behavioral distances: from A1_results (same pair ordering)
    beh_path = os.path.join(act_dir, "beh_upper.npy")
    beh_upper = (np.load(beh_path) if os.path.exists(beh_path)
                 else np.asarray(A1["beh_dists_upper"]))

    # Topology distances: from A1_results topo_dists
    topo_upper = np.asarray(A1["topo_dists"])

    # Sensitivity distances: 1 − RSM upper triangle (from A4_results)
    sens_path = os.path.join(act_dir, "sens_upper.npy")
    if os.path.exists(sens_path):
        sens_upper = np.load(sens_path)
    else:
        sens_rsm   = np.asarray(A4["sens_RSM"])
        sens_upper = 1.0 - sens_rsm[ii, jj]

    # Mouse index per pair for Row 3 colouring (−1 = between-mouse)
    mouse_to_idx   = {m: i for i, m in enumerate(MICE)}
    pair_mouse_idx = np.full(len(ii), -1, dtype=int)
    for k, (i, j) in enumerate(zip(ii, jj)):
        if agent_mouse_labels[i] == agent_mouse_labels[j]:
            pair_mouse_idx[k] = mouse_to_idx[agent_mouse_labels[i]]

    # ── Layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=FIGSIZE['emb_rsa'])
    outer = fig.add_gridspec(
        3, 3,
        height_ratios=[1.0, 1.6, 1.6],
        hspace=0.38, wspace=0.28,
        left=0.06, right=0.97, top=0.96, bottom=0.10,
    )

    # Row 0: three simple panels
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[0, 2])

    # Rows 1 + 2: KDE and scatter jointplots (subfigures with nested GridSpec)
    ax_d_m, ax_d_t, ax_d_r = _make_joint_axes(fig, outer[1, 0])
    ax_e_m, ax_e_t, ax_e_r = _make_joint_axes(fig, outer[1, 1])
    ax_f_m, ax_f_t, ax_f_r = _make_joint_axes(fig, outer[1, 2])
    ax_g_m, ax_g_t, ax_g_r = _make_joint_axes(fig, outer[2, 0])
    ax_h_m, ax_h_t, ax_h_r = _make_joint_axes(fig, outer[2, 1])
    ax_i_m, ax_i_t, ax_i_r = _make_joint_axes(fig, outer[2, 2])

    # ── Draw ──────────────────────────────────────────────────────────────
    _panel_a_kde(ax_a, within_proc, between_proc, gen_dists, p_proc)
    _panel_b_mantel(ax_b, rho_D1, p_D1, rho_D2, p_D2, rho_D3, p_D3)
    _panel_c_heatmap(ax_c, D_proc_54, agent_mouse_labels)

    x_labels = [
        "Behavioral distance (cosine)",
        "Topological distance (Jaccard)",
        "Sensitivity distance (1 − RSM)",
    ]
    y_label = "Representational distance (Procrustes)"

    # Axis limits: sensitivity is bounded [0, 1] by construction
    _sens_xlim = (0.0, 1.0)

    # Row 2: KDE contours
    for (ax_m, ax_t, ax_r), x_arr, x_lab, rho, pv, plabel, xlim in [
        ((ax_d_m, ax_d_t, ax_d_r), beh_upper,  x_labels[0], rho_D1, p_D1, "D", None),
        ((ax_e_m, ax_e_t, ax_e_r), topo_upper, x_labels[1], rho_D2, p_D2, "E", None),
        ((ax_f_m, ax_f_t, ax_f_r), sens_upper, x_labels[2], rho_D3, p_D3, "F", _sens_xlim),
    ]:
        _joint_kde(ax_m, ax_t, ax_r, x_arr, rep_upper, pair_types,
                   x_lab, y_label, rho, pv, plabel, x_lim=xlim)

    # Row 3: scatter, mouse-coloured within pairs
    for (ax_m, ax_t, ax_r), x_arr, x_lab, rho, pv, plabel, xlim in [
        ((ax_g_m, ax_g_t, ax_g_r), beh_upper,  x_labels[0], rho_D1, p_D1, "G", None),
        ((ax_h_m, ax_h_t, ax_h_r), topo_upper, x_labels[1], rho_D2, p_D2, "H", None),
        ((ax_i_m, ax_i_t, ax_i_r), sens_upper, x_labels[2], rho_D3, p_D3, "I", _sens_xlim),
    ]:
        _joint_mouse(ax_m, ax_t, ax_r, x_arr, rep_upper, pair_types,
                     pair_mouse_idx, x_lab, y_label, rho, pv, plabel, x_lim=xlim)

    # Footer: mouse identity colour legend (for Row 3)
    mouse_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=MOUSE_COLORS[m], markersize=6 * MARKER_SCALE, label=m)
        for m in MICE
    ]
    fig.legend(handles=mouse_handles, loc="upper center", ncol=9,
               fontsize=FS_LEGEND, frameon=False,
               bbox_to_anchor=(0.5, -0.005), columnspacing=1.4)

    out = os.path.join(figures_dir, "fig_embedding_rsa_v2.pdf")
    save_figure(fig, out)
    return [out]


if __name__ == "__main__":
    class _Store:
        _analysis = os.path.join(_PROJECT, "analysis")
    generate(_Store(), os.path.join(_PROJECT, "figures"))
