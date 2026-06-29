"""
Generate two figures for §2.7 (Gerstner / representational geometry):

  1. fig_gerstner_smallmult.pdf — 3×3 small multiples of PC1 vs PC2 loading
     clouds, one representative agent per mouse.  Direct parallel to Gerstner
     2025 Fig 4F: our clouds are *indistinguishable* across mice, showing the
     footprint is absent.

  2. fig_b4_updated.pdf — Updated B4 KDE (within vs between Procrustes) with
     generalist as a third KDE line.

Run from repo root:
    python scripts/gen_fig_gerstner_smallmult.py
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ────────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "figure_modules"))
from _style import (
    apply_pub_style, save_figure, label_panel, pub_despine,
    MOUSE_COLORS, MICE, GEN_COL, FS_PANEL, FS_TITLE, FS_TICK,
    FS_LEGEND, FS_ANNOT, NEURON_LABELS,
)

OUT_DIR  = os.path.join(REPO, "analysis", "activity_embeddings")
B_PKL    = os.path.join(OUT_DIR, "B_results.pkl")

apply_pub_style()

# ── load data ────────────────────────────────────────────────────────────────
with open(B_PKL, "rb") as f:
    B = pickle.load(f)

loadings_all = B["B4"]["loadings_aligned"]   # list[54] of (14, 6)

# agent ordering: B5 r1–r6 (0–5), B6 r1–r6 (6–11), …, D9 r1–r6 (48–53)
N_REPS = 6
mouse_rep0_idx = {m: i * N_REPS for i, m in enumerate(MICE)}   # first rep each mouse

# B5 results (generalist distances)
gen_dists = None
if "B5" in B:
    raw = B["B5"]
    # try common key names used in the notebook
    for key in ("gen_dists_proc", "gen_proc_dists", "D_gen_vs_spec"):
        if key in raw:
            gen_dists = raw[key]
            break
    if gen_dists is None and isinstance(raw, dict):
        # fallback: any array with 'gen' in key
        for key, val in raw.items():
            if "gen" in key.lower() and isinstance(val, np.ndarray):
                gen_dists = val.ravel()
                break

# B4 RSA stats
within_proc   = B["B4"]["within_proc"]
between_proc  = B["B4"]["between_proc"]
p_proc        = B["B4"]["p_proc"]

# ── neuron type metadata ──────────────────────────────────────────────────────
IDX_SENSORY = np.arange(6)
IDX_INTER   = np.arange(6, 12)
IDX_MOTOR   = np.arange(12, 14)

# Tol muted colours for neuron types (colourblind-safe)
TYPE_COL = {
    "S": "#88CCEE",   # sky blue   — sensory
    "I": "#DDCC77",   # sand       — interneuron
    "M": "#CC6677",   # rose       — motor (stands out)
}
TYPE_LABEL = {"S": "Sensory (0–5)", "I": "Interneuron (6–11)", "M": "Motor (12–13)"}
TYPE_MARKER = {"S": "o", "I": "s", "M": "^"}
TYPE_SIZE   = {"S": 18, "I": 18, "M": 42}    # motors larger so they show

neuron_types = (["S"] * 6 + ["I"] * 6 + ["M"] * 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: 3×3 small multiples — Gerstner Fig 4F parallel
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 3, figsize=(13.5, 13.5), sharex=False, sharey=False)
axes_flat = axes.ravel()

# compute global axis limits from all first-rep agents for comparability
all_pc1, all_pc2 = [], []
for m in MICE:
    L = loadings_all[mouse_rep0_idx[m]]   # (14, 6)
    all_pc1.append(L[:, 0])
    all_pc2.append(L[:, 1])
all_pc1 = np.concatenate(all_pc1)
all_pc2 = np.concatenate(all_pc2)
pad = 0.05
xlim = (all_pc1.min() - pad, all_pc1.max() + pad)
ylim = (all_pc2.min() - pad, all_pc2.max() + pad)

for ax_i, (m, ax) in enumerate(zip(MICE, axes_flat)):
    idx  = mouse_rep0_idx[m]
    L    = loadings_all[idx]   # (14, 6)

    for ntype, nidxs in [("S", IDX_SENSORY), ("I", IDX_INTER), ("M", IDX_MOTOR)]:
        ax.scatter(
            L[nidxs, 0], L[nidxs, 1],
            c=TYPE_COL[ntype],
            marker=TYPE_MARKER[ntype],
            s=TYPE_SIZE[ntype],
            alpha=0.85,
            linewidths=0.3,
            edgecolors="k" if ntype == "M" else "none",
            zorder=3 if ntype == "M" else 2,
        )

    # annotate motor neuron labels explicitly (Spd / Trn)
    for n_i in IDX_MOTOR:
        ax.annotate(
            NEURON_LABELS[n_i],
            (L[n_i, 0], L[n_i, 1]),
            fontsize=7.5, ha="center", va="bottom",
            xytext=(0, 4), textcoords="offset points",
            color="#AA2233",
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(0, lw=0.4, color="grey", zorder=1)
    ax.axvline(0, lw=0.4, color="grey", zorder=1)
    pub_despine(ax)

    # mouse label as title (no in-figure suptitle per style rules)
    ax.set_title(m, fontsize=FS_TITLE, fontweight="bold",
                 color=MOUSE_COLORS[m], pad=3)

    # axis labels on edge panels only
    row, col = divmod(ax_i, 3)
    if row == 2:
        ax.set_xlabel("PC1 loading", fontsize=FS_TICK)
    if col == 0:
        ax.set_ylabel("PC2 loading", fontsize=FS_TICK)

# panel letter
label_panel(axes_flat[0], "A")

# shared legend (bottom-centre)
handles = [
    mpatches.Patch(color=TYPE_COL["S"], label=TYPE_LABEL["S"]),
    mpatches.Patch(color=TYPE_COL["I"], label=TYPE_LABEL["I"]),
    mpatches.Patch(color=TYPE_COL["M"], label=TYPE_LABEL["M"]),
]
fig.legend(handles=handles, loc="lower center", ncol=3,
           fontsize=FS_LEGEND, frameon=False,
           bbox_to_anchor=(0.5, -0.01))

fig.tight_layout(rect=[0, 0.03, 1, 1])
save_figure(fig, os.path.join(OUT_DIR, "fig_embedding_clouds.pdf"))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: B4 KDE updated — within / between / generalist
# ═══════════════════════════════════════════════════════════════════════════════

from scipy.stats import gaussian_kde

fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))

def plot_kde(ax, data, color, label, ls="-"):
    xs = np.linspace(data.min() - 0.2, data.max() + 0.2, 400)
    kde = gaussian_kde(data, bw_method=0.25)
    ax.plot(xs, kde(xs), color=color, lw=1.5, ls=ls, label=label)
    ax.fill_between(xs, kde(xs), alpha=0.12, color=color)

plot_kde(ax2, within_proc,  "#CC6677", f"Within-mouse (n={len(within_proc)})", "-")
plot_kde(ax2, between_proc, "#88CCEE", f"Between-mouse (n={len(between_proc)})", "-")

# Generalist: n=15 is too small for a reliable KDE — show mean ± SD as a
# vertical line + shaded band instead.
if gen_dists is not None:
    gen_flat = np.asarray(gen_dists).ravel()
    gen_flat = gen_flat[np.isfinite(gen_flat)]
    if len(gen_flat) > 2:
        gm, gs = gen_flat.mean(), gen_flat.std()
        ax2.axvline(gm, color=GEN_COL, lw=1.4, ls="--",
                    label=f"Generalist mean ± SD (n={len(gen_flat)})")
        ax2.axvspan(gm - gs, gm + gs, color=GEN_COL, alpha=0.10)

# MW p annotation
ax2.text(0.97, 0.92, f"MW p = {p_proc:.3f}",
         transform=ax2.transAxes, ha="right", va="top",
         fontsize=FS_ANNOT, color="0.35")

ax2.set_xlabel("Procrustes distance (activity embeddings)", fontsize=FS_TICK)
ax2.set_ylabel("Density", fontsize=FS_TICK)
ax2.legend(frameon=False, fontsize=FS_LEGEND)
pub_despine(ax2)
label_panel(ax2, "A")

fig2.tight_layout()
save_figure(fig2, os.path.join(OUT_DIR, "fig_b4_updated.pdf"))

print("Done.")
