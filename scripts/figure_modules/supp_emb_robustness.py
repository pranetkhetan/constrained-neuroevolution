"""
paper_v2 §2.7 Supplementary — Representational degeneracy is robust across 6 methods.

Single panel: within-mouse vs between-mouse mean distance for 6 independent metrics,
all normalised to their between-mouse mean so they appear on a common scale.
All MW p-values > 0.05 (null result).

Sources:
  Proc-6PC (distance):  B_results.pkl B4 — p=0.2596 (corrected scale→distance)
  RSA, CKA, Cov-Frob, Proc-raw, Log-Euc SPD: colab_20b_sanity.ipynb Cell 13 outputs
  (sealed analysis — values verified and frozen here)

Output: figures/fig_supp_emb_robustness.pdf
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figure_modules._style import (
    apply_pub_style, save_figure, label_panel, pub_despine,
    FIGSIZE,
    FS_PANEL, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    LW_SCALE, MARKER_SCALE,
)

# Six-metric table: (display_name, within_mean, between_mean, mw_p).
# Recomputed by scripts/embedding_robustness.py -> embedding_robustness.pkl.
# The hardcoded list below is a fallback used only if that pkl is absent; the
# live values are loaded in generate(). (Proc-6PC distance = scale→distance of B4.)
_METHODS_FALLBACK = [
    ("Proc-6PC\n(distance)",   1.9738, 1.9877, 0.2596),
    ("Proc-raw\n(no PCA)",     1.0762, 1.0698, 0.6254),
    ("RSA\n(RDMs)",            0.8685, 0.8557, 0.2737),
    ("Linear\nCKA",            0.7970, 0.7889, 0.6245),
    ("Cov-Frob",               0.7413, 0.7493, 0.7449),
    ("Log-Euclidean\nSPD",    14.5340, 14.5945, 0.5283),
]

# Map embedding_robustness.pkl keys -> two-line display labels (panel order).
_ROB_DISPLAY = [
    ("Proc-6PC",   "Proc-6PC\n(distance)"),
    ("Proc-raw",   "Proc-raw\n(no PCA)"),
    ("RSA",        "RSA\n(RDMs)"),
    ("CKA",        "Linear\nCKA"),
    ("Cov-Frob",   "Cov-Frob"),
    ("LogEuc-SPD", "Log-Euclidean\nSPD"),
]


def _load_methods():
    """Live within/between/p from embedding_robustness.pkl, else the frozen fallback."""
    pkl = os.path.join(_PROJECT, "analysis", "activity_embeddings",
                       "embedding_robustness.pkl")
    if not os.path.exists(pkl):
        return _METHODS_FALLBACK
    with open(pkl, "rb") as f:
        rob = pickle.load(f)
    out = []
    for key, label in _ROB_DISPLAY:
        r = rob[key]
        out.append((label, r["within_mean"], r["between_mean"], r["mw_p"]))
    return out


def _sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()

    methods = _load_methods()

    fig, ax = plt.subplots(figsize=FIGSIZE['supp_emb_rob'])

    n = len(methods)
    ys = np.arange(n)

    for i, (name, w_mean, b_mean, p) in enumerate(methods):
        # Normalise: between-mean = 1.0 reference
        w_norm = w_mean / b_mean
        b_norm = 1.0
        # within dot (red), between dot (blue), connected by line
        ax.plot([w_norm, b_norm], [i, i], color="0.70", lw=1.2 * LW_SCALE, zorder=1)
        ax.scatter(w_norm, i, color="#CC6677", s=40 * MARKER_SCALE, zorder=3, clip_on=False)
        ax.scatter(b_norm, i, color="#88CCEE", s=40 * MARKER_SCALE, zorder=3, clip_on=False)
        # p-value annotation to the right
        ax.text(1.02, i, _sig_label(p), va="center", ha="left",
                fontsize=FS_ANNOT, color="0.35")

    ax.axvline(1.0, color="#88CCEE", lw=0.9 * LW_SCALE, ls="--", alpha=0.6,
               label="Between-mouse mean (=1)")
    ax.set_yticks(ys)
    ax.set_yticklabels([m[0] for m in methods], fontsize=FS_TICK)
    ax.set_xlabel("Within-mouse mean distance\n(normalised to between-mouse mean)",
                  fontsize=FS_TICK)

    # legend
    handles = [
        mpatches.Patch(color="#CC6677", label="Within-mouse mean"),
        mpatches.Patch(color="#88CCEE", label="Between-mouse mean (= 1.0)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=FS_LEGEND,
              loc="lower right")

    ax.set_xlim(0.96, 1.06)
    ax.invert_yaxis()
    pub_despine(ax)
    label_panel(ax, "A")

    out = os.path.join(figures_dir, "fig_supp_emb_robustness.pdf")
    save_figure(fig, out)
    return [out]


if __name__ == "__main__":
    class _Store:
        _analysis = os.path.join(_PROJECT, "analysis")
    generate(_Store(), os.path.join(_PROJECT, "figures"))
