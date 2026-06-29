"""
Supplementary: Structural k-means clustering visualization (§2.4).

4-panel 2×2 figure for Supplementary S13.

Panel A: PCA of agents in topology feature space, colored by k-means cluster (k=5).
         Shows "what do the 5 structural groups look like?"
Panel B: Same PCA projection, colored by mouse identity (9 mice, Tol palette).
         Shows "mouse identity does not align with structural clusters."
Panel C: PCA of magnitude feature space, colored by mouse identity.
         Confirms the null on a different structural axis.
Panel D: Agents ranked by own-mouse fitness, colored by topology k-means cluster.
         Shows "structurally similar agents do not perform similarly" — the fitness-
         quintile companion to the mouse-identity MI analysis (body text).

Requires A3_results.pkl updated by scripts/update_A3_mouse_identity.py.

Output: supp_mi_clustering.pdf
"""

import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figure_modules._style import (
    apply_pub_style, pub_despine, save_figure, label_panel,
    FIGSIZE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    MICE, MOUSE_COLORS,
    LW_SCALE, MARKER_SCALE,
)

# tab10 first 5 — distinct from Tol mouse palette
CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _load_pkl(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        class _U(pickle.Unpickler):
            def find_class(self, mod, name):
                if mod.startswith("cupy"):
                    mod = mod.replace("cupy._core.core", "numpy").replace("cupy", "numpy")
                return super().find_class(mod, name)
        with open(path, "rb") as f:
            return _U(f).load()


def _cluster_legend(ax: plt.Axes) -> None:
    handles = [
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=CLUSTER_COLORS[k],
                      markersize=7 * MARKER_SCALE, label=f"Cluster {k+1}")
        for k in range(5)
    ]
    ax.legend(handles=handles, fontsize=FS_ANNOT, frameon=False,
              ncol=3, handlelength=0.6, handletextpad=0.3)


def _mouse_legend(ax: plt.Axes) -> None:
    handles = [
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=MOUSE_COLORS[m],
                      markersize=7 * MARKER_SCALE, label=m)
        for m in MICE
    ]
    ax.legend(handles=handles, fontsize=FS_ANNOT, frameon=False,
              ncol=3, handlelength=0.6, handletextpad=0.3)


def _draw_kde_contours(
    ax: plt.Axes,
    coords: np.ndarray,
    groups: list,
    level: float = 0.5,
) -> None:
    """Draw one KDE iso-density contour per group at the `level` probability mass."""
    x0, x1 = coords[:, 0].min() - 0.5, coords[:, 0].max() + 0.5
    y0, y1 = coords[:, 1].min() - 0.5, coords[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x0, x1, 150), np.linspace(y0, y1, 150))
    grid   = np.vstack([xx.ravel(), yy.ravel()])
    for mask, color in groups:
        pts = coords[mask]
        if len(pts) < 3:
            continue
        kde    = gaussian_kde(pts.T, bw_method="scott")
        z      = kde(grid).reshape(xx.shape)
        thresh = np.percentile(kde(pts.T), (1.0 - level) * 100)
        ax.contour(xx, yy, z, levels=[thresh], colors=[color],
                   alpha=0.7, linewidths=1.5 * LW_SCALE, zorder=2)


def _scatter_by_cluster(
    ax: plt.Axes,
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    var_exp: np.ndarray,
    panel_letter: str,
    title: str,
) -> None:
    for k in range(5):
        mask = cluster_labels == k
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=CLUSTER_COLORS[k], s=70 * MARKER_SCALE, zorder=3,
                   linewidths=0.4 * LW_SCALE, edgecolors="white", alpha=0.90)
    _draw_kde_contours(ax, coords, [(cluster_labels == k, CLUSTER_COLORS[k]) for k in range(5)])
    _cluster_legend(ax)
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.0f}% var)", fontsize=FS_LABEL)
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.0f}% var)", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_LABEL, pad=4)
    ax.tick_params(labelsize=FS_TICK)
    pub_despine(ax)
    label_panel(ax, panel_letter)


def _scatter_by_mouse(
    ax: plt.Axes,
    coords: np.ndarray,
    mouse_labels: np.ndarray,
    var_exp: np.ndarray,
    panel_letter: str,
    title: str,
) -> None:
    for m in MICE:
        mask = mouse_labels == m
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=MOUSE_COLORS[m], s=70 * MARKER_SCALE, zorder=3,
                   linewidths=0.4 * LW_SCALE, edgecolors="white", alpha=0.90)
    _draw_kde_contours(ax, coords, [(mouse_labels == m, MOUSE_COLORS[m]) for m in MICE])
    _mouse_legend(ax)
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.0f}% var)", fontsize=FS_LABEL)
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.0f}% var)", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_LABEL, pad=4)
    ax.tick_params(labelsize=FS_TICK)
    pub_despine(ax)
    label_panel(ax, panel_letter)


def _fitness_by_cluster(
    ax: plt.Axes,
    fits: np.ndarray,
    cluster_labels: np.ndarray,
    panel_letter: str,
    title: str,
) -> None:
    """Agents ranked by own-mouse fitness, colored by topology cluster.

    If structural clusters tracked performance, the colors would grade smoothly
    from left (best) to right (worst). Mixed colors across the rank axis confirm
    the fitness-quintile null: structurally similar agents do not perform similarly.
    """
    order = np.argsort(fits)   # ascending: lowest (best) fitness first
    for rank, idx in enumerate(order):
        ax.scatter(rank, fits[idx], color=CLUSTER_COLORS[cluster_labels[idx]],
                   s=70 * MARKER_SCALE, zorder=3, linewidths=0.3 * LW_SCALE, edgecolors="white", alpha=0.90)

    _cluster_legend(ax)
    ax.set_xlabel("Agent rank (sorted by own-mouse fitness)", fontsize=FS_LABEL)
    ax.set_ylabel("Own-mouse fitness (lower = better)", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_LABEL, pad=4)
    ax.tick_params(labelsize=FS_TICK)
    pub_despine(ax)
    label_panel(ax, panel_letter)


def create_figure(store: dict) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Build the 4-panel 2×2 supplementary clustering figure."""
    analysis_dir = os.path.join(_PROJECT, "analysis", "degeneracy_analyses")
    A3 = _load_pkl(os.path.join(analysis_dir, "A3_results.pkl"))
    A1 = _load_pkl(os.path.join(analysis_dir, "A1_results.pkl"))

    if "topo_vecs" not in A3:
        raise RuntimeError(
            "A3_results.pkl is missing feature vectors. "
            "Run scripts/update_A3_mouse_identity.py first."
        )

    topo_vecs    = np.array(A3["topo_vecs"])
    mag_vecs     = np.array(A3["mag_vecs"])
    mouse_labels = np.array(A3["agent_mouse_labels"])
    topo_km      = np.array(A3["Topology"]["cluster_labels"])

    # Own-mouse fitness for each of the 54 agents from the 54×9 fitness matrix
    fitness_matrix = np.array(A1["fitness_matrix"])
    fits = np.array([fitness_matrix[i, MICE.index(mouse_labels[i])]
                     for i in range(len(mouse_labels))])

    # PCA projections
    pca_topo = PCA(n_components=2, random_state=42)
    topo_2d  = pca_topo.fit_transform(topo_vecs)
    topo_var = pca_topo.explained_variance_ratio_

    pca_mag = PCA(n_components=2, random_state=42)
    mag_2d  = pca_mag.fit_transform(mag_vecs)
    mag_var = pca_mag.explained_variance_ratio_

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE['supp_mi_clust'])
    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.08,
                        hspace=0.38, wspace=0.30)

    _scatter_by_cluster(axes[0, 0], topo_2d, topo_km, topo_var,
                        "A", "Topology space — k-means clusters ($k=5$)")

    _scatter_by_mouse(axes[0, 1], topo_2d, mouse_labels, topo_var,
                      "B", "Topology space — mouse identity")

    _scatter_by_mouse(axes[1, 0], mag_2d, mouse_labels, mag_var,
                      "C", "Magnitude space — mouse identity")

    _fitness_by_cluster(axes[1, 1], fits, topo_km,
                        "D", "Own-mouse fitness rank — topology clusters")

    return fig, {"A": axes[0, 0], "B": axes[0, 1],
                 "C": axes[1, 0], "D": axes[1, 1]}


def generate(store: dict, figures_dir: str) -> list[str]:
    apply_pub_style()
    fig, _ = create_figure(store)
    out = os.path.join(figures_dir, "supp_mi_clustering.pdf")
    return save_figure(fig, out)
