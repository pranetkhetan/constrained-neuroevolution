"""
Method schematic: topology–behaviour correlation pipeline (§2.3).

Three panels (2:1 aspect ratio):
  A: STEP A — Jaccard distance 𝒥 between two 14×14 binary weight matrices
              (matrices stacked vertically; title "Topology / Network Weights")
  B: STEP B — Cosine distance 𝒟 between rows of the 54×9 behavioural profile matrix
              (arc angle labelled 𝒟_ij directly)
  C: STEP C — Spearman(𝒥, 𝒟) ≈ 0: circular cloud 0.2–0.75, axes 0–1,
              horizontal trend line through the cloud

All content is synthetic (seed-fixed random) — pure schematic.

API
---
create_figure()                    -> fig
generate(figures_dir)              -> [paths]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Arc
from matplotlib.colors import LinearSegmentedColormap, to_rgb

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figure_modules._style import (
    apply_pub_style, save_figure, pub_despine,
    FS_BASE, FS_ANNOT, FS_LABEL,
)

_BLUE  = "#4472C4"   # agent i
_GREEN = "#70AD47"   # agent j


def _binary_cmap(color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("_bc", ["white", color])


# ─────────────────────────────────────────────────────────────────────────────
# Panel A — Two 14×14 matrices stacked VERTICALLY + 𝒥 arrow
# ─────────────────────────────────────────────────────────────────────────────

def _panel_a(fig: plt.Figure, subplot_spec):
    """Returns ax_title so callers can attach a panel letter via label_panel()."""
    gs = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=subplot_spec,
        height_ratios=[1, 1],
        width_ratios=[1, 1],
        hspace=0.08, wspace=0.08,
    )

    # # Step title
    # ax_title = fig.add_subplot(gs[0, 0])
    # ax_title.axis("off")

    # Subtitle
    # ax_sub = fig.add_subplot(gs[1, :])
    # ax_sub.axis("off")
    # ax_sub.text(
    #     0.5, 0.5, "Topology / Network Weights",
    #     ha="center", va="center", fontsize=FS_ANNOT,
    #     color="#666666", fontstyle="italic",
    #     transform=ax_sub.transAxes,
    # )

    # Matrix i (top)
    ax_i = fig.add_subplot(gs[0, 0])
    ax_i.axis("off")
    # Arrow row
    # ax_sep = fig.add_subplot(gs[3, :])
    # Matrix j (bottom)
    ax_j = fig.add_subplot(gs[1, 1])
    ax_j.axis("off")
    # Bottom padding
    # ax_bot = fig.add_subplot(gs[5, :])

    rng = np.random.default_rng(1)
    n = 14
    mat_i = rng.integers(0, 2, (n, n))
    mat_j = rng.integers(0, 2, (n, n))

    ax_i.imshow(mat_i, cmap=_binary_cmap(_BLUE),  vmin=0, vmax=1,
                aspect="equal", interpolation="nearest")
    ax_i.annotate(xy=(0.5, -0.05), text=r"Agent $i$  Network Weights", xycoords="axes fraction",
                    fontsize=FS_ANNOT, color=_BLUE, ha="center", va="top")
    ax_j.imshow(mat_j, cmap=_binary_cmap(_GREEN), vmin=0, vmax=1,
                aspect="equal", interpolation="nearest")
    ax_j.annotate(xy=(0.5, -0.05), text=r"Agent $j$  Network Weights", xycoords="axes fraction",
                    fontsize=FS_ANNOT, color=_GREEN, ha="center", va="top")

    for ax, color, lbl in [
        (ax_i, _BLUE,  r"Agent $i$  (14×14)"),
        (ax_j, _GREEN, r"Agent $j$  (14×14)"),
    ]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(lbl, color=color, fontsize=FS_ANNOT, labelpad=4)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color("#888888")

    # Vertical double-headed arrow + 𝒥 label
    # # ax_sep.axis("off")
    # ax_i.annotate(
    #     "", xy=(0.5, 0.08), xytext=(0.5, 0.92),
    #     arrowprops=dict(arrowstyle="<->", color="#555555",
    #                     lw=1.5, mutation_scale=12),
    #     xycoords="axes fraction", textcoords="axes fraction",
    # )
    ax_i.text(
        1.0, 0.1,  (r"$\mathcal{J}_{ij}$" + "\n" + r"$\leftrightarrow$"),
        rotation=-30, ha="left", va="center", fontsize=FS_BASE * 2.0,
        transform=ax_i.transAxes, color="#222222",
    )

    return ax_i


# ─────────────────────────────────────────────────────────────────────────────
# Panel B — 54×9 behavioural profile matrix + cosine angle vectors
# ─────────────────────────────────────────────────────────────────────────────

def _panel_b(fig: plt.Figure, subplot_spec) -> None:
    gs = gridspec.GridSpecFromSubplotSpec(
        2,2, subplot_spec=subplot_spec,
        height_ratios=[1, 0.4],
        width_ratios=[1, 0.4],
        hspace=0.10, wspace=0,
    )

    # ax_title = fig.add_subplot(gs[0])
    # ax_title.axis("off")

    ax_data  = fig.add_subplot(gs[0,0])
    # ax_label = fig.add_subplot(gs[2])
    ax_vec   = fig.add_subplot(gs[1,1])

    # --- 54×9 behavioural profile matrix ---
    rows, cols = 54, 9
    row_i, row_j = 18, 38

    rng = np.random.default_rng(1)
    mask = rng.uniform(0.7, 0.9, (rows, cols, 1))
    # repeat the same mask across the 3 RGB channels to get a greyscale image
    cmat = np.repeat(mask, 3, axis=2) 
    cmat[row_i, :] = to_rgb(_BLUE)
    cmat[row_j, :] = to_rgb(_GREEN)

    ax_data.imshow(cmat, aspect="auto", interpolation=None,)

    ax_data.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax_data.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax_data.grid(which="minor", color="white", linewidth=0.35)
    ax_data.tick_params(which="minor", length=0)
    ax_data.set_xticks([])
    ax_data.set_yticks([])

    ax_data.text(-1.0, row_i, r"$i$", ha="right", va="center",
                 color=_BLUE,  fontsize=FS_BASE, fontweight="bold")
    ax_data.text(-1.0, row_j, r"$j$", ha="right", va="center",
                 color=_GREEN, fontsize=FS_BASE, fontweight="bold")
    ax_data.axis("off")
    ax_data.text(0.5, -0.05, r"Cross-mouse Fitness Profiles",
                  ha="center", va="center", fontsize=FS_ANNOT,
                  color="#444444", transform=ax_data.transAxes)

    # ax_label.axis("off")
    # ax_label.text(0.5, 0.5, r"Cosine distance: $\mathcal{D}_{ij}$",
    #               ha="center", va="center", fontsize=FS_ANNOT,
    #               color="#444444", transform=ax_label.transAxes)

    # --- Vector angle diagram; the arc IS 𝒟_ij ---
    ax_vec.set_xlim(0, 1)
    ax_vec.set_ylim(0, 1)
    ax_vec.set_aspect("equal")
    ax_vec.axis("off")

    origin = np.array([0.0, 0.0])
    angle_vi, angle_vj = np.radians(75), np.radians(25) # degrees to radians
    amp_vi, amp_vj = 1, 1 # vector amplitudes 
    vi = np.array([amp_vi * np.cos(angle_vi),
                   amp_vi * np.sin(angle_vi)])   # agent i — steeper
    vj = np.array([amp_vj * np.cos(angle_vj),
                   amp_vj * np.sin(angle_vj)])   # agent j — shallower

    ax_vec.annotate("", xy=origin + vi, xytext=origin,
                    arrowprops=dict(arrowstyle="->", color=_BLUE,
                                   lw=2.0, mutation_scale=14))
    ax_vec.annotate("", xy=origin + vj, xytext=origin,
                    arrowprops=dict(arrowstyle="->", color=_GREEN,
                                   lw=2.0, mutation_scale=14))

    arc = Arc(xy=origin, width=0.4, height=0.4,
              theta1=np.degrees(angle_vj), theta2=np.degrees(angle_vi),
                color="#555555", lw=1.5, zorder=1)
    ax_vec.add_patch(arc)

    # Arc label IS 𝒟_ij (no separate θ)
    mid_rad = ((angle_vi + angle_vj) / 2)
    print(mid_rad)
    ax_vec.text(
        0.5 * np.cos(mid_rad), 0.5 * np.sin(mid_rad),
        r"$\mathcal{D}_{ij}$",
        ha="center", va="center",
        fontsize=FS_BASE * 1.25, color="#333333",
    )
    ax_vec.text(0.5, -0.1, r"Cosine distance: $\mathcal{D}_{ij}$",
                  ha="center", va="center", fontsize=FS_ANNOT,
                  color="#444444", transform=ax_vec.transAxes)


# ─────────────────────────────────────────────────────────────────────────────
# Panel C — 𝒥 vs 𝒟 scatter: circular cloud, axes 0–1, horizontal trend line
# ─────────────────────────────────────────────────────────────────────────────

def _panel_c(fig: plt.Figure, subplot_spec) -> None:
    gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=subplot_spec,
        height_ratios=[1, 11],
        hspace=0.12,
    )

    ax_title = fig.add_subplot(gs[0])
    ax_title.axis("off")

    ax_sc = fig.add_subplot(gs[1])

    np.random.seed(7)
    n = 100

    # Step 1: latent shared factor
    z = np.random.randn(n)

    # Step 2: weakly correlated raw variables
    x_raw = z + 0.01 * np.random.randn(n)
    y_raw = 0.05 * z - np.random.randn(n)

    # Step 3: squash to [0.2, 0.8] using sigmoid
    def squash(v):
        s = 1 / (1 + np.exp(-v))        # sigmoid → [0,1]
        return 0.2 + 0.6 * s            # rescale to [0.2, 0.8]

    J = squash(x_raw)
    D = squash(y_raw)

    print("Correlation:", np.corrcoef(J, D)[0, 1])
    print("Range J:", (J.min(), J.max()))
    print("Range D:", (D.min(), D.max()))

    ax_sc.scatter(J, D, color="#777777", alpha=0.65, s=18, linewidths=0, zorder=3)

    # Annotate one representative point as (𝒥_ij, 𝒟_ij)
    k = 12   # seed-stable index that sits clearly from the midline
    ax_sc.scatter([J[k]], [D[k]], color="#333333", s=28, linewidths=0, zorder=5)
    ax_sc.annotate(
        r"$(\mathcal{J}_{ij},\, \mathcal{D}_{ij})$",
        xy=(J[k], D[k]),
        xytext=(J[k] + 0.18, D[k] + 0.12),
        fontsize=FS_ANNOT,
        color="#222222",
        arrowprops=dict(arrowstyle="-|>", color="#444444",
                        lw=0.9, mutation_scale=8),
        va="center",
    )

    coeffs = np.polyfit(J, D, 1)
    example_spearman = np.corrcoef(J, D)[0, 1]
    ax_sc.plot(
        [np.min(J), np.max(J)],
        [np.polyval(coeffs, np.min(J)), np.polyval(coeffs, np.max(J))],
        color="#333333",
        lw=1.5,
        zorder=2
    )
    ax_sc.set_xlim(0.2,0.8)
    ax_sc.set_ylim(0.2,0.8)
    ax_sc.set_xticks([])
    ax_sc.set_yticks([])

    ax_sc.set_xlabel(r"$\mathcal{J}$  (Jaccard distance)", fontsize=FS_ANNOT)
    ax_sc.set_ylabel(r"$\mathcal{D}$  (Cosine distance)",  fontsize=FS_ANNOT)
    ax_sc.text(
        0.5, 1.0, r"Spearman$(\mathcal{J},\,\mathcal{D})\approx $" + "{:.2f}".format(example_spearman),
        ha="center", va="top", fontsize=FS_ANNOT, color="#333333",
        transform=ax_sc.transAxes,
    )

    ax_sc.tick_params(labelsize=FS_ANNOT)
    # pub_despine(ax_sc)
    # despine manually to avoid cutting off the top annotation
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["bottom"].set_visible(False)
    ax_sc.spines["left"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def draw_into_spec(fig, subplot_spec):
    """Embed the 3-panel schematic into an existing GridSpec slot.

    Draws Panels A/B/C into *subplot_spec* without creating a new figure or
    changing rcParams.  Returns the top-left anchor axis so the caller can
    attach a panel letter with label_panel().
    """
    gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=subplot_spec,
        width_ratios=[1,1,1],
        wspace=0.15,
    )
    anchor = _panel_a(fig, gs[0])
    _panel_b(fig, gs[1])
    _panel_c(fig, gs[2])
    return anchor


def create_figure() -> plt.Figure:
    apply_pub_style()

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[1, 1, 0.8],
        wspace=0.15,
        left=0.04, right=0.97, top=0.94, bottom=0.11,
    )

    _panel_a(fig, gs[0])
    _panel_b(fig, gs[1])
    _panel_c(fig, gs[2])

    return fig


def generate(figures_dir: str) -> list[str]:
    fig = create_figure()
    out = os.path.join(figures_dir, "fig_method_topology_schematic.pdf")
    return save_figure(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _figures_dir = os.path.join(_PROJECT, "figures")
    paths = generate(_figures_dir)
    print("Done:", paths)
