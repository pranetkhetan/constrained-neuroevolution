"""
Supplementary Figure S4 (fig:supp_trajectories) — Activation trajectory analysis.
Output: fig_s4_trajectories.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from ._style import apply_pub_style, MICE, MOUSE_COLORS, FS_PANEL, FS_SUPTITLE, FIGSIZE, LW_SCALE, MARKER_SCALE


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    dyn = store.dynamics_results()
    traj = dyn['trajectory']

    pca_coords = np.array(traj['pca_coords'])
    mouse_ids = traj['mouse_ids']
    rsm = np.array(traj['rsm'])
    within_sims = np.array(traj['within_sims'])
    between_sims = np.array(traj['between_sims'])
    p_val = traj['mannwhitney_p']
    n = len(mouse_ids)

    # PCA variance ratio may not be stored; use placeholder if absent
    var_ratio = traj.get('pca_var_ratio', [0.0, 0.0])

    fig = plt.figure(figsize=FIGSIZE['supp_s4'])
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.25, width_ratios=[1, 1.5, 1])
    # width ratios = [1.5, 1, 1]

    # Panel A: PCA scatter
    ax1 = fig.add_subplot(gs[0])
    for m in MICE:
        idxs = [i for i, mid in enumerate(mouse_ids) if mid == m]
        if not idxs:
            continue
        coords = pca_coords[idxs]
        ax1.scatter(coords[:, 0], coords[:, 1],
                    color=MOUSE_COLORS[m], label=m, s=60 * MARKER_SCALE, zorder=3,
                    edgecolors='white', linewidths=0.5 * LW_SCALE)
        if len(coords) > 1:
            cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
            ax1.annotate(m, (cx, cy), fontsize=FS_PANEL, ha='center', va='center',
                         color=MOUSE_COLORS[m], fontweight='bold')

    ax1.set_xlabel(f'PC1 ({var_ratio[0] * 100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({var_ratio[1] * 100:.1f}%)')
    ax1.set_title('A  Activation PCA\n(54 agents, same inputs)', fontweight='bold')
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    sns.despine(ax=ax1, offset=5, trim=True)

    # Panel B: RSM heatmap
    ax2 = fig.add_subplot(gs[1])
    order = sorted(range(n), key=lambda i: (mouse_ids[i], i))
    rsm_ordered = rsm[np.ix_(order, order)]
    im = ax2.imshow(rsm_ordered, cmap='RdYlGn', vmin=-0.2, vmax=1.0, aspect='auto')
    plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.1, label='Cosine similarity')

    boundaries = []
    prev_m = mouse_ids[order[0]]
    for k, idx in enumerate(order):
        if mouse_ids[idx] != prev_m:
            boundaries.append(k - 0.5)
            prev_m = mouse_ids[idx]
    for b in boundaries:
        ax2.axhline(b, color='black', lw=0.8 * LW_SCALE)
        ax2.axvline(b, color='black', lw=0.8 * LW_SCALE)

    ax2.set_title('B  Representational\nSimilarity Matrix', fontweight='bold')
    ax2.set_xlabel('Agent (ordered by mouse)')
    ax2.set_ylabel('Agent (ordered by mouse)')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Panel C: within vs between violin
    ax3 = fig.add_subplot(gs[2])
    vp = ax3.violinplot([within_sims, between_sims], positions=[0, 1],
                        showmedians=True, showextrema=False)
    vp['bodies'][0].set_facecolor('#2196F3')
    vp['bodies'][1].set_facecolor('#FF9800')
    for body in vp['bodies']:
        body.set_alpha(0.7)

    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Within\nmouse', 'Between\nmouse'])
    ax3.set_ylabel('Cosine similarity', labelpad=2)
    ax3.set_ylim(-1,1)
    ax3.set_title(f'C  Similarity comparison\np={p_val:.4f}', fontweight='bold')
    sns.despine(ax=ax3,offset=2, trim=True)

    if p_val < 0.05:
        y_bar = max(np.max(within_sims), np.max(between_sims)) + 0.05
        ax3.plot([0, 1], [y_bar, y_bar], 'k-', lw=1 * LW_SCALE)
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else '*')
        ax3.text(0.5, y_bar + 0.01, sig, ha='center', fontsize=FS_PANEL)

    plt.suptitle('Activation Trajectory Analysis: Same inputs, different internal dynamics',
                 fontsize=FS_SUPTITLE, y=1.02)
    out = os.path.join(figures_dir, 'fig_s4_trajectories.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return [out]
