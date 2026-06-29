"""
sim_setup_figure.py – Simulation setup overview figure.

Creates a publication-quality 4-panel figure:
  A (top-left)    : Hierarchical Binary Tree Maze with representative mouse trajectories
  B (top-right)   : Agent neural network architecture
  C (bottom-left) : Truncation selection mechanism (flowchart)
  D (bottom-right): Behavioural fitness metric illustrations

Usage:
    python sim_setup_figure.py
    python sim_setup_figure.py --output figures/fig1_setup.png
"""
import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from dataclasses import make_dataclass

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config import load_config
from utils.maze import create_maze
from core.agent import Agent
from scripts.figure_modules._style import LW_SCALE, MARKER_SCALE

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.0,
})

PANEL_LABEL_KW = dict(fontsize=13, fontweight='bold', ha='left', va='top',
                      transform=None)  # set per-ax below

# Colour palette
TRAJ_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
EXC_COLOR   = '#cc2222'
INH_COLOR   = '#2244cc'
MAZE_WALL   = '#2a2a2a'

# Network layout constants (matching visualize.py)
X_SPREAD   = 3.0
Y_SPREAD_S = 5.6
Y_SPREAD_M = 2.0
Y_SPREAD_I = 5.6


# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Rosenberg Traj dataclass (must match the pickled format exactly)
_Traj = make_dataclass('Traj', ['fr', 'ce', 'ke', 'no', 're'])
_Traj.__module__ = '__main__'   # ensures unpickling works from any call site


class _TrajUnpickler(pickle.Unpickler):
    """Redirect any 'Traj' class from any module to our local definition."""
    def find_class(self, module, name):
        if name == 'Traj':
            return _Traj
        return super().find_class(module, name)


def _ke_segments(ke_bout):
    """
    Split a raw-keypoint array into finite (NaN-free) contiguous segments.

    Parameters
    ----------
    ke_bout : (N, 2) array in normalised [0,1] coordinates.

    Returns
    -------
    list of (M, 2) float64 arrays in maze coordinates, each fully finite.
    Raises ValueError if ke_bout has unexpected shape.
    """
    if ke_bout.ndim == 3 and ke_bout.shape[1] == 1:
        # Shape (N,1,2) – squeeze the middle dim
        ke_bout = ke_bout[:, 0, :]
    if ke_bout.ndim != 2 or ke_bout.shape[1] != 2:
        raise ValueError(
            f"Expected ke shape (N,2) or (N,1,2), got {ke_bout.shape}"
        )

    # Scale to maze coordinates:  maze = -0.5 + 15.0 * normalised
    scaled = -0.5 + 15.0 * ke_bout.astype(np.float64)

    # Find row indices where both x and y are finite
    finite = np.isfinite(scaled).all(axis=1)
    segments = []
    # Walk consecutive finite runs
    in_seg = False
    seg_start = 0
    for i, ok in enumerate(finite):
        if ok and not in_seg:
            seg_start = i
            in_seg = True
        elif not ok and in_seg:
            seg = scaled[seg_start:i]
            if len(seg) >= 2:
                segments.append(seg)
            in_seg = False
    if in_seg:
        seg = scaled[seg_start:]
        if len(seg) >= 2:
            segments.append(seg)
    return segments


def load_mouse_trajectories(mouse_id='B5', n_bouts=5, min_frames=50):
    """
    Load real mouse trajectories (raw keypoints) from a pre-processed .pkl file.

    Uses tr.ke (raw snout keypoints) rather than tr.ce (cell-snapped) because
    ke gives smooth, continuous paths that look good in a figure.
    NaN gaps in ke are split into separate segments.

    Parameters
    ----------
    mouse_id  : str  – e.g. 'B5', 'D3'
    n_bouts   : int  – maximum number of bouts to load
    min_frames: int  – skip bouts shorter than this many finite frames

    Returns
    -------
    list of lists-of-segments, one entry per bout.
    Each entry is a list of (M, 2) float64 arrays in maze coordinates.

    Raises
    ------
    FileNotFoundError  if the .pkl does not exist.
    ValueError         if tr.ke is None or empty.
    RuntimeError       if no bouts pass the min_frames threshold.
    """
    path = f'data/mouse_{mouse_id}.pkl'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mouse data not found: {path}")

    with open(path, 'rb') as f:
        tr = _TrajUnpickler(f).load()

    if tr.ke is None or len(tr.ke) == 0:
        raise ValueError(f"tr.ke is empty for mouse {mouse_id} — cannot extract trajectories")

    print(f"  mouse_{mouse_id}: {len(tr.ke)} bouts available in tr.ke")

    bouts_out = []
    skipped = 0
    for ke in tr.ke:
        if len(bouts_out) >= n_bouts:
            break
        segs = _ke_segments(ke)
        total_finite = sum(len(s) for s in segs)
        if total_finite < min_frames:
            skipped += 1
            continue
        bouts_out.append(segs)

    if skipped:
        print(f"  Skipped {skipped} bouts with fewer than {min_frames} finite frames")

    if not bouts_out:
        raise RuntimeError(
            f"No bouts passed min_frames={min_frames} filter for mouse {mouse_id}. "
            f"Total bouts checked: {len(tr.ke)}"
        )

    print(f"  Loaded {len(bouts_out)} bouts "
          f"({sum(sum(len(s) for s in b) for b in bouts_out)} total finite frames)")
    return bouts_out


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A – Maze + Trajectories
# ══════════════════════════════════════════════════════════════════════════════

def draw_maze_panel(ax, maze, trajectories):
    walls = maze.walls

    # Maze walls — near-white fill so trajectory lines pop at print width
    ax.fill(walls[:, 0], walls[:, 1], color='#F7F7F7', zorder=0, alpha=1.0)
    ax.plot(walls[:, 0], walls[:, 1], '-', color=MAZE_WALL,
            linewidth=1.5 * LW_SCALE, alpha=0.9, zorder=1)

    # Trajectories – each bout is one colour, fading as it progresses.
    # `trajectories` is a list of bouts; each bout is a list of finite segments
    # (each segment is an (M,2) array in maze coordinates, fully NaN-free).
    for i, bout_segments in enumerate(trajectories):
        color = TRAJ_COLORS[i % len(TRAJ_COLORS)]
        # Concatenate all segments to compute total length for alpha scaling
        all_pts = np.concatenate(bout_segments, axis=0)
        n_total = len(all_pts)
        drawn = 0
        for seg in bout_segments:
            n = len(seg)
            for j in range(n - 1):
                alpha = 0.25 + 0.65 * (drawn + j) / max(n_total - 1, 1)
                ax.plot(seg[j:j+2, 0], seg[j:j+2, 1],
                        '-', color=color, linewidth=1.8 * LW_SCALE,
                        alpha=alpha, solid_capstyle='round', zorder=2)
            drawn += n
        # Arrowhead at end of last segment
        last_seg = bout_segments[-1]
        if len(last_seg) >= 3:
            dx = last_seg[-1, 0] - last_seg[-3, 0]
            dy = last_seg[-1, 1] - last_seg[-3, 1]
            norm = np.hypot(dx, dy) + 1e-9
            ax.annotate('', xy=last_seg[-1],
                        xytext=last_seg[-1] - 0.5 * np.array([dx, dy]) / norm,
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.0 * LW_SCALE),
                        zorder=3)

    # Entry / exit labels — agents are spawned at left-centre of the maze
    ax.text(-0.55, 7.0, 'Entry', ha='left', va='center', fontsize=9,
            color='#555555', fontstyle='italic', rotation=90)

    ax.set_xlim(-0.6, 15.1)
    ax.set_ylim(-1.0, 15.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Small colour legend for bouts – bottom-right to keep top-left clear for panel label
    handles = [mpatches.Patch(color=TRAJ_COLORS[i], alpha=0.85,
                              label=f'Bout {i+1}')
               for i in range(min(len(trajectories), 5))]
    ax.legend(handles=handles, loc='lower left', fontsize=8.5,
              framealpha=0.85, edgecolor='#cccccc', ncol=2, columnspacing=0.8)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL B – Agent Neural Network Architecture
# ══════════════════════════════════════════════════════════════════════════════

SENSORY_LABELS = {
    'S0': 'Fwd dist',
    'S1': 'Left dist',
    'S2': 'Right dist',
    'S3': 'Prev spd',
    'S4': 'Prev turn',
    'S5': 'Noise',
}
MOTOR_LABELS = {'M0': 'Speed', 'M1': 'Turn'}


def _get_node_pos(G):
    S = sorted([n for n in G.nodes if G.nodes[n]['type'] == 'sensory'],
               key=lambda x: G.nodes[x]['idx'])
    I = sorted([n for n in G.nodes if G.nodes[n]['type'] == 'inter'],
               key=lambda x: G.nodes[x]['idx'])
    M = sorted([n for n in G.nodes if G.nodes[n]['type'] == 'motor'],
               key=lambda x: G.nodes[x]['idx'])
    pos = {}
    for lst, xval, spread in [(S, -X_SPREAD, Y_SPREAD_S),
                               (I,  0.0,     Y_SPREAD_I),
                               (M,  X_SPREAD, Y_SPREAD_M)]:
        ys = np.linspace(spread/2, -spread/2, len(lst)) if len(lst) > 1 else [0.0]
        for i, n in enumerate(lst):
            pos[n] = np.array([xval, ys[i]])
    return pos


def draw_network_panel(ax, agent, gfx_scale: float = 1.6):
    G = agent.to_networkx()
    pos = _get_node_pos(G)

    exc_edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if d['weight'] > 0]
    inh_edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if d['weight'] < 0]

    def _scale(ws):
        return [0.6 + 1.2 * abs(w) for w in ws]

    # Visual node sizes (matplotlib points² area), scaled with gfx_scale.
    node_size_map = {'sensory': int(400*gfx_scale),
                     'inter':   int(340*gfx_scale),
                     'motor':   int(480*gfx_scale)}
    nodelist = list(G.nodes())

    # networkx shrinks edges by sqrt(node_size)/2 from the target centre, but a
    # circular marker drawn with the same `s` value has visible radius
    # sqrt(node_size/π) ≈ 1.13× larger. Inflate the size we hand to the edge
    # router by 4/π so its shrink lands at the visible boundary, not inside it.
    EDGE_INFLATE = 4.0 / np.pi
    node_size_for_edges = [node_size_map[G.nodes[n]['type']] * EDGE_INFLATE
                           for n in nodelist]

    # Lock the data limits NOW so transData is correct when we convert pixels
    # to data units below (used to land the inhibitory dots exactly where
    # networkx clipped the line).
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-3.7, 3.7)
    ax.axis('off')

    edge_kw = dict(
        connectionstyle='arc3,rad=0.22',
        nodelist=nodelist, node_size=node_size_for_edges,
        min_source_margin=0, min_target_margin=0,
    )
    if exc_edges:
        nx.draw_networkx_edges(G, pos, ax=ax,
                               edgelist=[(u, v) for u, v, _ in exc_edges],
                               width=_scale([w for _, _, w in exc_edges]),
                               edge_color=EXC_COLOR, alpha=0.70,
                               arrowstyle='-|>', arrowsize=10, **edge_kw)
    if inh_edges:
        # No bracket terminator — render the line then overlay a small filled
        # circle at each target end as the inhibitory symbol.
        nx.draw_networkx_edges(G, pos, ax=ax,
                               edgelist=[(u, v) for u, v, _ in inh_edges],
                               width=_scale([w for _, _, w in inh_edges]),
                               edge_color=INH_COLOR, alpha=0.70,
                               arrowstyle='-', **edge_kw)

        # Helper: convert N pixels along a data-direction to a data-space offset.
        o_disp = ax.transData.transform([0.0, 0.0])

        def _pixels_to_data(direction, n_pixels):
            direction = np.asarray(direction, dtype=float)
            direction = direction / (np.linalg.norm(direction) + 1e-9)
            p_disp = ax.transData.transform(direction) - o_disp
            p_disp = p_disp / (np.linalg.norm(p_disp) + 1e-9)
            q_disp = o_disp + n_pixels * p_disp
            return (ax.transData.inverted().transform(q_disp)
                    - ax.transData.inverted().transform(o_disp))

        rad = 0.22
        dpi = ax.figure.dpi
        for u, v, _ in inh_edges:
            src = np.asarray(pos[u], dtype=float)
            tgt = np.asarray(pos[v], dtype=float)
            d = tgt - src
            mid = 0.5 * (src + tgt)
            # arc3 control point: (mid_x + rad*dy, mid_y - rad*dx)
            ctrl = mid + rad * np.array([d[1], -d[0]])
            tan = tgt - ctrl
            tan /= (np.linalg.norm(tan) + 1e-9)
            # Place the dot at the visible node radius (matches the inflated
            # edge clip above), so dot and line tip coincide on the boundary.
            radius_pts = np.sqrt(node_size_map[G.nodes[v]['type']] / np.pi)
            radius_px = radius_pts * dpi / 72.0
            offset_data = _pixels_to_data(tan, radius_px)
            cx, cy = tgt - offset_data
            ax.scatter([cx], [cy], s=22, color=INH_COLOR,
                       alpha=0.95, edgecolors='none', zorder=4)

    nS = [n for n in G.nodes if G.nodes[n]['type'] == 'sensory']
    nI = [n for n in G.nodes if G.nodes[n]['type'] == 'inter']
    nM = [n for n in G.nodes if G.nodes[n]['type'] == 'motor']

    def _colors(lst):
        return [EXC_COLOR if G.nodes[n].get('node_type', 1) == 1
                else INH_COLOR for n in lst]

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nS, node_color=_colors(nS),
                           node_shape='s', node_size=node_size_map['sensory'], alpha=0.82,
                           edgecolors='#333333', linewidths=1.3 * LW_SCALE)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nI, node_color=_colors(nI),
                           node_shape='o', node_size=node_size_map['inter'], alpha=0.82,
                           edgecolors='#333333', linewidths=1.3 * LW_SCALE)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nM, node_color=_colors(nM),
                           node_shape='^', node_size=node_size_map['motor'], alpha=0.82,
                           edgecolors='#333333', linewidths=1.3 * LW_SCALE)

    # Sensory input labels (left of nodes)
    for n in nS:
        ax.text(pos[n][0] - 0.75, pos[n][1],
                SENSORY_LABELS.get(n, n),
                ha='right', va='center', fontsize=7, color='#333333')

    # Motor output labels (right of nodes)
    for n in nM:
        ax.text(pos[n][0] + 0.55, pos[n][1],
                MOTOR_LABELS.get(n, n),
                ha='left', va='center', fontsize=7.5,
                color='#333333', fontweight='bold')

    # Column headers
    for xval, lbl in [(-X_SPREAD, 'Sensory'), (0.0, 'Interneurons'), (X_SPREAD, 'Motor')]:
        ax.text(xval, Y_SPREAD_I/2 + 0.35, lbl, ha='center', fontsize=8,
                fontweight='bold', color='#444444')

    # Dale's Law / constraint annotation
    ax.text(0.0, -(Y_SPREAD_I/2 + 0.5),
            "Dale's Law  |  max 3 in / 3 out  |  weights in {0.25, 1.0}",
            ha='center', va='top', fontsize=6.5, color='#666666', fontstyle='italic')

    # Legend
    from matplotlib.lines import Line2D
    exc_h = Line2D([0], [0], color=EXC_COLOR, lw=2.0 * LW_SCALE,
                   marker=r'$\rightarrow$', markersize=9 * MARKER_SCALE, markeredgewidth=0,
                   markerfacecolor=EXC_COLOR, label='Excitatory (E)')
    inh_h = Line2D([0], [0], color=INH_COLOR, lw=2.0 * LW_SCALE,
                   marker='o', markersize=6 * MARKER_SCALE, markeredgewidth=0,
                   markerfacecolor=INH_COLOR, label='Inhibitory (I)')
    ax.legend(handles=[exc_h, inh_h], fontsize=7,
              framealpha=0.85, edgecolor='#cccccc',
              handlelength=2.0, handletextpad=0.6, loc=[0.8, 0.1])


# ══════════════════════════════════════════════════════════════════════════════
# PANEL C – Truncation Selection Mechanism
# ══════════════════════════════════════════════════════════════════════════════

def draw_selection_panel(ax):
    ax.set_xlim(0.6, 9.6)
    ax.set_ylim(1.7, 6.25)
    ax.axis('off')

    # ── Box layout ──────────────────────────────────────────────────────────
    BOX_W, BOX_H = 6.0, 0.80
    cx     = 4.8          # slightly left of centre to leave room for loop arrow
    GAP    = 0.35         # vertical gap between boxes

    # Boxes top → bottom: init, evaluate, select, mutate
    top_y = 5.8
    ys = [top_y - i * (BOX_H + GAP) for i in range(4)]

    box_defs = [
        (ys[0], "Initialise population of 500 agents\nwith random network weights",               '#4DBBD5'),
        (ys[1], "Evaluate each agent in the maze simulation\nusing 4 behavioural fitness metrics", '#F39B7F'),
        (ys[2], "Select top 10%  (50 elites)\nCopy unchanged to next generation",                  '#00A087'),
        (ys[3], "Mutate each elite to create 9 offspring\nRestore population to 500 agents",       '#3C5488'),
    ]

    box_centres = []
    for by, txt, accent in box_defs:
        # Coloured accent bar on left
        ax.add_patch(FancyBboxPatch(
            (cx - BOX_W/2, by - BOX_H/2), 0.18, BOX_H,
            boxstyle='round,pad=0.02', facecolor=accent, edgecolor='none', zorder=3))
        # Main box body
        ax.add_patch(FancyBboxPatch(
            (cx - BOX_W/2 + 0.18, by - BOX_H/2), BOX_W - 0.18, BOX_H,
            boxstyle='round,pad=0.02',
            facecolor='#f8f8f8', edgecolor='#bbbbbb', linewidth=0.9 * LW_SCALE, zorder=3))
        # Label text
        ax.text(cx + 0.09, by, txt,
                ha='center', va='center', fontsize=10.0, color='#222222', zorder=4)
        box_centres.append((cx, by))

    # ── Downward arrows between consecutive boxes ────────────────────────────
    for i in range(len(box_centres) - 1):
        _, y0 = box_centres[i]
        _, y1 = box_centres[i + 1]
        ax.annotate('',
                    xy=(cx, y1 + BOX_H/2 + 0.05),
                    xytext=(cx, y0 - BOX_H/2 - 0.05),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.3 * LW_SCALE), zorder=5)

    # ── Right-angle loop arrow: box 4 right edge → column → up → box 2 right ─
    right_edge = cx + BOX_W / 2          # right side of all boxes
    loop_x     = right_edge + 0.65       # loop column

    y_from = box_centres[-1][1]          # centre-y of "Mutate" box
    y_to   = box_centres[1][1]           # centre-y of "Evaluate" box

    # Leg 1: horizontal from box-4 right to loop column
    ax.plot([right_edge, loop_x], [y_from, y_from],
            '-', color='#888888', lw=1.3 * LW_SCALE, zorder=5)
    # Leg 2: vertical from box-4 height up to box-2 height
    ax.plot([loop_x, loop_x], [y_from, y_to],
            '-', color='#888888', lw=1.3 * LW_SCALE, zorder=5)
    # Leg 3: arrow pointing left back into box-2 right edge
    ax.annotate('',
                xy=(right_edge, y_to),
                xytext=(loop_x, y_to),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1.3 * LW_SCALE), zorder=5)

    # Label on the vertical leg
    ax.text(loop_x + 0.15, (y_from + y_to) / 2,
            'Repeat\nx150 gen',
            ha='left', va='center', fontsize=9.5, color='#888888', fontstyle='italic')


# ══════════════════════════════════════════════════════════════════════════════
# PANEL D – Metric Illustrations  (four 2×2 inset subpanels)
# ══════════════════════════════════════════════════════════════════════════════

def _style_metric_ax(ax, title):
    ax.set_xlim(0, 4)
    ax.set_ylim(0.0, 2.6)
    ax.axis('off')
    ax.text(2.0, 2.55, title, ha='center', va='top',
            fontsize=10.5, fontweight='bold', color='#333333')


def _draw_tortuosity(ax):
    _style_metric_ax(ax, 'Path Straightness')

    # Direct path – straightness ratio close to 1 (low tortuosity)
    ax.annotate('', xy=(3.5, 2.0), xytext=(0.5, 2.0),
                arrowprops=dict(arrowstyle='->', color='#00A087', lw=2.0 * LW_SCALE))
    ax.text(2.0, 2.15, 'Direct  (ratio near 1)', ha='center', va='bottom',
            fontsize=8.5, color='#00A087')

    # Winding path – straightness ratio much less than 1
    t = np.linspace(0, 1, 100)
    xw = 0.5 + 3.0 * t
    yw = 1.3 + 0.28 * np.sin(6 * np.pi * t)
    ax.plot(xw, yw, '-', color='#E64B35', lw=2.0 * LW_SCALE, alpha=0.85, solid_capstyle='round')
    ax.text(2.0, 0.95, 'Winding  (ratio near 0)', ha='center', va='top',
            fontsize=8.5, color='#E64B35')

    # Formula: displacement / path-length  (1 = straight, 0 = maximally winding)
    ax.text(2.0, 0.45, r'$d_{\rm disp}\,/\,d_{\rm path}\ \in [0,1]$',
            ha='center', va='center', fontsize=10, color='#666666')


def _draw_occupancy(ax):
    _style_metric_ax(ax, 'Occupancy')

    rng = np.random.RandomState(3)
    n_corridors = 9
    freqs_mouse = rng.dirichlet(np.ones(n_corridors) * 1.5)
    freqs_agent = rng.dirichlet(np.ones(n_corridors) * 1.5)

    bw = 0.38
    scale = 1.9

    for i, (fm, fa) in enumerate(zip(freqs_mouse, freqs_agent)):
        xi = 0.2 + i * (3.6 / n_corridors)
        ax.bar(xi,          fm * scale, width=bw * 3.6 / n_corridors * 0.9,
               color='#4DBBD5', alpha=0.80, bottom=0.3, edgecolor='white', linewidth=0.3 * LW_SCALE)
        ax.bar(xi + bw * 3.6 / n_corridors * 0.95, fa * scale,
               width=bw * 3.6 / n_corridors * 0.9,
               color='#E64B35', alpha=0.70, bottom=0.3, edgecolor='white', linewidth=0.3 * LW_SCALE)

    # Legend
    ax.legend(loc='upper right', fontsize=8, framealpha=0.0,
              handles=[mpatches.Patch(color='#4DBBD5', alpha=0.80, label='Mouse'),
                       mpatches.Patch(color='#E64B35', alpha=0.70, label='Agent')])

    ax.text(2.0, 0.12, 'Maze corridors  →  JS divergence',
            ha='center', va='bottom', fontsize=8.5, color='#666666')


def _draw_markov(ax):
    _style_metric_ax(ax, 'Markov bias profile')

    # Small 3-level binary tree
    nodes = {
        'A': (2.0, 2.1),
        'B': (1.0, 1.35),
        'C': (3.0, 1.35),
        'D': (0.45, 0.55),
        'E': (1.55, 0.55),
        'F': (2.45, 0.55),
        'G': (3.55, 0.55),
    }
    edges = [
        ('A', 'B', 0.65), ('A', 'C', 0.35),
        ('B', 'D', 0.40), ('B', 'E', 0.60),
        ('C', 'F', 0.55), ('C', 'G', 0.45),
    ]
    for u, v, p in edges:
        xu, yu = nodes[u]; xv, yv = nodes[v]
        lw=0.7 * LW_SCALE + 2.5 * p
        ax.annotate('', xy=(xv, yv + 0.14), xytext=(xu, yu - 0.14),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=lw))
        mx, my = (xu + xv) / 2, (yu + yv) / 2 + 0.05
        offset = -0.18 if xv < xu else 0.18
        ax.text(mx + offset, my, f'{p:.2f}',
                fontsize=8, color='#555555', ha='center', va='center')

    for _, (x, y) in nodes.items():
        ax.add_patch(plt.Circle((x, y), 0.13, color='#4DBBD5',
                                ec='#334466', lw=1.1 * LW_SCALE, zorder=3))

    ax.text(2.0, 0.08, '2nd-order: P(next | curr, prev)  — Euclidean distance',
            ha='center', va='bottom', fontsize=8.0, color='#666666')


def _draw_turn_bias(ax):
    _style_metric_ax(ax, 'Turn Bias')

    # Corridor stem (coming from below)
    ax.plot([2.0, 2.0], [0.9, 1.55], '-', color='#777777', lw=3.0 * LW_SCALE,
            solid_capstyle='round')

    # Left branch (~35%)
    ax.annotate('', xy=(0.8, 2.3), xytext=(2.0, 1.55),
                arrowprops=dict(arrowstyle='->', color='#E64B35',
                                lw=2.5 * LW_SCALE, mutation_scale=14))
    ax.text(0.65, 2.42, '35%\nLeft', ha='center', va='bottom',
            fontsize=9.5, color='#E64B35', fontweight='bold')

    # Right branch (~65%)
    ax.annotate('', xy=(3.2, 2.3), xytext=(2.0, 1.55),
                arrowprops=dict(arrowstyle='->', color='#00A087',
                                lw=3.8 * LW_SCALE, mutation_scale=14))
    ax.text(3.35, 2.42, '65%\nRight', ha='center', va='bottom',
            fontsize=9.5, color='#00A087', fontweight='bold')

    # Bias bar
    bar_x0, bar_y0, bar_w, bar_h = 0.6, 0.55, 2.8, 0.25
    ax.add_patch(FancyBboxPatch((bar_x0, bar_y0), bar_w * 0.35, bar_h,
                                boxstyle='round,pad=0.01',
                                facecolor='#E64B35', alpha=0.7, ec='none'))
    ax.add_patch(FancyBboxPatch((bar_x0 + bar_w * 0.35, bar_y0), bar_w * 0.65, bar_h,
                                boxstyle='round,pad=0.01',
                                facecolor='#00A087', alpha=0.7, ec='none'))

    ax.text(2.0, 0.22, r'$(R - L)\,/\,(R + L)$  per junction',
            ha='center', va='center', fontsize=9, color='#666666')


def draw_metrics_panel(ax_host):
    """Embed 4 metric sub-panels as inset axes in a 2×2 grid."""
    ax_host.axis('off')
    pad = 0.012
    cell_w = (1.0 - 3 * pad) / 2.0
    cell_h = (1.0 - 3 * pad) / 2.0
    left2 = pad + cell_w + pad
    bot2 = pad + cell_h + pad
    # [left, bottom, width, height] in axes-fraction coords
    positions = [
        [pad,   bot2, cell_w, cell_h],   # top-left:     Tortuosity
        [left2, bot2, cell_w, cell_h],   # top-right:    Occupancy
        [pad,   pad,  cell_w, cell_h],   # bottom-left:  Markov
        [left2, pad,  cell_w, cell_h],   # bottom-right: Turn Bias
    ]
    funcs = [_draw_tortuosity, _draw_occupancy, _draw_markov, _draw_turn_bias]

    for (l, b, w, h), fn in zip(positions, funcs):
        inset = ax_host.inset_axes([l, b, w, h])
        fn(inset)
        # Light border around each sub-panel
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#dddddd')
            spine.set_linewidth(0.6)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def create_setup_figure(output_path='sim_setup_figure.png'):
    print("=== Simulation Setup Figure ===")
    config = load_config()
    maze   = create_maze(6)

    # Trajectories – load real mouse keypoints; no silent fallback
    print("Loading trajectories...")
    trajs = load_mouse_trajectories('B5', n_bouts=5)
    print(f"  {len(trajs)} bouts ready.")

    # Random agent (seed fixed for reproducibility)
    np.random.seed(42)
    agent = Agent(config.network)

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13.5, 8.4))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.018, right=0.982,
        top=0.97, bottom=0.018,
        wspace=0.0, hspace=0.05,
        width_ratios=[1.05, 1.0],
        height_ratios=[1.4, 1.0],
    )

    ax_maze = fig.add_subplot(gs[0, 0])
    ax_net  = fig.add_subplot(gs[0, 1])
    ax_sel  = fig.add_subplot(gs[1, 0])
    ax_met  = fig.add_subplot(gs[1, 1])

    # Subtle panel backgrounds matching the sketch colours
    bg_colors = {
        ax_maze: '#edf2fb',
        ax_net:  '#fff5e8',
        ax_sel:  '#edf2fb',
        ax_met:  '#fffde8',
    }
    for ax, fc in bg_colors.items():
        ax.set_facecolor(fc)
        for sp in ax.spines.values():
            sp.set_edgecolor('#c8c8c8')
            sp.set_linewidth(0.7)

    # ── Draw ─────────────────────────────────────────────────────────────────
    print("Drawing maze panel…")
    draw_maze_panel(ax_maze, maze, trajs)

    print("Drawing network panel…")
    draw_network_panel(ax_net, agent)

    print("Drawing selection panel…")
    draw_selection_panel(ax_sel)

    print("Drawing metrics panel…")
    draw_metrics_panel(ax_met)

    # ── Titles ───────────────────────────────────────────────────────────────
    ax_maze.set_title('Hierarchical Binary Tree Maze & Representative Trajectories',
                      fontsize=12.5, pad=3, color='#222222', fontweight='semibold')
    ax_net.set_title('Agent Neural Network Architecture',
                     fontsize=12.5, pad=3, color='#222222', fontweight='semibold')
    ax_sel.set_title('Truncation Selection Mechanism',
                     fontsize=12.5, pad=3, color='#222222', fontweight='semibold')
    ax_met.set_title('Behavioural Fitness Metrics',
                     fontsize=12.5, pad=3, color='#222222', fontweight='semibold')

    # ── Panel labels (A B C D) ───────────────────────────────────────────────
    for ax, lbl in [(ax_maze, 'A'), (ax_net, 'B'),
                    (ax_sel,  'C'), (ax_met, 'D')]:
        ax.text(0.015, 0.985, lbl, transform=ax.transAxes,
                fontsize=16, fontweight='bold',
                ha='left', va='top', color='#111111')

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"Saving -> {output_path}")
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Done!  {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate simulation setup figure')
    parser.add_argument('--output', '-o', default='sim_setup_figure.png',
                        help='Output path (default: sim_setup_figure.png)')
    args = parser.parse_args()
    create_setup_figure(args.output)
