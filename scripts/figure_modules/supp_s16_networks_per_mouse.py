"""
Supplementary Figure S16 (fig:supp_networks_per_mouse) — One evolved network per mouse.

For each of the 9 mice, loads the best agent from rep 1's final generation
(`data/agents/results_<MOUSE>_r1/gen_<MAX>/summary.pkl`) and renders its
NetworkX graph in a 3x3 panel. Layout/edge style follow the conventions in
`scripts/visualize.py`, with wider spread and explicit labels for every node
(sensory: F/L/R/P_Spd/P_Trn/N; inter: I0..I5; motor: Speed/Turn) plus an
(E)/(I) tag from each node's Dale type.

Output: fig_supp_networks_per_mouse.pdf
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from ._style import apply_pub_style, FIGSIZE, LW_SCALE, MARKER_SCALE

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figure_modules._loaders import _load_pickle_cpu
from scripts.figure_modules._style import MICE, FS_SMALL, FS_ANNOT, FS_TITLE, FS_SUPTITLE


# ── Panel layout (wider than visualize.py to keep edges and labels readable) ──
X_SPREAD   = 3.9
Y_SPREAD_S = 3.6
Y_SPREAD_I = 6.0
Y_SPREAD_M = 1.8

XLIM = (-6.0, 6.0)
YLIM = (-3.6, 3.8)

SENSORY_LABELS = {
    'S0': 'F', 'S1': 'L', 'S2': 'R',
    'S3': 'P_Spd', 'S4': 'P_Trn', 'S5': 'N',
}
MOTOR_LABELS = {'M0': 'Speed', 'M1': 'Turn'}


def _latest_gen_dir(run_dir: str) -> str:
    gens = [
        int(d.split('_')[1])
        for d in os.listdir(run_dir)
        if d.startswith('gen_') and d.split('_')[1].isdigit()
    ]
    if not gens:
        raise FileNotFoundError(f"No gen_* dirs in {run_dir}")
    return os.path.join(run_dir, f'gen_{max(gens)}')


def _best_agent_for_mouse(project_dir: str, mouse_id: str, rep: int = 1):
    # Prefer data/best_agents.pkl (repo default); fall back to data/agents/ scan.
    bp = os.path.join(project_dir, 'data', 'best_agents.pkl')
    if os.path.exists(bp):
        from scripts.figure_modules._loaders import _load_pickle_cpu
        blob = _load_pickle_cpu(bp)
        rec = blob.get('specialists', {}).get((mouse_id, rep))
        if rec is not None:
            return {'agent': rec['agent'], 'fitness': rec['fitness']}, 'gen_150'
    run_dir = os.path.join(project_dir, 'data', 'agents', f'results_{mouse_id}_r{rep}')
    gen_dir = _latest_gen_dir(run_dir)
    summary_path = os.path.join(gen_dir, 'summary.pkl')
    results = _load_pickle_cpu(summary_path)
    best = min(results, key=lambda r: r['fitness'])
    return best, os.path.basename(gen_dir)


def _layout(G):
    """Three-column layout with generous vertical spread."""
    S = sorted([n for n in G.nodes if G.nodes[n]['type'] == 'sensory'],
               key=lambda x: G.nodes[x]['idx'])
    I = sorted([n for n in G.nodes if G.nodes[n]['type'] == 'inter'],
               key=lambda x: G.nodes[x]['idx'])
    M = sorted([n for n in G.nodes if G.nodes[n]['type'] == 'motor'],
               key=lambda x: G.nodes[x]['idx'])

    pos = {}
    for nodes, x, span in [(S, -X_SPREAD, Y_SPREAD_S),
                           (I, 0.0,       Y_SPREAD_I),
                           (M, X_SPREAD,  Y_SPREAD_M)]:
        ys = (np.linspace(span / 2, -span / 2, len(nodes))
              if len(nodes) > 1 else [0.0])
        for n, y in zip(nodes, ys):
            pos[n] = np.array([x, y])
    return pos


def _node_label(n, G):
    if n in SENSORY_LABELS:
        base = SENSORY_LABELS[n]
    elif n in MOTOR_LABELS:
        base = MOTOR_LABELS[n]
    else:
        base = n
    tag = 'E' if G.nodes[n].get('node_type', 1) == 1 else 'I'
    return f"{base}({tag})"


EXC_COLOR = '#cc2222'   # match sim_setup_figure Panel B
INH_COLOR = '#2244cc'


# networkx shrinks edges by sqrt(node_size)/2 from the target centre, but a
# circular marker drawn with the same `s` value has visible radius
# sqrt(node_size/π) ≈ 1.13× larger. Inflate the size we hand to the edge
# router by 4/π so its shrink lands at the visible boundary, not inside it.
EDGE_INFLATE = 4.0 / np.pi

# Visual node sizes (matplotlib points² area). Slightly smaller than the
# visualize.py defaults so each network reads as airy at panel scale.
NODE_SIZE_MAP = {'sensory': 320, 'inter': 270, 'motor': 360}
EDGE_RAD = 0.18


def _draw_network_panel(G, ax):
    pos = _layout(G)
    # Skip self-loops: with arrowstyle='-' for inhibitory edges, networkx's
    # self-loop curve degenerates and renders invisibly under the node, but
    # any terminator dot we'd place at its tip would float in empty space.
    # Drop self-loops here — they're rendered manually after layout via
    # _draw_self_loops (networkx's default self-loop curve degenerates with
    # arrowstyle='-' and yields no visible loop).
    edges = [(u, v, d) for u, v, d in G.edges(data=True) if u != v]

    exc_edges = [(u, v) for u, v, d in edges if d['weight'] > 0]
    inh_edges = [(u, v) for u, v, d in edges if d['weight'] < 0]
    exc_w = [d['weight'] for _, _, d in edges if d['weight'] > 0]
    inh_w = [abs(d['weight']) for _, _, d in edges if d['weight'] < 0]

    def scale_w(ws):
        return [0.7 + 2.6 * w for w in ws]

    nodelist = list(G.nodes())
    node_size_for_edges = [NODE_SIZE_MAP[G.nodes[n]['type']] * EDGE_INFLATE
                           for n in nodelist]

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect('equal')
    ax.axis('off')

    edge_kw = dict(
        connectionstyle=f'arc3,rad={EDGE_RAD}',
        nodelist=nodelist, node_size=node_size_for_edges,
        min_source_margin=0, min_target_margin=0,
    )

    if exc_edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=exc_edges, width=scale_w(exc_w),
            edge_color=EXC_COLOR, alpha=0.70, arrowstyle='-|>',
            arrowsize=10, **edge_kw,
        )
    if inh_edges:
        # No bracket terminator — line only here, dots are added by
        # `_add_inh_dots` AFTER tight_layout has finalised the axes geometry.
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=inh_edges, width=scale_w(inh_w),
            edge_color=INH_COLOR, alpha=0.70, arrowstyle='-',
            **edge_kw,
        )

    nS = [n for n in G.nodes if G.nodes[n]['type'] == 'sensory']
    nI = [n for n in G.nodes if G.nodes[n]['type'] == 'inter']
    nM = [n for n in G.nodes if G.nodes[n]['type'] == 'motor']

    def colors(nodes):
        return [EXC_COLOR if G.nodes[n].get('node_type', 1) == 1 else INH_COLOR
                for n in nodes]

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nS, node_color=colors(nS),
                           node_shape='s', node_size=NODE_SIZE_MAP['sensory'],
                           alpha=0.82, edgecolors='#333333', linewidths=1.3 * LW_SCALE)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nI, node_color=colors(nI),
                           node_shape='o', node_size=NODE_SIZE_MAP['inter'],
                           alpha=0.82, edgecolors='#333333', linewidths=1.3 * LW_SCALE)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nM, node_color=colors(nM),
                           node_shape='^', node_size=NODE_SIZE_MAP['motor'],
                           alpha=0.82, edgecolors='#333333', linewidths=1.3 * LW_SCALE)

    # Labels: sensory → left, motor → right, inter → above
    label_font = dict(fontsize=FS_SMALL, color='black', fontweight='bold')
    for n in nS:
        ax.text(pos[n][0] - 0.35, pos[n][1], _node_label(n, G),
                ha='right', va='center', **label_font)
    for n in nM:
        ax.text(pos[n][0] + 0.35, pos[n][1], _node_label(n, G),
                ha='left', va='center', **label_font)
    for n in nI:
        ax.text(pos[n][0], pos[n][1] + 0.32, _node_label(n, G),
                ha='center', va='bottom', **label_font)

    # Column headers
    header = dict(color='#333333', weight='bold', size=FS_ANNOT)
    ax.text(-X_SPREAD, YLIM[1] - 0.18, "SENSORY",      ha='center', **header)
    ax.text(0.0,        YLIM[1] - 0.18, "INTERNEURONS", ha='center', **header)
    ax.text(X_SPREAD,   YLIM[1] - 0.18, "MOTOR",       ha='center', **header)


def _draw_self_loops(G, ax):
    """Render self-loops manually as small circles tangent to each node.

    networkx's default self-loop renderer degenerates with `arrowstyle='-'`
    (becomes invisible after node-shrink), so we draw them ourselves: one
    small open circle adjacent to the node per self-loop, rotated 90°
    clockwise (top → right → bottom → left) per successive self-loop on the
    same node. Excitatory loops carry a small inward-pointing arrowhead at
    the tangent point; inhibitory loops carry a filled circle terminator.
    Must run AFTER `tight_layout`.
    """
    # Group self-loops by node, preserving order
    self_loops_by_node: dict = {}
    for u, v, d in G.edges(data=True):
        if u == v:
            self_loops_by_node.setdefault(v, []).append(d['weight'])
    if not self_loops_by_node:
        return

    o_disp = ax.transData.transform([0.0, 0.0])
    o_data = ax.transData.inverted().transform(o_disp)

    def _pix_to_data(px, py):
        return ax.transData.inverted().transform(o_disp + np.array([px, py])) - o_data

    pos = _layout(G)
    dpi = ax.figure.dpi
    DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # top → right → bottom → left

    def _scale_w(w):
        return 0.7 + 2.6 * abs(w)

    for node, weights in self_loops_by_node.items():
        node_r_pts = np.sqrt(NODE_SIZE_MAP[G.nodes[node]['type']] / np.pi)
        node_r_px = node_r_pts * dpi / 72.0
        loop_r_pts = node_r_pts * 0.55
        loop_r_px = loop_r_pts * dpi / 72.0
        center_dist_px = node_r_px + loop_r_px

        # Loop radius in data coords (aspect='equal' → x and y equal).
        r_off = _pix_to_data(loop_r_px, 0)
        loop_r_data = abs(r_off[0])

        for k, w in enumerate(weights):
            dx, dy = DIRECTIONS[k % 4]
            color = EXC_COLOR if w > 0 else INH_COLOR

            # Loop centre offset from node centre by (node_r + loop_r) along dx, dy
            ctr_off = _pix_to_data(dx * center_dist_px, dy * center_dist_px)
            cx = pos[node][0] + ctr_off[0]
            cy = pos[node][1] + ctr_off[1]

            ax.add_patch(mpatches.Circle(
                (cx, cy), loop_r_data,
                fill=False, edgecolor=color, linewidth=_scale_w(w),
                alpha=0.70, zorder=1,
            ))

            # Tangent point: where the loop meets the node boundary.
            tan_off = _pix_to_data(dx * node_r_px, dy * node_r_px)
            tx = pos[node][0] + tan_off[0]
            ty = pos[node][1] + tan_off[1]

            if w < 0:
                # Inhibitory terminator
                ax.scatter([tx], [ty], s=36 * MARKER_SCALE, color=INH_COLOR,
                           alpha=0.95, edgecolors='none', zorder=4)
            else:
                # Excitatory: small inward-pointing arrowhead at the tangent.
                # Tail = tangent slightly outside the node along the loop's
                # local circumferential direction, head = tangent point.
                perp_dx, perp_dy = -dy, dx  # 90° CCW from radial direction
                tail_off_px = 0.6 * loop_r_px
                tail_disp_off = _pix_to_data(perp_dx * tail_off_px, perp_dy * tail_off_px)
                tail_x = tx + tail_disp_off[0]
                tail_y = ty + tail_disp_off[1]
                ax.annotate(
                    '', xy=(tx, ty), xytext=(tail_x, tail_y),
                    arrowprops=dict(arrowstyle='-|>', color=EXC_COLOR,
                                    alpha=0.85, lw=_scale_w(w),
                                    mutation_scale=10,
                                    shrinkA=0, shrinkB=0),
                    zorder=4,
                )


def _add_inh_dots(G, ax):
    """Place inhibitory terminator dots — must be called AFTER tight_layout."""
    pos = _layout(G)
    inh_edges = [(u, v) for u, v, d in G.edges(data=True)
                 if d['weight'] < 0 and u != v]  # self-loops handled by skipping
    if not inh_edges:
        return

    o_disp = ax.transData.transform([0.0, 0.0])

    def _pixels_to_data(direction, n_pixels):
        direction = np.asarray(direction, dtype=float)
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        p_disp = ax.transData.transform(direction) - o_disp
        p_disp = p_disp / (np.linalg.norm(p_disp) + 1e-9)
        q_disp = o_disp + n_pixels * p_disp
        return (ax.transData.inverted().transform(q_disp)
                - ax.transData.inverted().transform(o_disp))

    dpi = ax.figure.dpi
    for u, v in inh_edges:
        src = np.asarray(pos[u], dtype=float)
        tgt = np.asarray(pos[v], dtype=float)
        d = tgt - src
        mid = 0.5 * (src + tgt)
        # arc3 control point: (mid_x + rad*dy, mid_y - rad*dx)
        ctrl = mid + EDGE_RAD * np.array([d[1], -d[0]])
        tan = tgt - ctrl
        tan /= (np.linalg.norm(tan) + 1e-9)
        radius_pts = np.sqrt(NODE_SIZE_MAP[G.nodes[v]['type']] / np.pi)
        radius_px = radius_pts * dpi / 72.0
        offset_data = _pixels_to_data(tan, radius_px)
        cx, cy = tgt - offset_data
        ax.scatter([cx], [cy], s=36 * MARKER_SCALE, color=INH_COLOR,
                   alpha=0.95, edgecolors='none', zorder=4)


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    fig, axes = plt.subplots(3, 3, figsize=FIGSIZE['s16'])
    axes = axes.flatten()

    panel_data = []
    for ax, mouse_id in zip(axes, MICE):
        best, gen_name = _best_agent_for_mouse(store._project, mouse_id, rep=1)
        agent = best['agent']
        G = agent.to_networkx()

        _draw_network_panel(G, ax)
        ax.set_title(
            f"{mouse_id} (r1, {gen_name}, fit={best['fitness']:.2f})",
            fontsize=FS_TITLE, pad=4,
        )
        panel_data.append((ax, G))

    fig.suptitle(
        "Evolved networks across mice (rep 1, best agent of final generation)",
        fontsize=FS_SUPTITLE, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(wspace=0.1, hspace=0.07)

    # tight_layout / subplots_adjust have now resized every axes — force a
    # canvas draw so transData reflects the FINAL geometry, then place the
    # inhibitory dots using the correct pixel→data conversion.
    fig.canvas.draw()
    for ax, G in panel_data:
        _add_inh_dots(G, ax)
        _draw_self_loops(G, ax)

    out = os.path.join(figures_dir, 'fig_supp_networks_per_mouse.pdf')
    fig.savefig(out)
    plt.close(fig)
    return [out]
