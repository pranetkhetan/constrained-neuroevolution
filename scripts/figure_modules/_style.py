"""
Shared publication style, colour palettes, and plotting utilities.

All figure modules import from here to ensure consistent appearance.
Edit this file to change style across all figures without re-extracting data.
"""

import matplotlib.pyplot as plt


# ── Mice ────────────────────────────────────────────────────────────────
MICE = ['B5', 'B6', 'B7', 'D3', 'D4', 'D5', 'D7', 'D8', 'D9']
N_REPS = 6

# Paul Tol "muted" qualitative palette — colourblind-safe for 9 categories.
# Reference: https://personal.sron.nl/~pault/#sec:qualitative
# Spare: #DDDDDD (grey) if a 10th neutral category is needed.
MOUSE_COLORS = {
    "B5": "#CC6677",   # rose
    "B6": "#332288",   # indigo
    "B7": "#44AA99",   # teal
    "D3": "#117733",   # forest green
    "D4": "#882255",   # wine
    "D5": "#88CCEE",   # sky blue
    "D7": "#DDCC77",   # sand
    "D8": "#AA4499",   # purple
    "D9": "#999933",   # olive
}

# ── Role colours ────────────────────────────────────────────────────────
EVOLVED_COL = "#332288"   # indigo
RANDOM_COL  = "#DDDDDD"   # Tol grey (neutral / spare slot)
OWN_COL     = "#CC6677"   # rose
OTHER_COL   = "#88CCEE"   # sky blue
GEN_COL     = "#44AA99"   # teal

# ── 10-run palette (Tol muted + grey spare) ─────────────────────────────
RUN_COLORS = [
    "#CC6677", "#332288", "#44AA99", "#117733", "#882255",
    "#88CCEE", "#DDCC77", "#AA4499", "#999933", "#DDDDDD",
]

# ── Metric labels ───────────────────────────────────────────────────────
METRIC_LABELS = {
    "markov_score":     "Markov Penalty",
    "occupancy_score":  "Occupancy Penalty",
    "tortuosity_score": "Tortuosity Penalty",
    "turn_bias_score":  "Turn Bias Penalty",
}
METRIC_KEYS = ["markov_score", "occupancy_score", "tortuosity_score", "turn_bias_score"]

# ── Neuron labels (14 neurons) ──────────────────────────────────────────
NEURON_LABELS = [
    'F', 'L', 'R', 'PSpd', 'PTrn', 'N',
    'I0', 'I1', 'I2', 'I3', 'I4', 'I5',
    'Spd', 'Trn',
]

# ── Figure sizes (single source of truth) ───────────────────────────────
# Change FIG_SCALE to resize all figures uniformly (1.0 = publication baseline).
FIG_SCALE = 1.5

FIGSIZE = {
    # ── paper_v2 main figures (absolute authored sizes — not scaled by FIG_SCALE)
    # These are intentionally sized for final print layout. Adjust with care.
    'fig1':           (10,   11.0),   # fig1_system_v2
    'fig2':           (13.0,  7.8),   # fig2_paradox_v2
    'fig3':           ( 9.3,  6.5),   # fig3_causal_v2
    'figA':           (15.0, 10.5),   # figA_topo_flat_v2
    'figC':           (14.0,  9.0),   # figC_sensitivity_v2
    'emb_rsa':        (13.0, 11.5),   # fig_embedding_rsa
    'circuit':        (12.0, 15.0),   # fig_circuit_comparison
    'methods_topo':   (14.0,  7.0),   # fig_method_topology_schematic
    'methods_fitness': (8.0,  6.0),   # fig_methods_fitness
    'evr':            ( 7.0,  8.0),   # fig_evr (used as Panel G in fig1)
    'sim_setup':      (13.5,  8.4),   # sim_setup_figure (standalone preview)
    # ── paper_v2 supplementary figures (base / FIG_SCALE, scales uniformly)
    'supp_mi_clust':  (14.0, 11.0),   # supp_s_mi_clustering
    'supp_s2':        (12.0,  4.0),   # supp_s2_clustering
    'supp_s4':        (12.0,  4.0),   # supp_s4_trajectories
    'supp_s5':        (12.0,  4.0),   # supp_s5_fixedpoints
    'supp_s9':        ( 6.0,  4.0),   # supp_s9_nonzero
    'supp_s11':       ( 6.0,  4.0),   # supp_s11_randomnull
    'supp_s12':       ( 4.0,  4.0),   # supp_s12_difficulty
    'supp_s15':       ( 8.0,  4.0),   # supp_s15_generalist
    'supp_s17':       (13.0,  4.5),   # supp_s17_dynamics_null
    'supp_act_emb':   (11.0,  5.0),   # supp_act_emb_ctrl
    'supp_b3_dim':    (12.0,  5.0),   # supp_b3_dim
    'supp_d4_attr':   (18.0,  5.5),   # supp_d4_attractor
    'supp_emb_rob':   ( 8.0,  4.5),   # supp_emb_robustness
    'supp_holdout':   (10.0,  5.0),   # supp_holdout
    'supp_ablation':  (10.0,  9.0),   # supp_ablation_heatmap
    'supp_spec_evol': (15.0, 12.0),   # supp_spec_evol_traj
    'supp_sens_comm': ( 9.0, 12.0),   # supp_sens_commitment_evo
    # ── legacy main figures (keep for backward compat with non-v2 scripts)
    'fig5':  (14   * FIG_SCALE, 6.0  * FIG_SCALE),
    'fig7':  (12   * FIG_SCALE, 4.0  * FIG_SCALE),
    'fig8':  (13   * FIG_SCALE, 5.5  * FIG_SCALE),
    'fig9':  (18   * FIG_SCALE, 6.0  * FIG_SCALE),
    # ── legacy supplementary figures
    's2':    (10   * FIG_SCALE, 4.0  * FIG_SCALE),
    's3':    (7    * FIG_SCALE, 6.0  * FIG_SCALE),
    's4':    (14   * FIG_SCALE, 5.0  * FIG_SCALE),
    's5':    (14   * FIG_SCALE, 4.0  * FIG_SCALE),
    's6':    (9    * FIG_SCALE, 3.0  * FIG_SCALE),
    's7':    (8    * FIG_SCALE, 5.0  * FIG_SCALE),
    's8':    (6    * FIG_SCALE, 4.0  * FIG_SCALE),
    's9':    (6    * FIG_SCALE, 5.0  * FIG_SCALE),
    's10a':  (12   * FIG_SCALE, 8.0  * FIG_SCALE),
    's10b':  (6    * FIG_SCALE, 4.0  * FIG_SCALE),
    's11':   (6    * FIG_SCALE, 4.0  * FIG_SCALE),
    's12':   (5    * FIG_SCALE, 4.5  * FIG_SCALE),
    's13':   (5.5  * FIG_SCALE, 2.8  * FIG_SCALE),
    's14':   (4.5  * FIG_SCALE, 3.0  * FIG_SCALE),
    's15':   (12   * FIG_SCALE, 4.0  * FIG_SCALE),
    's16':   (16.0 * FIG_SCALE, 10.8 * FIG_SCALE),
}


# ── Publication font sizes (single source of truth) ─────────────────────
# Change FONT_SCALE to resize all text uniformly across every figure.
# 1.0 = compact publication baseline; 1.5 = 150% (current default).
FONT_SCALE  = 1.5

FS_BASE     = round(7   * FONT_SCALE, 2)   # default body text (= font.size in rcParams)
FS_PANEL    = round(8   * FONT_SCALE, 2)   # bold A / B / C panel-letter labels
FS_TITLE    = round(7.5 * FONT_SCALE, 2)   # ax.set_title() (= axes.titlesize in rcParams)
FS_LABEL    = round(7   * FONT_SCALE, 2)   # axis labels (= axes.labelsize in rcParams)
FS_TICK     = round(6.5 * FONT_SCALE, 2)   # tick labels (= x/ytick.labelsize in rcParams)
FS_LEGEND   = round(6.5 * FONT_SCALE, 2)   # legend entries (= legend.fontsize in rcParams)
FS_ANNOT    = round(6   * FONT_SCALE, 2)   # in-figure annotations, significance callouts
FS_SMALL    = round(5.5 * FONT_SCALE, 2)   # mouse-ID labels, boundary markers
FS_MICRO    = round(5   * FONT_SCALE, 2)   # colorbar ticks, dense scatter annotations
FS_SUPTITLE = round(8.5 * FONT_SCALE, 2)   # fig.suptitle() for supplementary figures


def save_figure(fig, pdf_path: str, png_dpi: int = 150) -> list:
    """Save as PDF (300 dpi) + PNG (png_dpi), close figure, print paths."""
    import matplotlib.pyplot as plt
    fig.savefig(pdf_path, dpi=300)
    png_path = pdf_path.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=png_dpi)
    plt.close(fig)
    print(f"Saved -> {pdf_path}")
    print(f"       -> {png_path}")
    return [pdf_path, png_path]


def label_panel(ax, letter: str) -> None:
    """Bold panel letter at a fixed axes-referenced position, no title text."""
    ax.text(-0.05, 1.05, letter, transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight="bold", ha="left", va="bottom")


def pub_despine(ax, **kwargs):
    """Despine with publication-style offset + trim. Drop into any panel."""
    import seaborn as sns
    sns.despine(ax=ax, offset=8, trim=True, **kwargs)


def apply_pub_style(font_scale: float | None = None):
    """Set matplotlib rcParams for publication. Call once at startup.
    Pass font_scale to override the module FONT_SCALE for a single figure
    (e.g. 1.15 for figures authored near final print width)."""
    fs = FONT_SCALE if font_scale is None else font_scale
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":       round(7*fs, 2),
        "axes.titlesize":  round(7.5*fs, 2),
        "axes.labelsize":  round(7*fs, 2),
        "xtick.labelsize": round(6.5*fs, 2),
        "ytick.labelsize": round(6.5*fs, 2),
        "legend.fontsize": round(6.5*fs, 2),
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "lines.linewidth": 0.9,
    })

# ── Graphic-size contract (keep font:graphic ratio stable across resizes) ──
# Scale graphic primitives the SAME way you scale a figure's authored width.
GFX_SCALE   = 1.0          # bump >1 to enlarge graphics relative to text
NODE_SCALE  = GFX_SCALE    # multiply node_size_map by this
LW_SCALE    = GFX_SCALE    # multiply line/edge widths
MARKER_SCALE= GFX_SCALE    # multiply scatter marker sizes


def sig_label(p: float) -> str:
    """Format a p-value for annotation."""
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3f}"


def bracket(ax, x1, x2, y, dy, label, fontsize=FS_ANNOT):
    """Draw a significance bracket between positions x1 and x2."""
    ax.plot([x1, x1, x2, x2], [y, y + dy, y + dy, y], lw=0.8, c='k')
    ax.text((x1 + x2) / 2, y + dy * 1.1, label,
            ha='center', va='bottom', fontsize=fontsize)
