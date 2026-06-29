"""
Methods figure — Behavioral fitness metrics (standalone panel D from sim_setup_figure).

Placed in the Methods / Fitness function section.
Output: figures/fig_methods_fitness_metrics.pdf
"""

import os
import sys
import matplotlib.pyplot as plt

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.sim_setup_figure import draw_metrics_panel
from scripts.figure_modules._style import apply_pub_style, save_figure, FIGSIZE, LW_SCALE, MARKER_SCALE


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    fig, ax = plt.subplots(figsize=FIGSIZE['methods_fitness'])
    ax.set_facecolor("#fffde8")
    for sp in ax.spines.values():
        sp.set_edgecolor("#c8c8c8")
        sp.set_linewidth(0.7)
    draw_metrics_panel(ax)
    fig.tight_layout()
    out = os.path.join(figures_dir, "fig_methods_fitness_metrics.pdf")
    return save_figure(fig, out)
