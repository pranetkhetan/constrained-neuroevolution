"""
Supplementary Figure S1 (fig:supp_permouse) — Per-mouse evolutionary trajectories.
Outputs: fig_s1_B5.png, fig_s1_B6.png, ..., fig_s1_D9.png (9 files)
"""

import os
import sys
import matplotlib.pyplot as plt

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figures_logic import figure1_supplement


def generate(store, figures_dir: str) -> list[str]:
    mouse_data = store.mouse_data()
    prefix = os.path.join(figures_dir, 'fig_s1')
    figs = figure1_supplement(mouse_data, save_path_prefix=prefix)
    outputs = []
    for fig in figs:
        plt.close(fig)
    for mouse_id in sorted(mouse_data):
        path = f"{prefix}_{mouse_id}.png"
        if os.path.exists(path):
            outputs.append(path)
    return outputs
