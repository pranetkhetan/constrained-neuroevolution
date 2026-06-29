"""
Figure 2 (fig:convergence) — Per-mouse evolutionary convergence curves.
Output: fig1_permouse.pdf
"""

import os
import sys
import matplotlib.pyplot as plt

# Ensure project root is importable
_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figures_logic import figure1_permouse


def generate(store, figures_dir: str) -> list[str]:
    mouse_data = store.mouse_data()
    out = os.path.join(figures_dir, 'fig1_permouse.pdf')
    fig = figure1_permouse(mouse_data, save_path=out)
    plt.close(fig)
    return [out]
