"""
Figure 3 (fig:emergent) — Emergent behavioral properties (per-mouse).
Output: fig2_permouse.pdf
"""

import os
import sys
import matplotlib.pyplot as plt

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figures_logic import figure2_permouse


def generate(store, figures_dir: str) -> list[str]:
    emdata = store.emergent_data_permouse()
    out = os.path.join(figures_dir, 'fig2_permouse.pdf')
    fig = figure2_permouse(emdata, save_path=out)
    plt.close(fig)
    return [out]
