"""
Figure 4 (fig:architecture) — E/I balance dynamics and structural convergence.
Output: fig3_network_ei.pdf
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from scripts.figures_logic import figure3_new


def generate(store, figures_dir: str) -> list[str]:
    wd = store.weight_data()
    weight_vectors = np.asarray(wd['weight_vectors'])  # (54, 196)
    mice_list = wd['mice']

    ei_inter = store.ei_cache_inter()
    ei_speed = store.ei_cache_speed()
    ei_turn  = store.ei_cache_turn()

    # Per-neuron E/I matrix: (54, 8) — non-sensory neurons 6-13
    W_all = weight_vectors.reshape(len(mice_list), 14, 14)
    neuron_targets = np.arange(6, 14)
    ei_matrix = np.zeros((len(mice_list), 8))
    for a_idx in range(len(mice_list)):
        W = W_all[a_idx]
        for n, target in enumerate(neuron_targets):
            incoming = W[:, target]
            exc = float(incoming[incoming > 0].sum())
            inh = float(abs(incoming[incoming < 0].sum()))
            total = exc + inh
            ei_matrix[a_idx, n] = exc / total if total > 0 else 0.5

    # Cosine heatmap removed from this figure — now lives in Figure 5A.
    out = os.path.join(figures_dir, 'fig3_network_ei.pdf')
    fig = figure3_new(
        ei_inter, ei_speed, ei_turn,
        ei_neuron_matrix=ei_matrix,
        mice_list=mice_list,
        save_path=out,
    )
    plt.close(fig)
    return [out]
