#!/usr/bin/env python
"""Rebuild ``analysis/generalist_results.pkl`` (generalist permutation-specificity
control) over the extended 15-replicate generalist set.

Context
-------
This pkl backs the generalist "flat delta-fit" control: Figure 4D, Supplementary
Figure S15 (``fig:supp_generalist_formal``), and the macro
``\statGeneralistKruskalP`` (Kruskal-Wallis across 9 test mice; expected n.s. =
no own-mouse bias, the negative control for the specialist specialization result).
It was computed on the original 6 generalist replicates; with the referee-follow-up
extension to 15 replicates, this control is regenerated at n=15 so every generalist
figure/number in the paper uses the same replicate set.

Provenance
----------
Ported verbatim from the producer notebook ``notebooks/colab_11_generalist.ipynb``
Section C (Cells C1, C3, C4, C5, C8): ``permute_agent_wiring_v2`` (source-preserving
topology permutation, seed ``SEED_C=999``), ``N_PERM_C=10`` permuted variants per
agent, evaluated against all 9 mice with the batched ``evaluate_batch``. The only
change is the replicate count (6 -> 15).

Requires a GPU/cupy environment (``evaluate_batch`` runs the simulation). Run in the
same Colab/venv used to evolve the agents::

    python scripts/rebuild_generalist_permutation_15reps.py --n_reps 15

The original pkl is backed up to ``generalist_results.pkl.n6.bak``.
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import load_config
from core.simulation import Simulation
from core.fitness import evaluate_batch
from utils.backend import xp

MICE = ['B5', 'B6', 'B7', 'D3', 'D4', 'D5', 'D7', 'D8', 'D9']
GEN = 150
N_PERM_C = 10
SEED_C = 999
DATA = _ROOT / 'data'
ANALYSIS = _ROOT / 'analysis'


def _get_weights(agent):
    W = agent.weights
    return np.array(W.get() if hasattr(W, 'get') else W, dtype=np.float64)


def permute_agent_wiring_v2(agent, rng):
    """Source-preserving topology permutation (colab_11 Cell A1, verbatim).

    Keeps per-source out-degree fixed, permutes which targets each source connects
    to, shuffles magnitudes within a source, preserves Dale sign. Preserves all 18
    structural features.
    """
    W = _get_weights(agent).copy()
    nt = np.asarray(agent.node_types)
    idx_s = np.array(agent.idx_sensory)
    idx_i = np.array(agent.idx_inter)
    idx_m = np.array(agent.idx_motor)
    W_perm = W.copy()

    pathway_blocks = [
        (idx_s, idx_i), (idx_s, idx_m), (idx_i, idx_i),
        (idx_i, idx_m), (idx_m, idx_i), (idx_m, idx_m),
    ]
    for src_global, tgt_global in pathway_blocks:
        n_tgt = len(tgt_global)
        sub = W[np.ix_(src_global, tgt_global)]
        sub_new = np.zeros_like(sub)
        for src_local_idx in range(len(src_global)):
            src_global_idx = src_global[src_local_idx]
            src_row = sub[src_local_idx]
            connected_cols = np.nonzero(src_row)[0]
            if len(connected_cols) == 0:
                continue
            mags = np.abs(src_row[connected_cols])
            n_conn = len(connected_cols)
            new_cols = rng.choice(n_tgt, size=n_conn, replace=False)
            mags_shuffled = rng.permutation(mags)
            sign = int(np.sign(nt[src_global_idx])) if nt[src_global_idx] != 0 else 1
            for nc, mag in zip(new_cols, mags_shuffled):
                sub_new[src_local_idx, nc] = mag * sign
        W_perm[np.ix_(src_global, tgt_global)] = sub_new

    class PermutedAgent:
        def __init__(self, weights):
            self.weights = weights
            self.node_types = nt
            self.idx_sensory = idx_s
            self.idx_inter = idx_i
            self.idx_motor = idx_m

    return PermutedAgent(W_perm)


def _generate_noise_matrix(n_bouts, max_frames):
    noise = np.zeros((n_bouts, max_frames), dtype=np.float64)
    for s in range(n_bouts):
        noise[s] = np.random.RandomState(s).uniform(-1, 1, size=max_frames)
    return xp.array(noise)


def _load_all_mouse_baselines():
    defaults = {'node_pdf': np.zeros(128), 'revisit_rate': 0.38,
                'straightness': 0.8, 'turn_bias': 0.5, 'markov_profile': None}
    out = {}
    for mid in MICE:
        bl = dict(defaults)
        path = DATA / f'mouse_{mid}_metrics.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                d = pickle.load(f)
            bl['node_pdf'] = d.get('node_pdf', bl['node_pdf'])
            bl['revisit_rate'] = d.get('reversal_rate', bl['revisit_rate'])
            bl['straightness'] = d.get('straightness', bl['straightness'])
            bl['turn_bias'] = d.get('turn_bias', bl['turn_bias'])
            bl['markov_profile'] = d.get('markov_profile', bl['markov_profile'])
            bl['physics'] = d.get('physics', {})
        out[mid] = bl
    return out


def _load_generalist_agents(n_reps):
    agents, fits = [], []
    for r in range(n_reps):
        path = DATA / 'generalist' / f'results_r{r}' / f'gen_{GEN}' / 'summary.pkl'
        with open(path, 'rb') as f:
            results = pickle.load(f)
        best = min(results, key=lambda x: x['fitness'])
        agents.append(best['agent'])
        fits.append(float(best['fitness']))
        print(f'  Rep {r}: best mean-across-9-mice fitness = {best["fitness"]:.4f}')
    return agents, fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_reps', type=int, default=15)
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()

    pkl_path = ANALYSIS / 'generalist_results.pkl'

    print(f'Loading {args.n_reps} generalist agents...')
    generalist_agents, generalist_fitness = _load_generalist_agents(args.n_reps)

    # C3: permuted variants
    rng_c = np.random.RandomState(SEED_C)
    generalist_batches = [
        [agent] + [permute_agent_wiring_v2(agent, rng_c) for _ in range(N_PERM_C)]
        for agent in generalist_agents
    ]

    # C4: evaluate (GPU)
    config = load_config(args.config)
    simulation = Simulation(config.physics)
    noise_matrix = _generate_noise_matrix(config.simulation.n_bouts, config.simulation.max_frames)
    baselines = _load_all_mouse_baselines()

    results_C = [{} for _ in range(len(generalist_agents))]
    for rep_idx, batch in enumerate(generalist_batches):
        for test_mouse in MICE:
            res = evaluate_batch(batch, simulation, config, baselines[test_mouse], noise_matrix)
            orig_fit = res[0].total
            perm_fits = [r.total for r in res[1:]]
            results_C[rep_idx][test_mouse] = {
                'orig_fitness': float(orig_fit),
                'perm_fitnesses': [float(x) for x in perm_fits],
                'delta_fit': float(np.mean(perm_fits) - orig_fit),
            }
        print(f'  Rep {rep_idx}: delta-fit computed across 9 mice')

    # C5: flatness stats
    delta_profiles = np.array([[results_C[r][m]['delta_fit'] for m in MICE]
                               for r in range(len(generalist_agents))])  # (n,9)
    groups = [delta_profiles[:, j].tolist() for j in range(len(MICE))]
    F_gen, p_gen = stats.f_oneway(*groups)
    H_gen, p_kw_gen = stats.kruskal(*groups)
    gen_mean_per_mouse = delta_profiles.mean(axis=0)
    gen_grand_mean = float(gen_mean_per_mouse.mean())
    gen_cv_pooled = float((gen_mean_per_mouse.std() / gen_grand_mean) * 100)

    print(f'\n  n generalist replicates: {len(generalist_agents)}')
    print(f'  Kruskal-Wallis across 9 mice: H={H_gen:.3f}, p={p_kw_gen:.4f}  (was p=0.093 at n=6)')
    print(f'  ANOVA: F={F_gen:.3f}, p={p_gen:.4f}')
    print(f'  grand mean delta-fit={gen_grand_mean:.4f}, CV={gen_cv_pooled:.1f}%')
    print(f'  Expected: still non-significant (flat profile = no own-mouse bias).')

    out = {
        'results_C': results_C,
        'delta_profiles': delta_profiles.tolist(),
        'gen_mean_per_mouse': gen_mean_per_mouse.tolist(),
        'gen_grand_mean': gen_grand_mean,
        'gen_cv_pooled': gen_cv_pooled,
        'anova': {'F': float(F_gen), 'p': float(p_gen)},
        'kruskal': {'H': float(H_gen), 'p': float(p_kw_gen)},
        'generalist_fitness': generalist_fitness,
        'N_PERM_C': N_PERM_C,
        'MICE': MICE,
        'N_REPS_G': len(generalist_agents),
    }

    backup = pkl_path.with_suffix('.pkl.n6.bak')
    if not backup.exists():
        shutil.copy2(pkl_path, backup)
        print(f'\n  Backed up original -> {backup.name}')
    with open(pkl_path, 'wb') as f:
        pickle.dump(out, f)
    print(f'  Saved rebuilt generalist_results.pkl (N_REPS_G={len(generalist_agents)}).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
