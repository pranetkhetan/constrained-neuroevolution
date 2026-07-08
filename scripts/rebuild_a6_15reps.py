#!/usr/bin/env python
"""Rebuild ``analysis/degeneracy_analyses/A6_results.pkl`` for the extended
generalist replicate set (n = 15) used in the referee-follow-up revision.

Why this exists
---------------
The paper's one positive claim -- specialists show ~2.2-2.6x higher functional
sensitivity variance than generalists -- carried a hierarchical-bootstrap 95% CI
of [1.02, 7.36] whose wide upper bound came from resampling **only 6 generalist
replicates**. We evolved 9 additional generalist replicates (r6-r14; identical
architecture/constraints, same seed family ``base_seed=42``), taking the axis
from n=6 to n=15. This script recomputes the *generalist* half of
``A6_results.pkl`` over all 15 replicates so ``build_paper_stats.py`` regenerates
a tighter CI.

The **specialist** half is unchanged and reused from shipped intermediates:
``spec_sens_var`` == ``var(source_sensitivity_results['sensitivity_matrix'])`` and
``spec_topo_sims`` / ``spec_own_fits`` are recovered from the existing A6 pkl
(the 54 specialist agents did not change).

Provenance
----------
The generalist topology / fitness-cost / sensitivity computations are ported
**verbatim** from the producer notebook ``notebooks/colab_15_degeneracy_analyses.ipynb``
(Cells 24-27): same fixed sensory sequence, same per-source permutation protocol,
``rng = np.random.default_rng(42)``, ``N_PERM_SENS = N_PERM_BASELINE = 20``. The
only change is the replicate count (6 -> 15) and the load path
(``data/generalist/results_r{r}`` instead of the notebook's stale
``results_generalist_r{r}``).

Requires a GPU/cupy environment: the evolved ``Agent`` objects pickle their
weights as CuPy arrays, and the fitness/cost step calls ``evaluate_batch``. Run
in the same Colab/venv used to evolve the agents::

    python scripts/rebuild_a6_15reps.py --n_reps 15

Output: ``analysis/degeneracy_analyses/A6_results.pkl`` (backed up first).
"""
from __future__ import annotations

import argparse
import copy
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import load_config
from core.simulation import Simulation
from core.fitness import evaluate_batch
from utils.backend import xp  # noqa: F401  (ensures backend initialised)

MICE = ['B5', 'B6', 'B7', 'D3', 'D4', 'D5', 'D7', 'D8', 'D9']
N_MICE = len(MICE)
GEN = 150
DATA = _ROOT / 'data'
ANALYSIS = _ROOT / 'analysis'
OUT_DIR = ANALYSIS / 'degeneracy_analyses'
SOURCE_NAMES = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5',
                'I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'M0', 'M1']


# ---------------------------------------------------------------------------
# Helpers (mirror run_generalist.py + colab_15 exactly)
# ---------------------------------------------------------------------------

def _to_numpy(w):
    return w.get() if hasattr(w, 'get') else np.asarray(w)


def _cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


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
    agents, weights, fits = [], [], []
    for r in range(n_reps):
        path = DATA / 'generalist' / f'results_r{r}' / f'gen_{GEN}' / 'summary.pkl'
        with open(path, 'rb') as f:
            results = pickle.load(f)
        best = min(results, key=lambda x: x['fitness'])
        agents.append(best['agent'])
        weights.append(_to_numpy(best['agent'].weights))
        fits.append(best['fitness'])
        print(f'  Generalist r{r}: fitness={best["fitness"]:.4f}')
    return agents, np.array(weights), fits


# ---- sensitivity protocol (colab_15 Cell 27, verbatim) --------------------

def _fixed_sequence(n_steps=1000, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10 * np.pi, n_steps)
    seq = np.column_stack([
        0.5 + 0.4 * np.sin(t * 0.7),
        0.3 + 0.3 * np.sin(t * 1.1 + 1),
        0.3 + 0.3 * np.cos(t * 1.1),
        0.2 + 0.2 * np.sin(t * 0.5),
        0.1 * np.sin(t * 0.9),
        0.05 * rng.standard_normal(n_steps),
    ])
    return seq, rng


def _run_agent_sequence(agent, seq):
    w = _to_numpy(agent.weights)
    state = np.zeros(14)
    outputs = []
    for s in seq:
        state[:6] = s
        state[6:] = np.tanh(state @ w)[6:]
        outputs.append(state[12:14].copy())
    return np.array(outputs)


def _permute_source_wiring(agent, src_idx, rng):
    a = copy.deepcopy(agent)
    a.weights = _to_numpy(a.weights)
    row = a.weights[src_idx].copy()
    nz = np.where(row != 0)[0]
    if len(nz) > 1:
        perm = rng.permutation(nz.copy())
        new_row = np.zeros_like(row)
        for old_t, new_t in zip(nz, perm):
            new_row[new_t] = row[old_t]
        a.weights[src_idx] = new_row
    return a


def _permute_all_sources(agent, rng):
    a = copy.deepcopy(agent)
    a.weights = _to_numpy(a.weights)
    for src in range(14):
        row = a.weights[src].copy()
        nz = np.where(row != 0)[0]
        if len(nz) > 1:
            perm = rng.permutation(nz.copy())
            new_row = np.zeros_like(row)
            for old_t, new_t in zip(nz, perm):
                new_row[new_t] = row[old_t]
            a.weights[src] = new_row
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_reps', type=int, default=15, help='generalist replicate count')
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a6_path = OUT_DIR / 'A6_results.pkl'

    # ---- specialist side: reuse shipped intermediates (unchanged) ----------
    SS = pickle.load(open(ANALYSIS / 'source_sensitivity_results.pkl', 'rb'))
    spec_sens_matrix = np.asarray(SS['sensitivity_matrix'], float)   # (54,14)
    spec_sens_var = spec_sens_matrix.var(axis=0)                      # (14,)

    old_A6 = pickle.load(open(a6_path, 'rb'))
    spec_topo_sims = np.asarray(old_A6['spec_topo_sims'], float)      # (135,) unchanged
    old_gmp = np.asarray(old_A6['gen_mean_per_mouse'], float)
    old_cost = np.asarray(old_A6['gen_cost_pct'], float)
    spec_own_fits = old_gmp / (1.0 + old_cost / 100.0)               # recovered specialist ref

    # ---- generalist side: recompute over n_reps ----------------------------
    print(f'Loading {args.n_reps} generalist agents...')
    gen_agents, W_gen, _ = _load_generalist_agents(args.n_reps)
    N_GEN = len(gen_agents)
    gen_topo = (W_gen != 0).astype(float).reshape(N_GEN, -1)

    # topology cosine sims within the generalist group
    gen_topo_sims = np.array([_cosine_sim(gen_topo[i], gen_topo[j])
                              for i in range(N_GEN) for j in range(i + 1, N_GEN)])
    from scipy.stats import mannwhitneyu
    _, mw_p_gen_spec_topo = mannwhitneyu(spec_topo_sims, gen_topo_sims,
                                         alternative='two-sided')

    # fitness / cost (evaluate_batch; GPU-backed)
    print('Evaluating generalist fitness across 9 mice...')
    config = load_config(args.config)
    simulation = Simulation(config.physics)
    noise_matrix = _generate_noise_matrix(config.simulation.n_bouts,
                                          config.simulation.max_frames)
    baselines = _load_all_mouse_baselines()
    A6_gen_fitness = np.full((N_GEN, N_MICE), np.nan)
    for j, m in enumerate(MICE):
        results = evaluate_batch(gen_agents, simulation, config, baselines[m], noise_matrix)
        A6_gen_fitness[:, j] = np.array([r.total for r in results])
    gen_mean_per_mouse = A6_gen_fitness.mean(axis=0)
    gen_cost_pct = (gen_mean_per_mouse - spec_own_fits) / spec_own_fits * 100.0

    # sensitivity (per-source ablation) + full-circuit normalisation
    print('Computing generalist sensitivity profiles...')
    seq, rng_sens = _fixed_sequence(1000, seed=42)
    N_PERM_SENS = 20
    gen_sensitivity = np.zeros((N_GEN, 14))
    for ai, agent in enumerate(gen_agents):
        base = _run_agent_sequence(agent, seq)
        for src in range(14):
            mses = []
            for _ in range(N_PERM_SENS):
                pa = _permute_source_wiring(agent, src, rng_sens)
                mses.append(np.mean((_run_agent_sequence(pa, seq) - base) ** 2))
            gen_sensitivity[ai, src] = np.mean(mses)
    gen_sens_var = gen_sensitivity.var(axis=0)

    N_PERM_BASELINE = 20
    gen_full_circuit_baselines = np.zeros(N_GEN)
    for ai, agent in enumerate(gen_agents):
        base = _run_agent_sequence(agent, seq)
        mses = []
        for _ in range(N_PERM_BASELINE):
            pa = _permute_all_sources(agent, rng_sens)
            mses.append(np.mean((_run_agent_sequence(pa, seq) - base) ** 2))
        gen_full_circuit_baselines[ai] = np.mean(mses)

    gen_sensitivity_norm = gen_sensitivity.copy().astype(float)
    for ai in range(N_GEN):
        b = gen_full_circuit_baselines[ai]
        gen_sensitivity_norm[ai] = gen_sensitivity_norm[ai] / b if b > 1e-12 else 0.0
    gen_sens_var_norm = gen_sensitivity_norm.var(axis=0)

    ratio = spec_sens_var.mean() / gen_sens_var_norm.mean() if gen_sens_var_norm.mean() > 0 else float('inf')
    print(f'\n  N_GEN = {N_GEN}')
    print(f'  Specialist normalised sens var (mean): {spec_sens_var.mean():.6f}')
    print(f'  Generalist normalised sens var (mean): {gen_sens_var_norm.mean():.6f}')
    print(f'  Variance ratio (spec/gen): {ratio:.2f}x')
    print(f'  Generalist cost (mean-of-ratios): {gen_cost_pct.mean():.1f}%'
          f'  (range {gen_cost_pct.min():.1f}-{gen_cost_pct.max():.1f}%)')

    A6_results = {
        'N_REPS_G': N_GEN,
        'W_gen': W_gen,
        'gen_topo_sims': gen_topo_sims,
        'spec_topo_sims': spec_topo_sims,
        'mw_p_gen_spec_topo': float(mw_p_gen_spec_topo),
        'gen_fitness_matrix': A6_gen_fitness,
        'gen_mean_per_mouse': gen_mean_per_mouse,
        'gen_cost_pct': gen_cost_pct,
        'gen_sensitivity': gen_sensitivity,
        'gen_sensitivity_norm': gen_sensitivity_norm,
        'gen_full_circuit_baselines': gen_full_circuit_baselines,
        'spec_sens_var': spec_sens_var,
        'gen_sens_var': gen_sens_var,
        'gen_sens_var_norm': gen_sens_var_norm,
    }

    backup = a6_path.with_suffix('.pkl.n6.bak')
    if not backup.exists():
        shutil.copy2(a6_path, backup)
        print(f'\n  Backed up original (n=6) -> {backup.name}')
    with open(a6_path, 'wb') as f:
        pickle.dump(A6_results, f)
    print(f'  Saved rebuilt A6_results.pkl (N_REPS_G={N_GEN}) -> {a6_path}')
    print('\nNext: python scripts/build_paper_stats.py  (after updating the'
          ' hardcoded n=6 bootstrap draw at build_paper_stats.py:997 to N_GEN).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
