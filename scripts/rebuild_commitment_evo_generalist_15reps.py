#!/usr/bin/env python
"""Recompute the *generalist* sensitivity-variance trajectory in
``analysis/degeneracy_analyses/sens_commitment_evolution.pkl`` over the extended
15-replicate generalist set, for Panel D of ``fig:sensitivity_commitment``.

Why only the generalist half
-----------------------------
The commitment-evolution pkl feeds two things in the paper:
  * Panel D of the sensitivity figure  -> ``gen_var_mean_traj`` (generalist line)
    and ``spec_var_mean_traj`` (specialist line);
  * the ``\statClaimTwentyEight*`` macros -> computed in ``build_paper_stats.py``
    from ``spec_sens_all[-1]`` ONLY (the 54 specialist agents at gen 150).

The 54 specialists are unchanged, so the specialist trajectory and every
Claim-28 macro are byte-identical whether or not we touch this pkl. Only the
generalist trajectory (Panel D's dashed line) depends on the replicate count.
This script therefore recomputes **only** ``gen_sens_all`` / ``gen_var_trajectory``
/ ``gen_var_mean_traj`` over the 15 reps and leaves every specialist array (and
the within/between similarity arrays) exactly as shipped.

Provenance
----------
Sensitivity protocol ported verbatim from the producer notebook
``notebooks/colab_18_sensitivity_commitment_evolution.ipynb`` (Cells 3, 6): raw
(un-normalised) per-source ablation MSE, ``N_PERM=20``, per-generation seed
``SEED_SENS + gen + 1000`` (the generalist offset the notebook uses). The only
changes are the replicate count (6 -> 15) and the load path
(``results_r{r}`` instead of the notebook's stale ``results_generalist_r{r}``).

CPU-only: the evolved agents pickle their weights as CuPy arrays, handled here by
a CuPy->NumPy unpickling shim, and the forward pass is pure NumPy. Runs locally::

    python scripts/rebuild_commitment_evo_generalist_15reps.py --n_reps 15

The original pkl is backed up to ``sens_commitment_evolution.pkl.n6.bak``.
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
DATA = _ROOT / 'data'
DEGEN = _ROOT / 'analysis' / 'degeneracy_analyses'

SEED_SENS = 42
N_PERM = 20
N_STEPS = 1000


class _CpuUnpickler(pickle.Unpickler):
    """Remap CuPy arrays to NumPy on CPU-only machines (matches nb18 Cell 3)."""

    def find_class(self, mod, name):
        # Remap CuPy array classes to NumPy; leave project modules (core.agent) intact.
        if mod.startswith('cupy'):
            mod = mod.replace('cupy._core.core', 'numpy').replace('cupy', 'numpy')
        return super().find_class(mod, name)


def _load_cpu(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        with open(path, 'rb') as f:
            return _CpuUnpickler(f).load()


def _fixed_sequence(n_steps=N_STEPS, seed=SEED_SENS):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10 * np.pi, n_steps)
    return np.column_stack([
        0.5 + 0.4 * np.sin(t * 0.7),
        0.3 + 0.3 * np.sin(t * 1.1 + 1),
        0.3 + 0.3 * np.cos(t * 1.1),
        0.2 + 0.2 * np.sin(t * 0.5),
        0.1 * np.sin(t * 0.9),
        0.05 * rng.standard_normal(n_steps),
    ])


def _run_agent_sequence(agent, seq):
    w = agent.weights.get() if hasattr(agent.weights, 'get') else np.asarray(agent.weights)
    state = np.zeros(14)
    out = []
    for s in seq:
        state[:6] = s
        state[6:] = np.tanh(state @ w)[6:]
        out.append(state[12:14].copy())
    return np.array(out)


def _permute_source_wiring(agent, src_idx, rng):
    a = copy.deepcopy(agent)
    a.weights = a.weights.get() if hasattr(a.weights, 'get') else np.asarray(a.weights)
    row = a.weights[src_idx].copy()
    nz = np.where(row != 0)[0]
    if len(nz) > 1:
        perm = rng.permutation(len(nz))
        new_row = np.zeros_like(row)
        for old_pos, new_pos in enumerate(perm):
            new_row[nz[new_pos]] = row[nz[old_pos]]
        a.weights[src_idx] = new_row
    return a


def _compute_sensitivity_matrix(agents, seq, n_perm, seed):
    rng = np.random.default_rng(seed)
    sens = np.zeros((len(agents), 14))
    for ai, agent in enumerate(agents):
        if agent is None:
            sens[ai, :] = np.nan
            continue
        base = _run_agent_sequence(agent, seq)
        for src in range(14):
            mses = []
            for _ in range(n_perm):
                pa = _permute_source_wiring(agent, src, rng)
                mses.append(np.mean((_run_agent_sequence(pa, seq) - base) ** 2))
            sens[ai, src] = np.mean(mses)
    return sens


def _load_best_generalist(rep, gen):
    path = DATA / 'generalist' / f'results_r{rep}' / f'gen_{gen}' / 'summary.pkl'
    if not path.exists():
        return None
    pop = _load_cpu(path)
    return min(pop, key=lambda x: x['fitness'])['agent']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_reps', type=int, default=15)
    args = ap.parse_args()

    pkl_path = DEGEN / 'sens_commitment_evolution.pkl'
    data = _load_cpu(pkl_path)
    sample_gens = list(data['sample_gens'])
    spec_var_traj = np.asarray(data['spec_var_trajectory'])       # (16,14) unchanged
    spec_var_mean_traj = np.asarray(data['spec_var_mean_traj'])   # (16,) unchanged

    seq = _fixed_sequence()
    gen_sens_all = []
    for gi, gen in enumerate(sample_gens):
        agents = [_load_best_generalist(r, gen) for r in range(args.n_reps)]
        n_ok = sum(a is not None for a in agents)
        gsens = _compute_sensitivity_matrix(agents, seq, N_PERM, seed=SEED_SENS + gen + 1000)
        gen_sens_all.append(gsens)
        print(f'  gen {gen:3d}: {n_ok}/{args.n_reps} generalists loaded, '
              f'var={np.nanvar(gsens, axis=0).mean():.6f}')

    gen_var_trajectory = np.array([np.nanvar(g, axis=0) for g in gen_sens_all])  # (16,14)
    gen_var_mean_traj = gen_var_trajectory.mean(axis=1)                          # (16,)

    print(f'\n  spec var @gen150: {spec_var_mean_traj[-1]:.6f}')
    print(f'  gen  var @gen150: {gen_var_mean_traj[-1]:.6f}  (was {np.asarray(data["gen_var_mean_traj"])[-1]:.6f} at n=6)')
    print(f'  ratio spec/gen @gen150: {spec_var_mean_traj[-1] / gen_var_mean_traj[-1]:.1f}x')

    # Update ONLY the generalist arrays; keep specialist + similarity arrays intact.
    data['gen_sens_all'] = gen_sens_all
    data['gen_var_trajectory'] = gen_var_trajectory
    data['gen_var_mean_traj'] = gen_var_mean_traj
    data['N_GEN_REPS'] = args.n_reps

    backup = pkl_path.with_suffix('.pkl.n6.bak')
    if not backup.exists():
        shutil.copy2(pkl_path, backup)
        print(f'\n  Backed up original -> {backup.name}')
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'  Saved rebuilt sens_commitment_evolution.pkl (generalist n={args.n_reps}).')
    print('  (Specialist trajectory + Claim-28 arrays unchanged.)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
