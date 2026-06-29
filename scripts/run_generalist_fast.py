"""
Fast generalist neuroevolution training.

Uses core.evaluate_generalist.evaluate_all_mice() which combines all 9 mice into
a single simulation pass — 9× fewer Python frame-loop iterations, plus:
  - Sub-steps reduced 30 → 5
  - Zero GPU sync in step/frame loops
  - Vectorised metrics from fitness.py

Expected speed: ~60-90 s/gen on L4  (vs ~1200 s/gen with the old approach).
150 gens × 6 reps total ≈ 2-3 hours.

Usage (from colab cell):
    from run_generalist_fast import train_all_reps, load_all_mouse_baselines
    from config import load_config
    config = load_config('config.yaml')
    baselines = load_all_mouse_baselines()
    train_all_reps(config, PROJECT_DIR, baselines,
                   reps=[0,1,2,3,4,5], gen_end=150, base_seed=42)
"""
import os
import sys

# Make project root (config, utils, core) importable when imported/invoked
# from outside scripts/ (e.g. from a notebook doing
# `from run_generalist_fast import ...`).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pickle
import time
import numpy as np
from datetime import datetime

from config import load_config
from utils.backend import xp, free_memory, HAS_GPU, set_seed
from core.agent import Agent
from core.simulation import Simulation
from core.evolution import initialize_population, select_elites, create_next_generation
from core.fitness import FitnessResult
from core.evaluate_generalist import evaluate_all_mice

MICE = ['B5', 'B6', 'B7', 'D3', 'D4', 'D5', 'D7', 'D8', 'D9']


# ---------------------------------------------------------------------------
# Helpers shared with run_generalist.py
# ---------------------------------------------------------------------------

def load_all_mouse_baselines(data_dir=None):
    """Load behavioural baselines for all 9 mice."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    defaults = dict(node_pdf=np.zeros(128), revisit_rate=0.38,
                    straightness=0.8, turn_bias=0.5, markov_profile=None)
    all_baselines = {}
    for mid in MICE:
        bl = dict(defaults)
        path = os.path.join(data_dir, f'mouse_{mid}_metrics.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            bl.update({k: data.get(v, bl[k]) for k, v in [
                ('node_pdf', 'node_pdf'), ('revisit_rate', 'reversal_rate'),
                ('straightness', 'straightness'), ('turn_bias', 'turn_bias'),
                ('markov_profile', 'markov_profile'),
            ]})
            bl['physics'] = data.get('physics', {})
            print(f'  [{mid}] loaded')
        else:
            print(f'  [{mid}] WARNING: not found, using defaults')
        all_baselines[mid] = bl
    return all_baselines


def _generate_noise(n_bouts, max_frames):
    noise = np.zeros((n_bouts, max_frames), dtype=np.float64)
    for s in range(n_bouts):
        noise[s] = np.random.RandomState(s).uniform(-1, 1, size=max_frames)
    return xp.array(noise)


def _find_last_gen(output_dir, gen_end):
    for g in range(gen_end, 0, -1):
        if os.path.exists(os.path.join(output_dir, f'gen_{g}', 'summary.pkl')):
            return g
    return 0


def _mean_fitness(results_list):
    n = len(results_list)
    return FitnessResult(
        total=np.mean([r.total      for r in results_list]),
        markov=np.mean([r.markov    for r in results_list]),
        occupancy=np.mean([r.occupancy for r in results_list]),
        tortuosity=np.mean([r.tortuosity for r in results_list]),
        turn_bias=np.mean([r.turn_bias for r in results_list]),
    )


def _save_generation(population, fitness_results, generation, output_dir):
    gen_dir = os.path.join(output_dir, f'gen_{generation}')
    os.makedirs(gen_dir, exist_ok=True)
    results = [{'id': i+1, 'fitness': fr.total,
                 'markov_score': fr.markov, 'occupancy_score': fr.occupancy,
                 'tortuosity_score': fr.tortuosity, 'turn_bias_score': fr.turn_bias,
                 'agent': agent}
               for i, (agent, fr) in enumerate(zip(population, fitness_results))]
    with open(os.path.join(gen_dir, 'summary.pkl'), 'wb') as f:
        pickle.dump(results, f)
    meta = {'generation': generation, 'timestamp': datetime.now().isoformat(),
            'output_dir': output_dir, 'gpu_available': HAS_GPU,
            'best_fitness': min(r['fitness'] for r in results),
            'mean_fitness': np.mean([r['fitness'] for r in results]),
            'n_mice': len(MICE), 'mice': MICE}
    with open(os.path.join(gen_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    return results


def _load_generation(generation, output_dir):
    with open(os.path.join(output_dir, f'gen_{generation}', 'summary.pkl'), 'rb') as f:
        results = pickle.load(f)
    return [r['agent'] for r in results], [r['fitness'] for r in results]


# ---------------------------------------------------------------------------
# Core training functions
# ---------------------------------------------------------------------------

def train_one_gen(gen, rep_info, config, simulation, noise_matrix, all_baselines,
                  base_seed=42):
    """
    Run one generation for ALL active reps simultaneously.

    All reps' populations are concatenated → one evaluate_all_mice() call
    → metrics computed per (agent, mouse) pair → split back per rep and saved.
    """
    reps_this_gen = {rep: info for rep, info in rep_info.items()
                     if info['last_done'] < gen}
    if not reps_this_gen:
        return

    populations = {}

    # ── CPU: build each rep's population ─────────────────────────────────
    for rep, info in reps_this_gen.items():
        set_seed(base_seed + rep * 10000 + gen)
        if gen == 1:
            pop = initialize_population(config)
        else:
            prev_pop, prev_fit = _load_generation(gen - 1, info['output_dir'])
            elites = select_elites(prev_pop, prev_fit, config)
            pop = create_next_generation(elites, config.population.size, config)
        populations[rep] = pop

    # ── GPU: one combined evaluate_all_mice call ─────────────────────────
    combined_agents = []
    offsets = {}
    offset = 0
    for rep, pop in populations.items():
        combined_agents.extend(pop)
        offsets[rep] = (offset, len(pop))
        offset += len(pop)

    t0 = time.time()
    per_mouse = evaluate_all_mice(combined_agents, simulation, config,
                                  all_baselines, noise_matrix)
    eval_sec = time.time() - t0

    # ── Split results per rep, average across mice, save ─────────────────
    best_per_rep = {}
    for rep, (start, size) in offsets.items():
        # Per-agent mean FitnessResult across all 9 mice
        generalist_fitness = [
            _mean_fitness([per_mouse[mid][start + i] for mid in MICE])
            for i in range(size)
        ]
        _save_generation(populations[rep], generalist_fitness, gen,
                         rep_info[rep]['output_dir'])
        rep_info[rep]['last_done'] = gen
        best_per_rep[rep] = min(fr.total for fr in generalist_fitness)

    # ── Progress line ─────────────────────────────────────────────────────
    rep_str = '  '.join(f'r{r}={best_per_rep[r]:.4f}' for r in sorted(best_per_rep))
    print(f'Gen {gen:3d} | {eval_sec:6.1f}s | {rep_str}', flush=True)
    free_memory()


def train_all_reps(config, project_dir, all_baselines,
                   reps=(0, 1, 2, 3, 4, 5), gen_end=150, base_seed=42):
    """
    Train generalist agents for all specified replicates.

    Skips reps that have already completed gen_end.
    Resumes from last completed generation per rep.

    Parameters
    ----------
    config      : Config object from load_config()
    project_dir : root of the project (same as PROJECT_DIR in colab)
    all_baselines : from load_all_mouse_baselines()
    reps        : list of rep indices to run
    gen_end     : final generation (default 150)
    base_seed   : base random seed; rep r uses base_seed + r*10000 + gen
    """
    print(f'GPU available: {HAS_GPU}')
    print(f'Population size: {config.population.size}')
    print(f'Reps to run: {list(reps)}')
    print(f'Generations: 1 → {gen_end}')
    print()

    simulation   = Simulation(config.physics)
    noise_matrix = _generate_noise(config.simulation.n_bouts,
                                   config.simulation.max_frames)

    # Build rep_info dict; skip already-complete reps
    rep_info = {}
    for rep in reps:
        out_dir   = os.path.join(project_dir, 'data', 'generalist', f'results_r{rep}')
        last_done = _find_last_gen(out_dir, gen_end)
        if last_done >= gen_end:
            print(f'Rep {rep}: already complete at gen {gen_end}, skipping.')
            continue
        rep_info[rep] = {'output_dir': out_dir, 'last_done': last_done}
        if last_done > 0:
            print(f'Rep {rep}: resuming from gen {last_done}')

    if not rep_info:
        print('All reps already complete.')
        return

    total_start = time.time()

    for gen in range(1, gen_end + 1):
        active = {r: info for r, info in rep_info.items() if info['last_done'] < gen}
        if not active:
            continue   # this gen already done by all reps — skip, don't break
        train_one_gen(gen, rep_info, config, simulation, noise_matrix,
                      all_baselines, base_seed=base_seed)

    total_min = (time.time() - total_start) / 60
    print(f'\nDone. Total time: {total_min:.1f} min')

    # Summary
    print('\n=== Completion check ===')
    for rep in reps:
        out_dir = os.path.join(project_dir, 'data', 'generalist', f'results_r{rep}')
        done = _find_last_gen(out_dir, gen_end)
        status = 'DONE' if done >= gen_end else f'partial (gen {done})'
        print(f'  Rep {rep}: {status}')
