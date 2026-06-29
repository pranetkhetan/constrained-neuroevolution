#!/usr/bin/env python
"""
E/I ratio analysis across generations for evolved agents.

Computes weighted excitatory/inhibitory input ratios for specific target
neurons (speed motor, turn motor, all non-sensory) across all generations
of per-mouse evolution runs.

Usage:
    # As library
    from ei_analysis import compute_ei_timeseries, load_best_overall_agent

    # Standalone: pre-compute and cache all E/I timeseries
    python ei_analysis.py --base-dir results --cache-dir figures
"""
import os
import re
import sys
import pickle
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)


class _CpuUnpickler(pickle.Unpickler):
    """Unpickler that maps CuPy arrays to NumPy."""
    def find_class(self, module, name):
        if module.startswith('cupy'):
            module = module.replace('cupy._core.core', 'numpy').replace('cupy', 'numpy')
        return super().find_class(module, name)


def _load_pickle_cpu(path):
    """Load a pickle file, converting CuPy arrays to NumPy if needed."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        with open(path, 'rb') as f:
            return _CpuUnpickler(f).load()


def _to_numpy(arr):
    """Ensure array is numpy (handles cupy arrays)."""
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def compute_ei_ratio(agent, target_indices):
    """
    Compute E/I balance index for incoming connections to target neurons.

    Returns E / (E + I), a bounded metric in [0, 1]:
      0.5 = balanced, 1.0 = pure excitatory, 0.0 = pure inhibitory.

    Due to Dale's law, positive weights come from excitatory sources,
    negative from inhibitory sources.

    Args:
        agent: Agent object with .weights attribute
        target_indices: array-like of target neuron indices

    Returns:
        float: E/I balance index in [0, 1]
    """
    W = _to_numpy(agent.weights)
    target_indices = np.atleast_1d(target_indices)
    incoming = W[:, target_indices].ravel()
    exc_sum = float(np.sum(incoming[incoming > 0]))
    inh_sum = float(np.abs(np.sum(incoming[incoming < 0])))
    total = exc_sum + inh_sum
    if total == 0:
        return 0.5  # no connections = neutral
    return exc_sum / total


def discover_runs(base_dir):
    """
    Discover per-mouse run directories.

    Returns:
        dict: {mouse_id: {rep_num: full_path}}
    """
    runs = defaultdict(dict)
    pattern = re.compile(r'^results_([A-Z]\d+)_r(\d+)$')
    for name in os.listdir(base_dir):
        m = pattern.match(name)
        if m:
            mouse_id, rep = m.group(1), int(m.group(2))
            full_path = os.path.join(base_dir, name)
            if os.path.isdir(full_path):
                runs[mouse_id][rep] = full_path
    return dict(runs)


def compute_ei_timeseries(base_dir, target_indices, elite_frac=0.1,
                          all_agents=False, cache_path=None):
    """
    Compute E/I ratio timeseries across generations for all runs.

    Args:
        base_dir: directory containing results_{mouse}_r{rep}/ dirs
        target_indices: neuron indices to compute E/I for
        elite_frac: fraction of top agents to use (by fitness)
        all_agents: if True, use all agents instead of elites
        cache_path: if set, cache results to this pickle path

    Returns:
        dict: {mouse_id: {rep_num: {'generations': list, 'ei_ratios': list}}}
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        print(f"  Loaded cached E/I data from {cache_path}")
        return cached

    runs = discover_runs(base_dir)
    target_indices = np.atleast_1d(target_indices)
    result = {}

    for mouse_id in sorted(runs):
        result[mouse_id] = {}
        for rep in sorted(runs[mouse_id]):
            run_dir = runs[mouse_id][rep]
            # Find all generations
            gen_dirs = [d for d in os.listdir(run_dir) if d.startswith('gen_')]
            gen_nums = sorted(int(d.split('_')[1]) for d in gen_dirs)

            generations = []
            ei_ratios = []

            for g in gen_nums:
                summary_path = os.path.join(run_dir, f'gen_{g}', 'summary.pkl')
                if not os.path.exists(summary_path):
                    continue

                pop = _load_pickle_cpu(summary_path)
                fitnesses = np.array([ind['fitness'] for ind in pop])

                # Select agents
                if all_agents:
                    selected = pop
                else:
                    n_elite = max(1, int(len(pop) * elite_frac))
                    elite_idx = np.argsort(fitnesses)[:n_elite]
                    selected = [pop[i] for i in elite_idx]

                # Compute E/I for each selected agent
                ei_vals = []
                for ind in selected:
                    ei = compute_ei_ratio(ind['agent'], target_indices)
                    ei_vals.append(ei)

                generations.append(g)
                ei_ratios.append(float(np.mean(ei_vals)))

            result[mouse_id][rep] = {
                'generations': generations,
                'ei_ratios': ei_ratios,
            }
            print(f"  {mouse_id}/r{rep}: {len(generations)} generations")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"  Cached E/I data to {cache_path}")

    return result


def load_best_overall_agent(base_dir, gen=None):
    """
    Find the single best agent across all mice and replicates.

    Args:
        base_dir: directory containing results_{mouse}_r{rep}/ dirs
        gen: specific generation to load (None = final generation per run)

    Returns:
        (agent, metadata_dict) where metadata has mouse, rep, fitness, gen
    """
    runs = discover_runs(base_dir)
    best_agent = None
    best_fitness = float('inf')
    best_meta = {}

    for mouse_id in sorted(runs):
        for rep in sorted(runs[mouse_id]):
            run_dir = runs[mouse_id][rep]
            if gen is not None:
                target_gen = gen
            else:
                gen_dirs = [d for d in os.listdir(run_dir) if d.startswith('gen_')]
                target_gen = max(int(d.split('_')[1]) for d in gen_dirs)

            summary_path = os.path.join(run_dir, f'gen_{target_gen}', 'summary.pkl')
            if not os.path.exists(summary_path):
                continue

            pop = _load_pickle_cpu(summary_path)
            best = min(pop, key=lambda x: x['fitness'])
            if best['fitness'] < best_fitness:
                best_agent = best['agent']
                best_fitness = best['fitness']
                best_meta = {
                    'mouse': mouse_id, 'rep': rep,
                    'fitness': best_fitness, 'gen': target_gen,
                }

    print(f"  Best overall: {best_meta['mouse']}/r{best_meta['rep']} "
          f"gen={best_meta['gen']} fitness={best_fitness:.4f}")
    return best_agent, best_meta


def main():
    parser = argparse.ArgumentParser(description="Pre-compute E/I timeseries")
    parser.add_argument('--base-dir', default='data/agents',
                        help="Directory containing results_{mouse}_r{rep}/")
    parser.add_argument('--cache-dir', default='figures',
                        help="Directory to save cached results")
    parser.add_argument('--elite-frac', type=float, default=0.1,
                        help="Fraction of top agents to use")
    args = parser.parse_args()

    # Speed motor neuron (index 12 for 6S+6I+2M architecture)
    print("Computing E/I for speed motor neuron (idx 12)...")
    compute_ei_timeseries(
        args.base_dir, target_indices=[12],
        elite_frac=args.elite_frac,
        cache_path=os.path.join(args.cache_dir, 'ei_cache_speed.pkl'),
    )

    # Turn motor neuron (index 13)
    print("\nComputing E/I for turn motor neuron (idx 13)...")
    compute_ei_timeseries(
        args.base_dir, target_indices=[13],
        elite_frac=args.elite_frac,
        cache_path=os.path.join(args.cache_dir, 'ei_cache_turn.pkl'),
    )

    # All non-sensory (indices 6-13)
    print("\nComputing E/I for all non-sensory neurons (idx 6-13)...")
    compute_ei_timeseries(
        args.base_dir, target_indices=np.arange(6, 14),
        elite_frac=args.elite_frac,
        cache_path=os.path.join(args.cache_dir, 'ei_cache_all.pkl'),
    )

    # Interneurons only (indices 6-11)
    print("\nComputing E/I for interneurons only (idx 6-11)...")
    compute_ei_timeseries(
        args.base_dir, target_indices=np.arange(6, 12),
        elite_frac=args.elite_frac,
        cache_path=os.path.join(args.cache_dir, 'ei_cache_inter.pkl'),
    )

    print("\nDone. Cached files in:", args.cache_dir)


if __name__ == '__main__':
    main()
