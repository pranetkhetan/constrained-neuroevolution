"""
extract_weight_data.py — One-time extraction of weight matrices for figure generation.

Produces analysis/weight_data.pkl containing:
  - weight_matrices: (54, 14, 14) full weight matrices
  - weight_vectors: (54, 196) flattened for cosine similarity
  - mice: list of 54 mouse IDs
  - reps: list of 54 rep numbers
  - fitnesses: list of 54 best fitness values
  - non_zero_mask: (14, 14) bool union of all non-zero positions
  - random_weight_vectors: (54, 196) from random constrained agents
  - best_agent: the single best overall agent object (for Fig 4 panel A)

Run once:
    python scripts/extract_weight_data.py
"""

import os
import sys
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config import load_config, NetworkConfig
from core.agent import Agent

MICE = ['B5', 'B6', 'B7', 'D3', 'D4', 'D5', 'D7', 'D8', 'D9']
N_REPS = 6
GEN = 150
_cfg = load_config(os.path.join(PROJECT_DIR, "config.yaml"))
_NETWORK_CONFIG = _cfg.network


class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('cupy'):
            module = module.replace('cupy._core.core', 'numpy').replace('cupy', 'numpy')
        return super().find_class(module, name)


def _load(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        with open(path, 'rb') as f:
            return _CpuUnpickler(f).load()


def _load_one(task):
    """Load best agent from one run. Called in thread pool."""
    mouse, rep, agents_dir = task
    summary_path = os.path.join(
        agents_dir, f'results_{mouse}_r{rep}', f'gen_{GEN}', 'summary.pkl')
    if not os.path.exists(summary_path):
        print(f"  WARNING: {summary_path} not found -- skipping")
        return None

    pop = _load(summary_path)
    best = min(pop, key=lambda r: r['fitness'])
    agent = best['agent']
    w = agent.weights
    if hasattr(w, 'get'):
        w = w.get()
    w = np.array(w, dtype=np.float64)
    print(f"  {mouse} r{rep}: fitness={best['fitness']:.4f}")
    return {'mouse': mouse, 'rep': rep, 'w': w, 'fitness': best['fitness'], 'agent': agent}


def main():
    agents_dir = os.path.join(PROJECT_DIR, 'data', 'agents')
    out_path = os.path.join(PROJECT_DIR, 'analysis', 'weight_data.pkl')

    tasks = [(m, r, agents_dir) for m in MICE for r in range(1, N_REPS + 1)]

    print(f"Loading {len(tasks)} best agents (8 threads)...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        raw = list(pool.map(_load_one, tasks))

    raw = [r for r in raw if r is not None]
    raw.sort(key=lambda r: (r['mouse'], r['rep']))

    weight_matrices = np.array([r['w'] for r in raw])
    weight_vectors = np.array([r['w'].flatten() for r in raw])
    mice_list = [r['mouse'] for r in raw]
    reps_list = [r['rep'] for r in raw]
    fitnesses = [r['fitness'] for r in raw]

    # Non-zero mask: union of all non-zero positions
    non_zero_mask = np.any(weight_matrices != 0, axis=0)

    # Best overall agent (for Fig 4 network panel)
    best_idx = int(np.argmin(fitnesses))
    best_agent = raw[best_idx]['agent']
    print(f"\nBest overall agent: {mice_list[best_idx]} r{reps_list[best_idx]}, "
          f"fitness={fitnesses[best_idx]:.4f}")

    # Generate 54 random constrained agents (CPU, deterministic seeds)
    print("Generating 54 random constrained agents...")
    random_vectors = []
    for i in range(54):
        np.random.seed(i * 31 + 7)
        a = Agent(_NETWORK_CONFIG)
        w = a.weights
        if hasattr(w, 'get'):
            w = w.get()
        random_vectors.append(np.array(w, dtype=np.float64).flatten())
    random_weight_vectors = np.array(random_vectors)

    result = {
        'weight_matrices': weight_matrices,
        'weight_vectors': weight_vectors,
        'mice': mice_list,
        'reps': reps_list,
        'fitnesses': fitnesses,
        'non_zero_mask': non_zero_mask,
        'random_weight_vectors': random_weight_vectors,
        'best_agent': best_agent,
    }

    with open(out_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"\nSaved: {out_path}")
    print(f"  Evolved: {weight_matrices.shape}")
    print(f"  Random:  {random_weight_vectors.shape}")
    print(f"  Non-zero positions: {non_zero_mask.sum()} / {non_zero_mask.size}")


if __name__ == '__main__':
    main()
