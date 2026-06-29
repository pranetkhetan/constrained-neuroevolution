#!/usr/bin/env python
"""
Circuit feature extraction for evolved agents.

Extracts 18 structural features from agent weight matrices and connectivity.
Supports loading best agents from completed runs, computing random baselines,
and cross-mouse generalization evaluation.

Usage:
    python analyze_circuits.py --dirs results_D3_r1 results_D3_r2 ... --gen 150
    python analyze_circuits.py --dirs results_1 results_2 ... --gen 200  # aggregate runs
"""
import argparse
import os
import pickle
import numpy as np
import csv
from collections import defaultdict


class _CpuUnpickler(pickle.Unpickler):
    """Unpickler that maps CuPy arrays to NumPy (for loading GPU-trained agents on CPU)."""
    def find_class(self, module, name):
        if module.startswith('cupy'):
            # Map cupy modules to numpy equivalents
            module = module.replace('cupy._core.core', 'numpy').replace('cupy', 'numpy')
        return super().find_class(module, name)

    def persistent_load(self, pid):
        # CuPy uses persistent_id for ndarray; convert to numpy
        return super().persistent_load(pid)


def _load_pickle_cpu(path):
    """Load a pickle file, converting CuPy arrays to NumPy if needed."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        # CuPy not available; retry with CPU unpickler
        with open(path, 'rb') as f:
            return _CpuUnpickler(f).load()


def load_best_agent(results_dir, gen):
    """Load the best agent from a completed run at a specific generation."""
    summary_path = os.path.join(results_dir, f'gen_{gen}', 'summary.pkl')
    results = _load_pickle_cpu(summary_path)
    # results is a list of dicts; find lowest fitness
    best = min(results, key=lambda r: r['fitness'])
    return best['agent'], best['fitness']


def load_meta(results_dir, gen):
    """Load meta.pkl for a run at a specific generation."""
    meta_path = os.path.join(results_dir, f'gen_{gen}', 'meta.pkl')
    with open(meta_path, 'rb') as f:
        return pickle.load(f)


def extract_features(agent):
    """
    Extract 18 circuit features from an agent.

    Uses agent's own idx_sensory, idx_inter, idx_motor attributes
    so it adapts to any architecture automatically.

    Returns:
        dict of feature_name -> float
    """
    W = agent.weights  # (n_total, n_total), float64
    nt = agent.node_types  # (n_total,), +1/-1

    idx_s = agent.idx_sensory  # e.g. [0..5]
    idx_i = agent.idx_inter    # e.g. [6..11]
    idx_m = agent.idx_motor    # e.g. [12, 13]

    n_total = len(nt)
    n_s = len(idx_s)
    n_i = len(idx_i)
    n_m = len(idx_m)

    # Connection mask: nonzero weights
    conn = (W != 0)

    # --- Global features ---
    n_connections = int(conn.sum())
    # Max possible connections (no self-loops, no incoming to sensory)
    max_possible = n_total * (n_total - 1) - n_s * n_total  # rough upper bound
    # More accurate: any neuron can send to inter+motor only
    max_possible = n_total * (n_i + n_m)
    density = n_connections / max(max_possible, 1)

    # E/I counts (interneurons + motor only, sensory are always excitatory)
    inter_motor = np.concatenate([idx_i, idx_m])
    n_exc = int((nt[inter_motor] > 0).sum())
    n_inh = int((nt[inter_motor] < 0).sum())
    ei_ratio = n_exc / max(n_inh, 1)

    # --- Pathway counts ---
    # W[i, j] = connection from i to j
    si_count = int(conn[np.ix_(idx_s, idx_i)].sum())   # sensory -> inter
    sm_count = int(conn[np.ix_(idx_s, idx_m)].sum())   # sensory -> motor (shortcut)
    ii_count = int(conn[np.ix_(idx_i, idx_i)].sum())   # inter -> inter (recurrence)
    im_count = int(conn[np.ix_(idx_i, idx_m)].sum())   # inter -> motor
    mi_count = int(conn[np.ix_(idx_m, idx_i)].sum())   # motor -> inter (efference copy)
    mm_count = int(conn[np.ix_(idx_m, idx_m)].sum())   # motor -> motor

    # --- Excitatory fractions per pathway ---
    def exc_frac(src_idx, dst_idx):
        """Fraction of connections in pathway that are excitatory."""
        sub_W = W[np.ix_(src_idx, dst_idx)]
        active = sub_W[sub_W != 0]
        if len(active) == 0:
            return 0.0
        return float((active > 0).sum()) / len(active)

    si_exc_frac = exc_frac(idx_s, idx_i)
    ii_exc_frac = exc_frac(idx_i, idx_i)
    im_exc_frac = exc_frac(idx_i, idx_m)

    # --- Interneuron degree stats ---
    in_degree = conn[:, idx_i].sum(axis=0)   # incoming to each interneuron
    out_degree = conn[idx_i, :].sum(axis=1)  # outgoing from each interneuron
    inter_in_mean = float(in_degree.mean()) if n_i > 0 else 0.0
    inter_out_mean = float(out_degree.mean()) if n_i > 0 else 0.0

    # --- Weight statistics ---
    active_weights = np.abs(W[W != 0])
    w_mean_mag = float(active_weights.mean()) if len(active_weights) > 0 else 0.0
    frac_strong = float((active_weights >= 0.99).sum()) / max(len(active_weights), 1)

    return {
        'n_connections': n_connections,
        'density': density,
        'ei_ratio': ei_ratio,
        'n_exc': n_exc,
        'n_inh': n_inh,
        'si_count': si_count,
        'sm_count': sm_count,
        'ii_count': ii_count,
        'im_count': im_count,
        'mi_count': mi_count,
        'mm_count': mm_count,
        'si_exc_frac': si_exc_frac,
        'ii_exc_frac': ii_exc_frac,
        'im_exc_frac': im_exc_frac,
        'inter_in_mean': inter_in_mean,
        'inter_out_mean': inter_out_mean,
        'w_mean_mag': w_mean_mag,
        'frac_strong': frac_strong,
    }


FEATURE_NAMES = list(extract_features.__code__.co_varnames)  # won't work; define explicitly
FEATURE_NAMES = [
    'n_connections', 'density', 'ei_ratio', 'n_exc', 'n_inh',
    'si_count', 'sm_count', 'ii_count', 'im_count', 'mi_count', 'mm_count',
    'si_exc_frac', 'ii_exc_frac', 'im_exc_frac',
    'inter_in_mean', 'inter_out_mean',
    'w_mean_mag', 'frac_strong',
]


def random_baseline(config, n=200):
    """Create n random agents and return mean feature dict."""
    from core.agent import Agent
    features_list = []
    for i in range(n):
        np.random.seed(i)
        agent = Agent(config.network, batch_size=1)
        features_list.append(extract_features(agent))

    # Average across random agents
    mean_features = {}
    for key in FEATURE_NAMES:
        mean_features[key] = np.mean([f[key] for f in features_list])
    return mean_features, features_list


def cross_evaluate(agents_by_mouse, mice, config):
    """
    Evaluate each mouse's best agent against all mice's baselines.

    Args:
        agents_by_mouse: dict of mouse_id -> agent (best agent for that mouse)
        mice: list of mouse IDs
        config: Config object

    Returns:
        (n_mice, n_mice) fitness matrix. Entry [i,j] = fitness of mouse_i's agent on mouse_j's baseline.
    """
    from core.simulation import Simulation
    from core.fitness import evaluate_batch
    from run import load_mouse_baselines, generate_noise_matrix

    n = len(mice)
    matrix = np.zeros((n, n))
    simulation = Simulation(config.physics)

    for j, test_mouse in enumerate(mice):
        baselines = load_mouse_baselines(test_mouse)
        noise_matrix = generate_noise_matrix(config.simulation.n_bouts, config.simulation.max_frames)

        for i, train_mouse in enumerate(mice):
            agent = agents_by_mouse[train_mouse]
            agent.reset()
            results = evaluate_batch(
                [agent], simulation, config, baselines, noise_matrix
            )
            matrix[i, j] = results[0].total

    return matrix


def main():
    parser = argparse.ArgumentParser(description="Extract circuit features from evolved agents")
    parser.add_argument('--dirs', nargs='+', required=True,
                        help="Result directories to analyze")
    parser.add_argument('--gen', type=int, required=True,
                        help="Generation to load best agent from")
    parser.add_argument('--output', default='analysis',
                        help="Output directory (default: analysis)")
    parser.add_argument('--random-baseline', type=int, default=200,
                        help="Number of random agents for baseline (default: 200)")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Extract features from all runs
    rows = []
    for d in args.dirs:
        summary_path = os.path.join(d, f'gen_{args.gen}', 'summary.pkl')
        if not os.path.exists(summary_path):
            print(f"Skipping {d}: gen_{args.gen}/summary.pkl not found")
            continue

        agent, fitness = load_best_agent(d, args.gen)
        features = extract_features(agent)

        # Parse mouse ID from directory name (e.g. results_D3_r1 -> D3)
        parts = os.path.basename(d).split('_')
        if len(parts) >= 2:
            mouse_id = parts[1]
        else:
            mouse_id = 'aggregate'

        # Parse replicate from directory name
        rep = ''
        for p in parts:
            if p.startswith('r') and p[1:].isdigit():
                rep = p[1:]

        row = {'dir': d, 'mouse': mouse_id, 'rep': rep, 'fitness': fitness}
        row.update(features)
        rows.append(row)
        print(f"  {d}: {features['n_connections']} connections, density={features['density']:.3f}, "
              f"ei_ratio={features['ei_ratio']:.2f}, fitness={fitness:.4f}")

    if not rows:
        print("No valid runs found.")
        return

    # Save CSV
    csv_path = os.path.join(args.output, 'circuit_features.csv')
    fieldnames = ['dir', 'mouse', 'rep', 'fitness'] + FEATURE_NAMES
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {csv_path}")

    # Save pickle (includes numpy arrays for downstream analysis)
    pkl_path = os.path.join(args.output, 'circuit_features.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(rows, f)
    print(f"Saved pickle to {pkl_path}")

    # Random baseline
    if args.random_baseline > 0:
        print(f"\nComputing random baseline ({args.random_baseline} agents)...")
        from config import load_config
        config = load_config()
        mean_features, all_features = random_baseline(config, n=args.random_baseline)

        baseline_path = os.path.join(args.output, 'random_baseline.pkl')
        with open(baseline_path, 'wb') as f:
            pickle.dump({'mean': mean_features, 'all': all_features}, f)
        print(f"Saved random baseline to {baseline_path}")

        # Print comparison
        print("\nEvolved (mean) vs Random baseline:")
        print(f"  {'Feature':<18} {'Evolved':>10} {'Random':>10}")
        print(f"  {'-'*18} {'-'*10} {'-'*10}")
        for key in FEATURE_NAMES:
            evolved_mean = np.mean([r[key] for r in rows])
            print(f"  {key:<18} {evolved_mean:>10.3f} {mean_features[key]:>10.3f}")


if __name__ == '__main__':
    main()
