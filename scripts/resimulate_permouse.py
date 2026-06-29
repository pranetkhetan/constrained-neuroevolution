"""
resimulate_permouse.py

Re-simulates the BEST agent per mouse (across all replicates) in a
single GPU-batched pass. Records position history for displacement-based
speed/turn/thigmotaxis metrics — SAME pipeline as mouse preprocessing.

Also computes per-mouse behavioral distributions from raw tracking data.

Outputs  →  figures/emergent_data_permouse.pkl

Usage:
    python resimulate_permouse.py
    python resimulate_permouse.py --base-dir results --mice B5 B6 B7
"""
import os
import sys

# Make project root (config, utils, core) and scripts/ (sibling modules like
# `run`, `ei_analysis`, `resimulate_best_agents`) importable when invoked as
# `python scripts/X.py` from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pickle
import argparse
import numpy as np

from config import load_config
from core.simulation import Simulation
from utils.backend import xp, to_cpu, HAS_GPU
from run import load_mouse_baselines
from ei_analysis import discover_runs, _load_pickle_cpu
from resimulate_best_agents import (
    batched_simulate,
    compute_displacement_metrics,
    build_wall_distance_field,
    position_to_wall_distance,
    SPEED_THRESHOLD,
    MAX_TURN_MAGNITUDE,
    SMOOTHING_SIGMA,
)

OUTPUT_PATH = "figures/emergent_data_permouse.pkl"


def load_best_agent_per_mouse(base_dir, gen=None):
    """
    For each mouse, find the best agent across all replicates.

    Returns:
        dict: {mouse_id: {'agent': Agent, 'fitness': float, 'rep': int, 'gen': int}}
    """
    runs = discover_runs(base_dir)
    result = {}

    for mouse_id in sorted(runs):
        best_agent = None
        best_fitness = float('inf')
        best_rep = None
        best_gen = None

        for rep in sorted(runs[mouse_id]):
            run_dir = runs[mouse_id][rep]
            if gen is not None:
                target_gen = gen
            else:
                gen_dirs = [d for d in os.listdir(run_dir) if d.startswith('gen_')]
                if not gen_dirs:
                    continue
                target_gen = max(int(d.split('_')[1]) for d in gen_dirs)

            summary_path = os.path.join(run_dir, f'gen_{target_gen}', 'summary.pkl')
            if not os.path.exists(summary_path):
                continue

            pop = _load_pickle_cpu(summary_path)
            best = min(pop, key=lambda x: x['fitness'])
            if best['fitness'] < best_fitness:
                best_agent = best['agent']
                best_fitness = best['fitness']
                best_rep = rep
                best_gen = target_gen

        if best_agent is not None:
            result[mouse_id] = {
                'agent': best_agent,
                'fitness': best_fitness,
                'rep': best_rep,
                'gen': best_gen,
            }
            print(f"  {mouse_id}: best = r{best_rep} gen={best_gen} "
                  f"fitness={best_fitness:.4f}")

    return result


def compute_mouse_distributions_single(simulation, mouse_id):
    """
    Extract speed/turn/thigmotaxis from raw tracking data for one mouse.

    Falls back to pre-computed metrics if raw data unavailable.
    """
    from scipy.ndimage import gaussian_filter1d

    mouse_dir = "data/raw"

    # Try raw data first
    if os.path.isdir(mouse_dir):
        try:
            from preprocess_mouse import load_tf_file, extract_trajectories
            from utils.maze import create_maze

            maze = create_maze(6)
            files = [f for f in os.listdir(mouse_dir)
                     if f.startswith(mouse_id) and os.path.isfile(
                         os.path.join(mouse_dir, f))]

            if files:
                all_trajectories = []
                for fname in files:
                    fpath = os.path.join(mouse_dir, fname)
                    try:
                        tr = load_tf_file(fpath)
                        raw_trajs, _ = extract_trajectories(tr, maze=maze)
                        all_trajectories.extend(raw_trajs)
                    except Exception as e:
                        print(f"    Skipping {fname}: {e}")
                        continue

                if all_trajectories:
                    all_speeds, all_turns = compute_displacement_metrics(
                        all_trajectories, smooth=True)

                    dist_field, occ_grid = build_wall_distance_field(simulation)
                    all_pos = []
                    for traj in all_trajectories:
                        mask = np.isfinite(traj).all(axis=1)
                        all_pos.append(traj[mask])
                    all_pos = np.concatenate(all_pos) if all_pos else np.empty((0, 2))
                    thigmotaxis = float((position_to_wall_distance(
                        all_pos, dist_field, occ_grid) < 0.15).mean()) if len(all_pos) > 0 else None

                    print(f"    {mouse_id}: {len(all_speeds)} speed, "
                          f"{len(all_turns)} turn samples from raw data")
                    speeds_arr = np.array(all_speeds)
                    turns_arr = np.array(all_turns)
                    return {
                        'speeds': speeds_arr,
                        'turns': turns_arr,
                        'thigmotaxis': thigmotaxis,
                        # Scalar summaries — same definition as the agent side
                        # (see compute_displacement_metrics) so per-mouse ticks
                        # in the emergent figure compare apples-to-apples.
                        'median_speed': (float(np.median(np.abs(speeds_arr)))
                                         if speeds_arr.size else None),
                        'mean_abs_turn': (float(np.mean(np.abs(turns_arr)))
                                          if turns_arr.size else None),
                    }
        except ImportError:
            pass

    # Fallback: pre-computed metrics (no distributions, but has summary stats)
    metrics_path = os.path.join('data', f'mouse_{mouse_id}_metrics.pkl')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            data = pickle.load(f)
        print(f"    {mouse_id}: using pre-computed metrics (no raw distributions)")
        physics = data.get('physics', {})
        return {
            'speeds': np.array([]),
            'turns': np.array([]),
            'thigmotaxis': data.get('thigmotaxis'),
            'median_speed': physics.get('median_speed'),
            # mean_abs_turn isn't in the preprocess physics dict; only present
            # when the raw branch above runs.
            'mean_abs_turn': None,
        }

    print(f"    {mouse_id}: no raw data or metrics found")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Re-simulate best agents per mouse")
    parser.add_argument('--base-dir', default='results')
    parser.add_argument('--mice', nargs='*', default=None,
                        help="Specific mice to process (default: all)")
    parser.add_argument('--gen', type=int, default=None,
                        help="Generation to load (default: final)")
    parser.add_argument('--output', default=OUTPUT_PATH)
    args = parser.parse_args()

    config = load_config("config.yaml")
    sim = Simulation(config.physics)

    print("Loading best agents per mouse...")
    best_per_mouse = load_best_agent_per_mouse(args.base_dir, gen=args.gen)

    if args.mice:
        best_per_mouse = {m: v for m, v in best_per_mouse.items()
                          if m in args.mice}

    mice_ids = sorted(best_per_mouse.keys())
    agents = [best_per_mouse[m]['agent'] for m in mice_ids]

    print(f"\nRunning batched simulation ({len(agents)} agents)...")
    # Use the first mouse's baselines for physics (they share the same maze)
    baselines = load_mouse_baselines(mice_ids[0])
    run_results = batched_simulate(agents, sim, config, baselines)

    print("\nComputing per-mouse distributions...")
    output_mice = {}
    for i, mouse_id in enumerate(mice_ids):
        mouse_dist = compute_mouse_distributions_single(sim, mouse_id)
        output_mice[mouse_id] = {
            'agent': run_results[i],
            'mouse': mouse_dist,
            'best_rep': best_per_mouse[mouse_id]['rep'],
            'best_fitness': best_per_mouse[mouse_id]['fitness'],
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        'mice': output_mice,
        'settings': {
            'n_bouts': 20,
            'max_frames': 2000,
            'thigmotaxis_threshold': 0.15,
            'speed_threshold': SPEED_THRESHOLD,
            'max_turn_magnitude': MAX_TURN_MAGNITUDE,
            'mouse_smoothing_sigma': SMOOTHING_SIGMA,
        },
    }
    with open(args.output, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
