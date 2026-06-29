#!/usr/bin/env python
"""
Generate representative trajectories for 9 best agents (one per mouse).

Lightweight CPU script — no GPU needed. Runs a single bout per agent
and saves (x,y) position histories + maze wall coordinates for Figure 3C.

Usage:
    python generate_trajectories.py [--base-dir agents] [--out figures/trajectories.pkl]
"""
import os
import sys

# Make project root (config, utils, core) and scripts/ (ei_analysis) importable
# regardless of how this file is invoked (e.g. python scripts/X.py from repo root).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import pickle
import numpy as np

from config import load_config
from core.simulation import Simulation
from ei_analysis import discover_runs, _load_pickle_cpu
from utils.backend import xp, to_cpu


def load_best_agent_per_mouse(base_dir, gen=150):
    """Load the best agent per mouse across replicates."""
    runs = discover_runs(base_dir)
    best = {}
    for mouse_id in sorted(runs):
        best_agent = None
        best_fitness = float('inf')
        best_rep = None
        for rep in sorted(runs[mouse_id]):
            run_dir = runs[mouse_id][rep]
            summary_path = os.path.join(run_dir, f'gen_{gen}', 'summary.pkl')
            if not os.path.exists(summary_path):
                continue
            pop = _load_pickle_cpu(summary_path)
            top = min(pop, key=lambda x: x['fitness'])
            if top['fitness'] < best_fitness:
                best_agent = top['agent']
                best_fitness = top['fitness']
                best_rep = rep
        if best_agent is not None:
            best[mouse_id] = {
                'agent': best_agent,
                'fitness': best_fitness,
                'rep': best_rep,
            }
            print(f"  {mouse_id}: r{best_rep} fitness={best_fitness:.4f}")
    return best


def simulate_single_bout(agent, sim, config, seed=0, max_frames=2000):
    """Run one bout and return (T, 2) position array."""
    agent.weights = xp.array(agent.weights)
    agent.batch_size = 1
    agent.reset()

    physics = config.physics
    p_max_speed = physics.max_speed
    p_max_turn = physics.max_turn_rate
    p_alpha_speed = physics.alpha_speed
    p_alpha_turn = physics.alpha_turn

    pos = xp.array([[-0.5, 7.0]], dtype=xp.float64)
    heading = xp.array([0.0], dtype=xp.float64)
    actual_speed = xp.zeros(1, dtype=xp.float64)
    actual_turn = xp.zeros(1, dtype=xp.float64)
    prev_speed = xp.zeros(1, dtype=xp.float64)
    prev_turn = xp.zeros(1, dtype=xp.float64)

    noise = xp.array(np.random.RandomState(seed).uniform(-1, 1, max_frames))

    trajectory = []
    for t in range(max_frames):
        if float(to_cpu(pos[0, 0])) < -1.0:
            break

        trajectory.append(to_cpu(pos[0]).copy())

        # Raycast
        d_f = sim.raycast(pos, heading)
        d_l = sim.raycast(pos, heading + np.pi / 2)
        d_r = sim.raycast(pos, heading - np.pi / 2)

        # Sensory input
        inputs = xp.zeros((1, 6), dtype=xp.float64)
        inputs[0, 0] = d_f[0] / 10.0
        inputs[0, 1] = d_l[0] / 10.0
        inputs[0, 2] = d_r[0] / 10.0
        inputs[0, 3] = prev_speed[0]
        inputs[0, 4] = prev_turn[0]
        inputs[0, 5] = noise[t]

        out = agent.forward(inputs)
        raw_speed = out[0, 0]
        raw_turn = out[0, 1]

        # Physics
        target_turn = raw_turn * p_max_turn
        actual_turn = (1 - p_alpha_turn) * actual_turn + p_alpha_turn * target_turn
        heading = heading + actual_turn
        heading = (heading + xp.pi) % (2 * xp.pi) - xp.pi

        target_speed = raw_speed * p_max_speed
        actual_speed = (1 - p_alpha_speed) * actual_speed + p_alpha_speed * target_speed

        step = xp.zeros((1, 2), dtype=xp.float64)
        step[0, 0] = actual_speed[0] * xp.cos(heading[0])
        step[0, 1] = actual_speed[0] * xp.sin(heading[0])

        pos, _ = sim.step(pos, step)
        prev_speed = actual_speed / p_max_speed
        prev_turn = actual_turn / p_max_turn

    return np.array(trajectory)


def main():
    parser = argparse.ArgumentParser(description="Generate trajectories for Fig 3C")
    parser.add_argument('--base-dir', default='data/agents')
    parser.add_argument('--out', default='figures/trajectories.pkl')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    config = load_config()
    sim = Simulation(config.physics, maze_levels=6)

    print("Loading best agent per mouse...")
    best = load_best_agent_per_mouse(args.base_dir)

    print("\nSimulating trajectories...")
    trajectories = {}
    for mouse_id, info in sorted(best.items()):
        traj = simulate_single_bout(info['agent'], sim, config, seed=args.seed)
        trajectories[mouse_id] = traj
        print(f"  {mouse_id}: {len(traj)} frames")

    # Extract maze walls for background
    maze_walls = sim.maze.walls

    output = {
        'trajectories': trajectories,
        'maze_walls': maze_walls,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
