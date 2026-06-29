"""
resimulate_best_agents.py

Re-simulates best agents from all runs in a SINGLE GPU-BATCHED pass,
recording POSITION HISTORY so that speed/turn metrics can be computed
from displacements — the SAME way mouse metrics are computed.

Outputs  →  figures/emergent_data.pkl

Usage:
    python resimulate_best_agents.py
"""

import os
import sys

# Make project root (config, utils, core) and scripts/ (sibling modules like
# `run`) importable when invoked as `python scripts/X.py` from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d, distance_transform_edt

from config import load_config
from core.simulation import Simulation
from core.agent import Agent
from utils.backend import xp, to_cpu, HAS_GPU
from run import load_mouse_baselines

# ── Settings ─────────────────────────────────────────────────────────
RESULTS_PREFIX = "results_"
OUTPUT_PATH = "figures/emergent_data.pkl"

# N_BOUTS and MAX_FRAMES are loaded from config.yaml at runtime (see main());
# they are set here as module-level names for backward compatibility with
# any callers that reference them directly.
N_BOUTS = None
MAX_FRAMES = None

# ── Filtering thresholds (SAME as preprocess_mouse.compute_momentum_params) ──
SPEED_THRESHOLD = 0.1        # exclude near-stationary frames
MAX_TURN_MAGNITUDE = np.pi / 2  # exclude implausible sharp turns
SMOOTHING_SIGMA = 1.0        # Gaussian smoothing sigma for trajectories


def load_best_agents(run_dirs):
    """Load the best agent from the final generation of each run."""
    agents = []
    for run_dir in run_dirs:
        gen_dirs = [d for d in os.listdir(run_dir) if d.startswith("gen_")]
        gen = max(int(d.split("_")[1]) for d in gen_dirs)
        path = os.path.join(run_dir, f"gen_{gen}", "summary.pkl")
        with open(path, "rb") as f:
            pop = pickle.load(f)
        best_idx = min(range(len(pop)), key=lambda i: pop[i]["fitness"])
        best = pop[best_idx]
        print(f"  {run_dir}: agent #{best['id']} fitness={best['fitness']:.4f} (gen {gen})")
        agents.append(best["agent"])
    return agents


def build_wall_distance_field(simulation):
    """
    Build a distance-to-nearest-wall field from the occupancy grid.
    Returns (dist_field, grid_resolution, grid_origin) for position lookups.

    The occupancy grid is a boolean array where True = wall.
    We invert it and compute the Euclidean distance transform.
    """
    grid = to_cpu(simulation.grid_gpu).astype(bool)
    # distance_transform_edt gives distance from each False cell to nearest True cell
    # We want distance from free space to nearest wall
    # grid: True = wall, so ~grid = free space
    dist_field = distance_transform_edt(~grid)

    # The grid spans the maze coordinate space.
    # OccupancyGrid in simulation.py uses: resolution of grid cells per unit
    # We need to know the mapping from position to grid indices.
    # From simulation.py, the grid is created by OccupancyGrid which maps
    # the maze walls. Let's find the resolution from grid shape and maze bounds.
    occ = simulation.occ_grid
    return dist_field, occ


def position_to_wall_distance(positions, dist_field, occ_grid):
    """
    Look up wall distance for an array of positions using the distance field.

    Parameters
    ----------
    positions : ndarray (N, 2)
    dist_field : ndarray (grid_h, grid_w) — in grid-cell units
    occ_grid : OccupancyGrid object

    Returns
    -------
    distances : ndarray (N,) in maze coordinate units
    """
    # OccupancyGrid: resolution = cell_size (e.g. 0.05 coord units per cell)
    # grid index = (position - min) / resolution
    res = occ_grid.resolution
    gx = ((positions[:, 0] - occ_grid.min_x) / res).astype(int)
    gy = ((positions[:, 1] - occ_grid.min_y) / res).astype(int)

    # Clamp to grid bounds
    h, w = dist_field.shape
    gx = np.clip(gx, 0, w - 1)
    gy = np.clip(gy, 0, h - 1)

    # distance_transform_edt returns distances in grid-cell units;
    # multiply by resolution to convert to coordinate units
    return dist_field[gx, gy] * res


def compute_displacement_metrics(positions_per_bout, smooth=False):
    """
    Compute speed and turn distributions from position data using the
    EXACT same pipeline as preprocess_mouse.compute_momentum_params.

    Parameters
    ----------
    positions_per_bout : list of (T, 2) arrays
    smooth : bool — whether to apply Gaussian smoothing (True for mouse, False for agent)

    Returns
    -------
    all_speeds, all_turns : lists of filtered values
    """
    all_speeds = []
    all_turns = []

    for traj in positions_per_bout:
        if len(traj) < 20:
            continue

        # Check for NaNs (mouse data may have them; agent data won't)
        mask = np.isfinite(traj).all(axis=1)
        if not np.any(mask):
            continue

        for seg in np.split(traj, np.where(~mask)[0]):
            seg = seg[np.isfinite(seg).all(axis=1)]
            if len(seg) < 10:
                continue

            # Smooth trajectory (same sigma as preprocess_mouse.py)
            if smooth:
                seg = gaussian_filter1d(seg, sigma=SMOOTHING_SIGMA, axis=0)

            # Velocity and Speed (displacement-based)
            vel = np.diff(seg, axis=0)
            speeds = np.linalg.norm(vel, axis=1)

            # Heading and Turns
            headings = np.arctan2(vel[:, 1], vel[:, 0])
            unwrapped = np.unwrap(headings)
            turns = np.diff(unwrapped)

            # Filter for stable movement (SAME threshold as preprocess_mouse)
            stable_mask = speeds > SPEED_THRESHOLD
            all_speeds.extend(speeds[stable_mask])

            robust_turn_mask = stable_mask[:-1] & stable_mask[1:]
            if np.any(robust_turn_mask):
                physical_turns = turns[robust_turn_mask]
                physical_turns = physical_turns[np.abs(physical_turns) < MAX_TURN_MAGNITUDE]
                all_turns.extend(physical_turns)

    return all_speeds, all_turns


def batched_simulate(agents, simulation, config, baselines):
    """
    Run all agents × all bouts in parallel on GPU.
    Records POSITION HISTORY for displacement-based metric computation.
    """
    n_agents = len(agents)
    total_batch = n_agents * N_BOUTS
    print(f"\n  Batching {n_agents} agents × {N_BOUTS} bouts = {total_batch} parallel sims")

    # ── Stack weights (same as fitness.py) ───────────────────────────
    n_total = agents[0].n_total
    weights_list = [to_cpu(a.weights) for a in agents]
    w_stack = xp.stack([xp.array(w, dtype=xp.float64) for w in weights_list])
    w_stack = xp.repeat(w_stack[:, None, :, :], N_BOUTS, axis=1)
    mega_weights = w_stack.reshape(total_batch, n_total, n_total)

    # ── Generate noise matrix ────────────────────────────────────────
    noise_matrix = xp.zeros((N_BOUTS, MAX_FRAMES), dtype=xp.float64)
    for i in range(N_BOUTS):
        noise_matrix[i] = xp.array(
            np.random.RandomState(i).uniform(-1, 1, MAX_FRAMES)
        )
    mega_noise = xp.tile(noise_matrix, (n_agents, 1))

    # ── Initialize vectorised agent ──────────────────────────────────
    vec_agent = Agent(config.network, batch_size=total_batch)
    vec_agent.weights = mega_weights
    vec_agent.reset()

    # ── State arrays ─────────────────────────────────────────────────
    agent_pos = xp.zeros((total_batch, 2), dtype=xp.float64)
    agent_pos[:, 0] = -0.5
    agent_pos[:, 1] = 7.0

    heading = xp.zeros(total_batch, dtype=xp.float64)
    prev_speed = xp.zeros(total_batch, dtype=xp.float64)
    prev_turn = xp.zeros(total_batch, dtype=xp.float64)
    actual_speed = xp.zeros(total_batch, dtype=xp.float64)
    actual_turn = xp.zeros(total_batch, dtype=xp.float64)

    active_mask = xp.ones(total_batch, dtype=bool)
    exit_frames = xp.full(total_batch, MAX_FRAMES, dtype=int)

    # ── Position history (CPU, for displacement-based metrics) ───────
    pos_history = np.zeros((MAX_FRAMES, total_batch, 2), dtype=np.float64)

    # ── Physics ──────────────────────────────────────────────────────
    physics = config.physics
    m_phys = baselines.get("physics", {})
    p_max_speed = m_phys.get("max_speed", physics.max_speed)
    p_max_turn = m_phys.get("max_turn_rate", physics.max_turn_rate)
    p_alpha_speed = m_phys.get("alpha_speed", physics.alpha_speed)
    p_alpha_turn = m_phys.get("alpha_turn", physics.alpha_turn)

    # ── Simulation loop (mirrors fitness.py) ─────────────────────────
    for t in range(MAX_FRAMES):
        exited = (agent_pos[:, 0] < -1.0) & active_mask
        if xp.any(exited):
            exit_frames[exited] = t
            active_mask[exited] = False

        if t % 100 == 0:
            n_active = int(xp.sum(active_mask))
            print(f"    Frame {t}/{MAX_FRAMES}  ({n_active}/{total_batch} active)", end="\r", flush=True)
            if n_active == 0:
                break

        # Record positions
        pos_history[t] = to_cpu(agent_pos)

        # Raycast
        d_f = simulation.raycast(agent_pos, heading)
        d_l = simulation.raycast(agent_pos, heading + np.pi / 2)
        d_r = simulation.raycast(agent_pos, heading - np.pi / 2)

        # Build inputs
        inputs = xp.column_stack([
            d_f / 10, d_l / 10, d_r / 10,
            prev_speed, prev_turn,
            mega_noise[:, t],
        ])

        # Forward pass
        out = vec_agent.forward(inputs)
        raw_speed = out[:, 0]
        raw_turn = out[:, 1]

        # Physics (split momentum)
        target_turn = raw_turn * p_max_turn
        actual_turn = (1 - p_alpha_turn) * actual_turn + p_alpha_turn * target_turn
        heading += actual_turn
        heading = (heading + np.pi) % (2 * np.pi) - np.pi

        target_speed = raw_speed * p_max_speed
        actual_speed = (1 - p_alpha_speed) * actual_speed + p_alpha_speed * target_speed

        # Move
        step = xp.column_stack([
            actual_speed * xp.cos(heading),
            actual_speed * xp.sin(heading),
        ])
        new_pos, _ = simulation.step(agent_pos, step)
        agent_pos[active_mask] = new_pos[active_mask]

        # Proprioception feedback
        prev_speed = actual_speed / p_max_speed
        prev_turn = actual_turn / p_max_turn

    print()
    exit_frames_cpu = to_cpu(exit_frames).astype(int)

    # ── Build wall distance field for thigmotaxis ────────────────────
    print("  Computing wall distance field...")
    dist_field, occ_grid = build_wall_distance_field(simulation)

    # ── Post-process per agent ───────────────────────────────────────
    results = []
    for a_idx in range(n_agents):
        bout_start = a_idx * N_BOUTS
        bout_end = bout_start + N_BOUTS

        # Extract per-bout trajectories
        bout_trajectories = []
        all_positions = []
        for b in range(bout_start, bout_end):
            ef = exit_frames_cpu[b]
            traj = pos_history[:ef, b, :]  # (T, 2)
            bout_trajectories.append(traj)
            all_positions.append(traj)

        # Displacement-based speed & turn (SAME pipeline as mouse)
        # Agent positions are smooth from physics — no Gaussian smoothing needed
        all_speeds, all_turns = compute_displacement_metrics(bout_trajectories, smooth=False)

        # Thigmotaxis from position-based wall distance
        all_pos = np.concatenate(all_positions) if all_positions else np.empty((0, 2))
        if len(all_pos) > 0:
            wall_dists = position_to_wall_distance(all_pos, dist_field, occ_grid)
            thigmotaxis = float((wall_dists < 0.15).mean())
        else:
            wall_dists = np.array([])
            thigmotaxis = 0.0

        results.append({
            "run_id": a_idx + 1,
            "speeds": np.array(all_speeds),
            "turns": np.array(all_turns),
            "wall_dists": wall_dists,
            "thigmotaxis": thigmotaxis,
            "median_speed": float(np.median(all_speeds)) if all_speeds else 0.0,
            "mean_abs_turn": float(np.mean(np.abs(all_turns))) if all_turns else 0.0,
            "n_total_frames": sum(len(t) for t in bout_trajectories),
            "n_speed_samples": len(all_speeds),
            "n_turn_samples": len(all_turns),
        })

        print(f"  Run {a_idx+1}: {results[-1]['n_total_frames']} frames → "
              f"{results[-1]['n_speed_samples']} speed / {results[-1]['n_turn_samples']} turn samples, "
              f"thigmo={thigmotaxis:.3f}, med_speed={results[-1]['median_speed']:.4f}")

    return results


def compute_mouse_distributions(simulation):
    """
    Extract speed, turn, and thigmotaxis distributions from raw mouse data.
    Uses SAME displacement pipeline + filtering as agents.
    """
    from preprocess_mouse import load_tf_file, extract_trajectories
    from utils.maze import create_maze

    mouse_dir = "data/raw"
    if not os.path.isdir(mouse_dir):
        print(f"  Warning: {mouse_dir} not found, skipping mouse distributions")
        return None

    maze = create_maze(6)
    files = [f for f in os.listdir(mouse_dir) if os.path.isfile(os.path.join(mouse_dir, f))]

    all_trajectories = []
    for fname in files:
        fpath = os.path.join(mouse_dir, fname)
        try:
            tr = load_tf_file(fpath)
            raw_trajs, _ = extract_trajectories(tr, maze=maze)
            all_trajectories.extend(raw_trajs)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
            continue

    # Displacement-based speed/turn with smoothing (mouse has tracking jitter)
    all_speeds, all_turns = compute_displacement_metrics(all_trajectories, smooth=True)

    print(f"  Mouse: {len(all_speeds)} speed samples, {len(all_turns)} turn samples")

    # Thigmotaxis from positions
    dist_field, occ_grid = build_wall_distance_field(simulation)
    all_pos = []
    for traj in all_trajectories:
        mask = np.isfinite(traj).all(axis=1)
        all_pos.append(traj[mask])
    all_pos = np.concatenate(all_pos) if all_pos else np.empty((0, 2))

    if len(all_pos) > 0:
        wall_dists = position_to_wall_distance(all_pos, dist_field, occ_grid)
        thigmotaxis = float((wall_dists < 0.15).mean())
    else:
        thigmotaxis = None

    print(f"  Mouse thigmotaxis: {thigmotaxis:.3f}" if thigmotaxis else "  Mouse thigmotaxis: N/A")

    return {
        "speeds": np.array(all_speeds),
        "turns": np.array(all_turns),
        "thigmotaxis": thigmotaxis,
    }


def main():
    global N_BOUTS, MAX_FRAMES  # populate module-level names from config
    config = load_config("config.yaml")
    N_BOUTS = config.simulation.n_bouts
    MAX_FRAMES = config.simulation.max_frames
    sim = Simulation(config.physics)
    baselines = load_mouse_baselines()

    run_dirs = sorted(
        [d for d in os.listdir(".") if d.startswith(RESULTS_PREFIX)],
        key=lambda x: int(x.split("_")[1]),
    )
    print(f"Found {len(run_dirs)} runs")

    print("\nLoading best agents...")
    agents = load_best_agents(run_dirs)

    print("\nRunning batched simulation...")
    run_results = batched_simulate(agents, sim, config, baselines)

    print("\nComputing mouse distributions (same pipeline)...")
    mouse_dist = compute_mouse_distributions(sim)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output = {
        "runs": run_results,
        "mouse": mouse_dist,
        "settings": {
            "n_bouts": N_BOUTS,
            "max_frames": MAX_FRAMES,
            "thigmotaxis_threshold": 0.15,
            "speed_threshold": SPEED_THRESHOLD,
            "max_turn_magnitude": MAX_TURN_MAGNITUDE,
            "agent_smoothing": False,
            "mouse_smoothing_sigma": SMOOTHING_SIGMA,
        },
    }
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(output, f)
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
