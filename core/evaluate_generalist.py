"""
Fast generalist evaluation: all mice simultaneously in ONE simulation pass.

Key insight: the Python frame loop (2000 iters) runs 9× for 9 mice sequentially.
Tiling agents 9× with per-agent physics collapses 9 × 2000 = 18,000 iterations
into 2,000, giving ~9× speedup on Python/GPU-launch overhead.

Additional optimisations vs evaluate_batch:
  - FIXED_SUB_STEPS reduced to 5 (was 30, still correct for maze scale)
  - No xp.any() GPU sync per frame or per sub-step
  - No boolean fancy indexing in step (uses clip + bulk gather + xp.where)
  - Downsampled pos stored from the start — no redundant memory
"""
import numpy as np
from utils.backend import xp, to_cpu, HAS_GPU
from core.agent import Agent
from core.fitness import (FitnessResult, _build_run_grid, _pos_to_run_ids,
                           _compute_node_metrics_fast, _compute_tortuosity)
from core.constants import DOWNSAMPLE_FACTOR

# Sub-steps for collision detection.  5 is sufficient for maze navigation
# (max agent displacement per frame ≈ 2 units; grid cell ≈ 0.5 units → 5×0.4 = safe).
GENERALIST_SUB_STEPS = 5


# ---------------------------------------------------------------------------
# Fast step (no GPU syncs, no boolean fancy indexing)
# ---------------------------------------------------------------------------

def _step_fast(positions, velocities, grid_gpu, occ, n_steps):
    """Vectorised sub-stepping with zero GPU→CPU synchronisation.

    Replaces simulation.step() for the generalist training loop.
    Uses clip+bulk-gather+xp.where instead of xp.any()+boolean fancy indexing.
    """
    step_delta = velocities / n_steps
    pos = positions.copy()
    min_x, min_y = occ.min_x, occ.min_y
    res    = occ.resolution
    width  = occ.width
    height = occ.height

    for _ in range(n_steps):
        proposed = pos + step_delta

        gx = ((proposed[:, 0] - min_x) / res).astype(xp.int32)
        gy = ((proposed[:, 1] - min_y) / res).astype(xp.int32)

        # Clip to grid bounds — safe gather without boolean mask or xp.any()
        xi = xp.clip(gx, 0, width  - 1)
        yi = xp.clip(gy, 0, height - 1)
        in_bounds  = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
        is_wall    = grid_gpu[xi, yi].astype(bool)
        collisions = ~in_bounds | is_wall           # (total_batch,) bool

        # xp.where: fully vectorised, no scatter/gather, no sync
        pos        = xp.where(collisions[:, None], pos,       proposed)
        step_delta = xp.where(collisions[:, None], xp.float64(0.0), step_delta)

    return pos


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_all_mice(agents, simulation, config, all_baselines, noise_matrix,
                      sub_steps=GENERALIST_SUB_STEPS):
    """Evaluate agents against ALL mice in a SINGLE simulation pass.

    Layout: (agent × mouse × bout) flattened.
      index(i, m, b) = i * n_mice * n_bouts + m * n_bouts + b

    Returns
    -------
    per_mouse_results : Dict[mouse_id → List[FitnessResult]]
        per_mouse_results[mouse_id][i] is agent i's FitnessResult vs that mouse.
    """
    mice     = list(all_baselines.keys())
    n_agents = len(agents)
    n_mice   = len(mice)
    n_bouts  = config.simulation.n_bouts
    max_frames = config.simulation.max_frames
    total_batch = n_agents * n_mice * n_bouts

    print(f"  Combined batch: {n_agents} agents × {n_mice} mice × {n_bouts} bouts "
          f"= {total_batch:,} simultaneous trajectories")

    # ── mega_weights ────────────────────────────────────────────────────────
    # Each agent's (14,14) weight matrix is tiled n_mice × n_bouts times.
    n_total = agents[0].n_total
    w_np = np.stack([to_cpu(a.weights) for a in agents])          # (A, n_total, n_total)
    w_tiled = np.tile(w_np[:, None, None, :, :],
                      (1, n_mice, n_bouts, 1, 1))                  # (A,M,B,n_total,n_total)
    mega_weights = xp.array(w_tiled.reshape(total_batch, n_total, n_total))  # (total_batch,n_total,n_total)

    # ── per-agent physics arrays ─────────────────────────────────────────
    phy = config.physics
    ms_arr = np.empty(total_batch, np.float64)
    mt_arr = np.empty(total_batch, np.float64)
    as_arr = np.empty(total_batch, np.float64)
    at_arr = np.empty(total_batch, np.float64)

    for m, mid in enumerate(mice):
        ph = all_baselines[mid].get('physics', {})
        ms = ph.get('max_speed',     phy.max_speed)
        mt = ph.get('max_turn_rate', phy.max_turn_rate)
        als= ph.get('alpha_speed',   phy.momentum_alpha)
        alt= ph.get('alpha_turn',    phy.momentum_alpha)
        for i in range(n_agents):
            s = i * n_mice * n_bouts + m * n_bouts
            e = s + n_bouts
            ms_arr[s:e] = ms;  mt_arr[s:e] = mt
            as_arr[s:e] = als; at_arr[s:e] = alt

    p_max_speed = xp.array(ms_arr)
    p_max_turn  = xp.array(mt_arr)
    p_alpha_s   = xp.array(as_arr)
    p_alpha_t   = xp.array(at_arr)

    # ── noise ───────────────────────────────────────────────────────────
    # Tile the (n_bouts, max_frames) noise matrix for n_agents × n_mice copies.
    mega_noise = xp.array(np.tile(noise_matrix, (n_agents * n_mice, 1)))

    # ── vectorised agent ────────────────────────────────────────────────
    vec_agent = Agent(config.network, batch_size=total_batch)
    vec_agent.weights = mega_weights
    vec_agent.reset()

    # ── state arrays ────────────────────────────────────────────────────
    agent_pos    = xp.zeros((total_batch, 2), dtype=xp.float64)
    agent_pos[:, 0] = -0.5
    agent_pos[:, 1] =  7.0
    heading      = xp.zeros(total_batch, dtype=xp.float64)
    prev_speed   = xp.zeros(total_batch, dtype=xp.float64)
    prev_turn    = xp.zeros(total_batch, dtype=xp.float64)
    actual_speed = xp.zeros(total_batch, dtype=xp.float64)
    actual_turn  = xp.zeros(total_batch, dtype=xp.float64)

    # Store only downsampled positions (saves GPU memory)
    n_ds = max_frames // DOWNSAMPLE_FACTOR + 1
    pos_history  = xp.zeros((n_ds, total_batch, 2), dtype=xp.float64)
    exit_frames  = xp.full(total_batch, max_frames, dtype=xp.int32)
    active_mask  = xp.ones(total_batch, dtype=bool)

    occ      = simulation.occ_grid
    grid_gpu = simulation.grid_gpu
    frame_ds = 0

    # ── simulation loop (ONE pass for all agents × all mice) ─────────────
    print(f"  Simulating {max_frames} frames (sub_steps={sub_steps}) …", flush=True)
    for t in range(max_frames):
        # Exit detection — no xp.any() sync (async mask update)
        new_exits   = (agent_pos[:, 0] < -1.0) & active_mask
        exit_frames = xp.where(new_exits, xp.int32(t), exit_frames)
        active_mask = active_mask & ~new_exits

        # Store downsampled positions
        if t % DOWNSAMPLE_FACTOR == 0:
            pos_history[frame_ds] = agent_pos
            frame_ds += 1

        d_f = simulation.raycast(agent_pos, heading)
        d_l = simulation.raycast(agent_pos, heading + np.pi / 2)
        d_r = simulation.raycast(agent_pos, heading - np.pi / 2)

        inputs = xp.column_stack([
            d_f / 10, d_l / 10, d_r / 10,
            prev_speed, prev_turn,
            mega_noise[:, t]
        ])

        out       = vec_agent.forward(inputs)
        raw_speed = out[:, 0]
        raw_turn  = out[:, 1]

        # Per-agent physics (element-wise arrays, not scalars)
        target_turn  = raw_turn  * p_max_turn
        actual_turn  = (1.0 - p_alpha_t) * actual_turn  + p_alpha_t * target_turn
        heading     += actual_turn
        heading      = (heading + np.pi) % (2 * np.pi) - np.pi

        target_speed = raw_speed * p_max_speed
        actual_speed = (1.0 - p_alpha_s) * actual_speed + p_alpha_s * target_speed

        step_vec = xp.column_stack([
            actual_speed * xp.cos(heading),
            actual_speed * xp.sin(heading),
        ])

        new_pos      = _step_fast(agent_pos, step_vec, grid_gpu, occ, sub_steps)
        agent_pos    = xp.where(active_mask[:, None], new_pos, agent_pos)
        prev_speed   = actual_speed / xp.maximum(p_max_speed,  1e-9)
        prev_turn    = actual_turn  / xp.maximum(p_max_turn,   1e-9)

        # Termination check every 500 frames (minimise syncs)
        if t % 500 == 499 and not bool(xp.any(active_mask)):
            break

    print(f"  Simulation done ({frame_ds} downsampled frames stored).", flush=True)

    # ── metrics ──────────────────────────────────────────────────────────
    pos_ds   = pos_history[:frame_ds]                      # still on GPU if HAS_GPU
    exit_cpu = to_cpu(exit_frames)
    run_grid, rg_x, rg_y = _build_run_grid(simulation.maze)
    run_ids_all = _pos_to_run_ids(pos_ds, run_grid, rg_x, rg_y)  # CPU int32
    pos_cpu  = to_cpu(pos_ds)
    exit_ds  = (exit_cpu // DOWNSAMPLE_FACTOR).clip(0, frame_ds).astype(np.int32)

    fw = config.fitness
    per_mouse = {mid: [] for mid in mice}

    for i in range(n_agents):
        for mi, mid in enumerate(mice):
            start = i * n_mice * n_bouts + mi * n_bouts
            end   = start + n_bouts
            bl    = all_baselines[mid]

            tort = _compute_tortuosity(pos_cpu, start, end, exit_cpu, bl['straightness'])
            node = _compute_node_metrics_fast(run_ids_all, start, end, exit_ds,
                                              simulation.maze, bl)

            nm = min(1.0, node['markov']    / 6.0)
            no = min(1.0, node['occupancy'])
            nt = min(1.0, tort)
            nb = min(1.0, node['turn_bias'])

            m_c = nm * fw.markov.weight    if fw.markov.enabled    else 0.0
            o_c = no * fw.occupancy.weight if fw.occupancy.enabled else 0.0
            t_c = nt * fw.tortuosity.weight if fw.tortuosity.enabled else 0.0
            b_c = nb * fw.turn_bias.weight  if fw.turn_bias.enabled  else 0.0

            per_mouse[mid].append(FitnessResult(
                total=m_c + o_c + t_c + b_c,
                markov=m_c, occupancy=o_c, tortuosity=t_c, turn_bias=b_c,
            ))

    return per_mouse
