"""
Fitness evaluation with mouse-likeness metrics.

Metrics:
- Markov Score: 2nd-order transition probability similarity
- Occupancy: Node visit distribution similarity
- Tortuosity: Path straightness deviation
- Turn Bias: Left/right preference matching
"""
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from core.agent import Agent
from core.constants import DOWNSAMPLE_FACTOR, TORTUOSITY_WINDOW
from utils.backend import to_cpu, xp, HAS_GPU
from utils.markov import calculate_markov_distance
from utils.metrics import compute_node_pdf, compute_straightness, compute_turn_bias


@dataclass
class FitnessResult:
    """Container for all fitness components."""
    total: float
    markov: float
    occupancy: float
    tortuosity: float
    turn_bias: float


def evaluate_batch(agents, simulation, config, mouse_baselines, noise_matrix):
    """
    Evaluate a batch of agents in parallel simulation.
    
    Args:
        agents: List of Agent objects
        simulation: Simulation environment
        config: Full Config object
        mouse_baselines: Dict with 'node_pdf', 'straightness', 'turn_bias', 'markov_profile'
        noise_matrix: (n_bouts, max_frames) pre-generated noise
        
    Returns:
        List of FitnessResult for each agent
    """
    n_agents = len(agents)
    n_bouts = config.simulation.n_bouts
    max_frames = config.simulation.max_frames
    total_batch = n_agents * n_bouts
    
    # Stack weights for parallel processing
    weights_list = [to_cpu(a.weights) for a in agents]
    w_stack = xp.stack([xp.array(w) for w in weights_list])
    w_stack = xp.repeat(w_stack[:, None, :, :], n_bouts, axis=1)
    n_total = agents[0].n_total
    mega_weights = w_stack.reshape(total_batch, n_total, n_total)
    
    # Expand noise for all agents
    mega_noise = xp.tile(noise_matrix, (n_agents, 1))
    
    # Initialize batch agent
    from core.agent import Agent
    vec_agent = Agent(config.network, batch_size=total_batch)
    vec_agent.weights = mega_weights
    vec_agent.reset()
    
    # Initialize positions and state
    agent_pos = xp.zeros((total_batch, 2), dtype=xp.float64)
    agent_pos[:, 0] = -0.5
    agent_pos[:, 1] = 7.0
    
    heading = xp.zeros(total_batch, dtype=xp.float64)
    prev_speed = xp.zeros(total_batch, dtype=xp.float64)
    prev_turn = xp.zeros(total_batch, dtype=xp.float64)
    actual_speed = xp.zeros(total_batch, dtype=xp.float64)
    actual_turn = xp.zeros(total_batch, dtype=xp.float64)
    
    # History tracking (only position needed for metrics)
    pos_history = xp.zeros((max_frames, total_batch, 2), dtype=xp.float64)
    
    active_mask = xp.ones(total_batch, dtype=bool)
    exit_frames = xp.full(total_batch, max_frames, dtype=int)
    
    physics = config.physics
    # Dynamic physics override from mouse data
    m_phys = mouse_baselines.get('physics', {})
    p_max_speed = m_phys.get('max_speed', physics.max_speed)
    p_max_turn = m_phys.get('max_turn_rate', physics.max_turn_rate)
    p_alpha_speed = m_phys.get('alpha_speed', physics.momentum_alpha)
    p_alpha_turn = m_phys.get('alpha_turn', physics.momentum_alpha)
    
    # Simulation loop
    for t in range(max_frames):
        # Check exits
        exited = (agent_pos[:, 0] < -1.0) & active_mask
        if xp.any(exited):
            exit_frames[exited] = t
            active_mask[exited] = False
        
        if t % 100 == 0:
            print(f"    Simulating frame {t}/{max_frames}...", end='\r', flush=True)
            
            if not xp.any(active_mask):
                break
        
        pos_history[t] = agent_pos
        
        # Raycast for distances
        d_f = simulation.raycast(agent_pos, heading)
        d_l = simulation.raycast(agent_pos, heading + np.pi / 2)
        d_r = simulation.raycast(agent_pos, heading - np.pi / 2)
        
        # Build inputs
        inputs = xp.column_stack([
            d_f / 10, d_l / 10, d_r / 10,
            prev_speed, prev_turn,
            mega_noise[:, t]
        ])
        
        # Forward pass
        out = vec_agent.forward(inputs)
        raw_speed = out[:, 0]
        raw_turn = out[:, 1]
        
        # Apply physics with split momentum
        target_turn = raw_turn * p_max_turn
        actual_turn = (1 - p_alpha_turn) * actual_turn + p_alpha_turn * target_turn
        heading += actual_turn
        heading = (heading + np.pi) % (2 * np.pi) - np.pi
        
        target_speed = raw_speed * p_max_speed
        actual_speed = (1 - p_alpha_speed) * actual_speed + p_alpha_speed * target_speed
        
        step = xp.column_stack([
            actual_speed * xp.cos(heading),
            actual_speed * xp.sin(heading)
        ])
        
        # Move
        new_pos, _ = simulation.step(agent_pos, step)
        agent_pos[active_mask] = new_pos[active_mask]
        
        # Update proprioception
        prev_speed = actual_speed / p_max_speed
        prev_turn = actual_turn / p_max_turn
    
    # Compute metrics
    results = _compute_metrics(
        n_agents, n_bouts, max_frames,
        pos_history, exit_frames,
        simulation.maze, config, mouse_baselines
    )
    
    return results


def _build_run_grid(maze):
    """Build a 2D numpy int32 array for fast vectorized (x,y) -> run_id lookup.

    Returns:
        grid  : (H, W) int32 CPU array, -1 for positions not in the maze
        x_min : int, minimum x coordinate in cell_lookup
        y_min : int, minimum y coordinate in cell_lookup
    """
    xs = [k[0] for k in maze.cell_lookup]
    ys = [k[1] for k in maze.cell_lookup]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    grid = np.full((y_max - y_min + 1, x_max - x_min + 1), -1, dtype=np.int32)
    for (x, y), cell_id in maze.cell_lookup.items():
        grid[y - y_min, x - x_min] = maze.run_lookup[cell_id]
    return grid, x_min, y_min


def _pos_to_run_ids(pos, grid_cpu, x_min, y_min):
    """Vectorized (T, B, 2) positions -> (T, B) run_id int32 array.

    Accepts either a CPU numpy array or a CuPy GPU array.  When GPU is
    available the lookup is performed entirely on-device; only the compact
    int32 result (4× smaller than the float64 input) is transferred to CPU.

    Out-of-maze and non-finite positions map to -1.
    """
    if HAS_GPU:
        import cupy as cp
        # Transfer small grid to GPU once; keep pos on GPU
        grid_gpu = cp.asarray(grid_cpu)
        H, W = grid_cpu.shape

        p0 = cp.where(cp.isfinite(pos[:, :, 0]), pos[:, :, 0], cp.float64(0.0))
        p1 = cp.where(cp.isfinite(pos[:, :, 1]), pos[:, :, 1], cp.float64(0.0))
        xs = cp.rint(p0).astype(cp.int32) - np.int32(x_min)
        ys = cp.rint(p1).astype(cp.int32) - np.int32(y_min)

        # Clip to grid bounds to allow safe gather (out-of-bounds corrected below)
        xi = cp.clip(xs, 0, W - 1)
        yi = cp.clip(ys, 0, H - 1)

        # Single bulk gather — no boolean fancy-index overhead
        run_ids_gpu = grid_gpu[yi, xi].astype(cp.int32)

        # Mask truly out-of-bounds positions to -1
        invalid = (xs < 0) | (xs >= W) | (ys < 0) | (ys >= H) | \
                  ~cp.isfinite(pos[:, :, 0]) | ~cp.isfinite(pos[:, :, 1])
        run_ids_gpu[invalid] = cp.int32(-1)

        return cp.asnumpy(run_ids_gpu)   # (T, B) CPU int32 — 4× smaller than float64 pos
    else:
        H, W = grid_cpu.shape
        finite = np.isfinite(pos[:, :, 0]) & np.isfinite(pos[:, :, 1])
        p0 = np.where(finite, pos[:, :, 0], float(x_min))
        p1 = np.where(finite, pos[:, :, 1], float(y_min))
        xs = np.rint(p0).astype(np.int32) - x_min
        ys = np.rint(p1).astype(np.int32) - y_min
        xi = np.clip(xs, 0, W - 1)
        yi = np.clip(ys, 0, H - 1)
        run_ids = grid_cpu[yi, xi].copy()
        run_ids[(xs < 0) | (xs >= W) | (ys < 0) | (ys >= H) | ~finite] = -1
        return run_ids   # (T, B) CPU int32


def _compute_node_metrics_fast(run_ids_all, start_idx, end_idx, exit_ds,
                                maze, baselines):
    """Fully vectorized node-based metrics for one agent.

    Parameters
    ----------
    run_ids_all : (T_ds, total_batch) int32 CPU array
    start_idx, end_idx : column range for this agent's bouts
    exit_ds : (total_batch,) int array — exit frame already divided by DOWNSAMPLE_FACTOR
    maze, baselines : as usual
    """
    n_runs  = len(maze.runs)
    n_bouts = end_idx - start_idx
    T_ds    = run_ids_all.shape[0]

    # ----- slice this agent's columns once -----
    agent_slice = run_ids_all[:, start_idx:end_idx]          # (T_ds, n_bouts)
    agent_exit  = exit_ds[start_idx:end_idx].clip(0, T_ds)   # (n_bouts,)

    # Vectorized exit-frame mask: frame t is valid when t < exit_ds[b]
    frame_idx  = np.arange(T_ds, dtype=np.int32)[:, np.newaxis]  # (T_ds, 1)
    within_exit = frame_idx < agent_exit[np.newaxis, :]           # (T_ds, n_bouts)
    valid_mask  = (agent_slice >= 0) & within_exit                # (T_ds, n_bouts)

    total_valid = int(valid_mask.sum())
    if total_valid < 3:
        return {'markov': 6.0, 'occupancy': 2.0, 'turn_bias': 1.0}

    # === Occupancy: one bincount over all valid positions ===
    node_counts = np.bincount(agent_slice[valid_mask], minlength=n_runs)

    # === Turn bias: vectorized across all bouts simultaneously ===
    prev_r     = agent_slice[:-1]       # (T_ds-1, n_bouts)
    next_r     = agent_slice[1:]        # (T_ds-1, n_bouts)
    prev_valid = valid_mask[:-1]
    next_valid = valid_mask[1:]
    trans_mask = prev_valid & next_valid & (prev_r != next_r)
    p_nodes    = prev_r[trans_mask]
    n_nodes_   = next_r[trans_mask]
    n_left = n_right = 0
    if len(p_nodes):
        v = (p_nodes < maze.st.shape[0]) & (n_nodes_ < maze.st.shape[1])
        if v.any():
            stypes  = maze.st[p_nodes[v], n_nodes_[v]]
            n_left  = int(np.sum((stypes == 0) | (stypes == 2)))
            n_right = int(np.sum((stypes == 1) | (stypes == 3)))

    # === Markov: per-bout run-transition sequences ===
    bouts = []
    for b in range(n_bouts):
        T     = int(agent_exit[b])
        rids  = agent_slice[:T, b]
        vr    = rids[rids >= 0]
        if not len(vr):
            continue
        ch    = np.empty(len(vr), dtype=bool)
        ch[0] = True
        ch[1:] = vr[1:] != vr[:-1]
        bouts.append([n_runs] + vr[ch].tolist() + [n_runs])

    if not bouts:
        return {'markov': 6.0, 'occupancy': 2.0, 'turn_bias': 1.0}

    mouse_profile = baselines.get('markov_profile')
    if mouse_profile is None:
        raise ValueError(
            "Mouse Markov profile not found. Run 'python preprocess_mouse.py <tf_file>' first."
        )
    markov_score = calculate_markov_distance(bouts, mouse_profile, maze)

    # Occupancy JSD
    total_c  = node_counts.sum()
    apdf     = node_counts.astype(np.float64) / (total_c + 1e-9)
    mpdf     = baselines.get('node_pdf', np.zeros(n_runs))
    sz       = min(len(apdf), len(mpdf))
    p        = np.clip(apdf[:sz], 1e-10, None);  p /= p.sum()
    q        = np.clip(mpdf[:sz], 1e-10, None);  q /= q.sum()
    m_pdf    = 0.5 * (p + q)
    js_div   = 0.5 * (np.sum(p * np.log(p / m_pdf)) + np.sum(q * np.log(q / m_pdf)))
    occ_diff = float(js_div / np.log(2))

    # Turn bias
    tot_t      = n_left + n_right
    agent_bias = (n_right - n_left) / tot_t if tot_t else 0.0
    tb_diff    = abs(agent_bias - baselines.get('turn_bias', 0.0))

    return {'markov': markov_score, 'occupancy': occ_diff, 'turn_bias': tb_diff}


def _compute_metrics(n_agents, n_bouts, max_frames,
                     pos_history, exit_frames,
                     maze, config, baselines):
    """Compute all fitness metrics for each agent."""

    exit_cpu = to_cpu(exit_frames)

    # Build the small (H×W) run-id lookup grid on CPU
    run_grid, rg_x_min, rg_y_min = _build_run_grid(maze)

    # ── GPU path: compute run_ids on GPU from pos_history while it is still
    #   resident; transfers only the compact int32 result (4× smaller than
    #   the float64 position array) to CPU.
    # ── CPU path: standard numpy.
    # pos_cpu is still needed for tortuosity regardless.
    pos_ds  = pos_history[::DOWNSAMPLE_FACTOR]   # (T_ds, total_batch, 2) — GPU or CPU
    run_ids_all = _pos_to_run_ids(pos_ds, run_grid, rg_x_min, rg_y_min)  # (T_ds, total_batch) CPU
    pos_cpu = to_cpu(pos_ds)
    T_ds    = run_ids_all.shape[0]

    # Pre-divide exit frames to avoid repeated integer division inside the loop
    exit_ds = (exit_cpu // DOWNSAMPLE_FACTOR).clip(0, T_ds).astype(np.int32)

    results = []
    weights = config.fitness

    for i in range(n_agents):
        start_idx = i * n_bouts
        end_idx   = (i + 1) * n_bouts

        # 1. Tortuosity
        raw_tortuosity = _compute_tortuosity(pos_cpu, start_idx, end_idx, exit_cpu,
                                              baselines['straightness'])
        norm_tortuosity = min(1.0, raw_tortuosity)

        # 2. Node-based metrics — fully vectorized fast path
        raw_node = _compute_node_metrics_fast(
            run_ids_all, start_idx, end_idx, exit_ds, maze, baselines
        )

        norm_occupancy = min(1.0, raw_node['occupancy'])
        norm_turn_bias = min(1.0, raw_node['turn_bias'])
        norm_markov    = min(1.0, raw_node['markov'] / 6.0)

        results.append(FitnessResult(
            total=0.0,
            markov=norm_markov,
            occupancy=norm_occupancy,
            tortuosity=norm_tortuosity,
            turn_bias=norm_turn_bias,
        ))

    # Calculate total fitness
    for res in results:
        m_c = res.markov    * weights.markov.weight    if weights.markov.enabled    else 0.0
        o_c = res.occupancy * weights.occupancy.weight if weights.occupancy.enabled else 0.0
        t_c = res.tortuosity* weights.tortuosity.weight if weights.tortuosity.enabled else 0.0
        b_c = res.turn_bias * weights.turn_bias.weight  if weights.turn_bias.enabled  else 0.0

        res.total      = m_c + o_c + t_c + b_c
        res.markov     = m_c
        res.occupancy  = o_c
        res.tortuosity = t_c
        res.turn_bias  = b_c

    return results


def _compute_tortuosity(pos_history, start_idx, end_idx, exit_frames, mouse_straightness):
    """Compute tortuosity metric across multiple bouts."""
    bouts_trajectories = []
    
    for b in range(start_idx, end_idx):
        T = min(exit_frames[b] // DOWNSAMPLE_FACTOR, len(pos_history))
        if T > 2:
            bouts_trajectories.append(pos_history[:T, b, :])
    
    if not bouts_trajectories:
        return 1.0 # Max penalty
    
    agent_straightness = compute_straightness(bouts_trajectories)
    baseline = max(mouse_straightness, 0.1)
    return abs(baseline - agent_straightness) / baseline


def _compute_node_metrics(pos_history, start_idx, end_idx, exit_frames, maze, baselines):
    """Compute node-based metrics: Markov, occupancy, turn bias."""
    
    # Extract node sequence
    node_seq = [] # Flat sequence for Occupancy
    bouts = []    # List of Bouts for Markov
    
    for b in range(start_idx, end_idx):
        T = min(exit_frames[b] // DOWNSAMPLE_FACTOR, len(pos_history))
        path = pos_history[:T, b, :]
        
        bout_seq = []
        current_node = -1
        
        for p in path:
            if not np.isfinite(p[0]) or not np.isfinite(p[1]):
                continue
            
            nx, ny = int(round(p[0])), int(round(p[1]))
            node_id = maze.cell_lookup.get((nx, ny), -1)
            
            if node_id >= 0:
                run_id = maze.run_lookup[node_id]
                
                if run_id != current_node:
                    bout_seq.append(run_id)
                    current_node = run_id
                
                # Log entry for every frame (account for downsampling)
                # This ensures consistent weighting with mouse data duration
                node_seq.extend([run_id] * DOWNSAMPLE_FACTOR)
                    
        if len(bout_seq) > 0:
            # Prepend/Append virtual exit node (n_runs)
            n_runs = len(maze.runs)
            bout_aug = [n_runs] + bout_seq + [n_runs]
            bouts.append(bout_aug)
    
    if len(node_seq) < 3:
        # Failure Penalty: Max distance for all metrics
        return {'markov': 6.0, 'occupancy': 2.0, 'turn_bias': 1.0}
    
    # 1. Markov Score
    # Note: preprocess_mouse.py MUST have been run to generate 'markov_profile' in baselines
    mouse_profile = baselines.get('markov_profile', None)
    
    if mouse_profile is None:
        raise ValueError(
            "Mouse Markov profile not found. Run 'python preprocess_mouse.py <tf_file>' first."
        )
    markov_score = calculate_markov_distance(bouts, mouse_profile, maze) 
    
    
    # Jensen-Shannon Divergence for Occupancy
    # JS is symmetric and bounded in [0, ln(2)] for probability distributions
    agent_pdf = compute_node_pdf(node_seq, len(maze.runs))
    mouse_pdf = baselines.get('node_pdf', np.zeros_like(agent_pdf))
    
    # Ensure same length
    sz = min(len(agent_pdf), len(mouse_pdf))
    p = agent_pdf[:sz]
    q = mouse_pdf[:sz]
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    
    # Renormalize after clipping
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Compute JS divergence
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    js_divergence = 0.5 * kl_pm + 0.5 * kl_qm
    
    # JS is bounded in [0, ln(2)] ≈ [0, 0.693]
    # Scale to [0, 1] for normalization consistency
    occupancy_diff = js_divergence / np.log(2)
    
    # 3. Turn bias (Signed)
    agent_bias = compute_turn_bias(node_seq, maze)
    mouse_bias = baselines.get('turn_bias', 0.0)
    turn_bias_diff = abs(agent_bias - mouse_bias)
    
    return {
        'markov': markov_score,
        'occupancy': occupancy_diff,
        'turn_bias': turn_bias_diff
    }
