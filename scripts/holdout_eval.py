#!/usr/bin/env python
"""
R9 Holdout Evaluation — Cross-bout generalization test.

Splits each mouse's trajectory bouts into two halves:
  half1 (bouts 0..N//2-1)  — original training target
  half2 (bouts N//2..N-1)  — held-out, NEVER seen during training

Evaluates all 54 existing evolved agents against held-out baselines.
Computes held-out 9×9 specialization matrix and ratio.

If held-out ratio ≈ training ratio (0.656) → no overfitting.
If held-out ratio → 1.0 → specialization was bout-specific.

Usage:
    python scripts/holdout_eval.py --tf_dir data/raw --agents_dir data/agents

Output:
    analysis/R9_holdout_matrix.npy      — (9,9) held-out cross-eval matrix
    analysis/R9_holdout_results.pkl     — full result dict
    figures/supp_R9_holdout.pdf         — comparison figure
"""

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import make_dataclass
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

MICE      = ['B5', 'B6', 'B7', 'D3', 'D4', 'D5', 'D7', 'D8', 'D9']
GEN       = 150
N_REPS    = 6
FIGURE_DPI = 150

# ---------------------------------------------------------------------------
# Traj dataclass (mirrors Rosenberg format)
# ---------------------------------------------------------------------------
Traj = make_dataclass('Traj', ['fr', 'ce', 'ke', 'no', 're'])
Traj.__module__ = __name__


def load_tf_file(path):
    """Load a Rosenberg -tf trajectory file."""
    class TrajUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'Traj':
                return Traj
            return super().find_class(module, name)
    with open(path, 'rb') as f:
        tr = TrajUnpickler(f).load()
    n_bouts = len(tr.no) if tr.no is not None else 0
    print(f'  Loaded {Path(path).name}: {n_bouts} bouts')
    return tr


# ---------------------------------------------------------------------------
# Baseline computation (mirrors preprocess_mouse.py logic)
# ---------------------------------------------------------------------------

def compute_metrics_from_bouts(tr_no, tr_ce, tr_ke, maze):
    """
    Compute behavioral baseline metrics from a subset of bouts.

    Args:
        tr_no : list of bouts (each bout = list of (node_id, end_frame))
        tr_ce : list of bouts (each bout = (N,) array of cell indices)
        tr_ke : list of bouts (each bout = (N, 1, 2) or (N, 2) keypoint array)
        maze  : Maze object from create_maze(6)

    Returns:
        dict with keys: node_pdf, straightness, turn_bias, markov_profile, physics
    """
    from utils.maze import create_maze
    from utils.markov import compute_bias_profile
    from utils.metrics import compute_node_pdf, compute_straightness, compute_turn_bias

    ds_factor = 5

    # ── logical trajectories (for straightness) ──────────────────────────────
    logical_trajectories = []
    if tr_ce is not None:
        for ce_bout in tr_ce:
            x = maze.xc[ce_bout]
            y = maze.yc[ce_bout]
            logical_trajectories.append(np.column_stack([x, y]))
    ds_logical = [t[::ds_factor] for t in logical_trajectories]

    # ── raw trajectories (for physics) ───────────────────────────────────────
    raw_trajectories = []
    if tr_ke is not None:
        for ke in tr_ke:
            if len(ke.shape) == 3 and ke.shape[1] == 1:
                xy = ke[:, 0, :]
            else:
                xy = ke
            raw_trajectories.append(-0.5 + 15.0 * xy)

    # ── node_pdf ─────────────────────────────────────────────────────────────
    # Rosenberg's convention: bout_no[n, 1] is the START frame of node n within
    # the bout. Dwell time of node n = next entry's start frame - this start frame.
    # The final node of each bout has no successor and contributes zero dwell.
    visit_frames = []
    if tr_no is not None:
        n_runs = len(maze.runs)
        for bout_no in tr_no:
            for n in range(len(bout_no) - 1):
                node_id  = int(bout_no[n, 0])
                duration = int(bout_no[n + 1, 1]) - int(bout_no[n, 1])
                if duration > 0 and 0 <= node_id < n_runs:
                    visit_frames.extend([node_id] * duration)
    node_pdf = compute_node_pdf(visit_frames, len(maze.runs))

    # ── straightness ─────────────────────────────────────────────────────────
    straightness = compute_straightness(ds_logical, window=20 // ds_factor) \
                   if ds_logical else 0.8

    # ── turn_bias ────────────────────────────────────────────────────────────
    # tr_no already stores run IDs (0-126), not cell IDs.
    # Do NOT apply maze.run_lookup — that maps cell IDs to run IDs and would
    # corrupt values that are already run IDs.
    n_runs = len(maze.runs)
    node_transitions = []
    if tr_no is not None:
        for bout_no in tr_no:
            prev_run = -1
            for node_entry in bout_no:
                run_id = int(node_entry[0])
                if 0 <= run_id < n_runs and run_id != prev_run:
                    node_transitions.append(run_id)
                    prev_run = run_id
    turn_bias = compute_turn_bias(node_transitions, maze)

    # ── markov_profile ───────────────────────────────────────────────────────
    n_runs = len(maze.runs)
    bouts  = []
    if tr_no is not None:
        for bout_no in tr_no:
            b     = [int(entry[0]) for entry in bout_no]
            b_aug = [n_runs] + b + [n_runs]
            bouts.append(b_aug)
    try:
        markov_profile = compute_bias_profile(bouts, maze)
    except Exception as e:
        print(f'    Warning: Markov profile failed ({e}); using zeros')
        markov_profile = np.zeros((6, 3, 2))

    # ── physics ──────────────────────────────────────────────────────────────
    from preprocess_mouse import compute_momentum_params
    physics = compute_momentum_params(raw_trajectories) if raw_trajectories else {
        'max_speed': 1.0, 'median_speed': 0.1, 'max_turn_rate': 1.1,
        'alpha_speed': 0.5, 'alpha_turn': 0.5
    }

    return {
        'node_pdf':      node_pdf,
        'straightness':  straightness,
        'turn_bias':     turn_bias,
        'markov_profile': markov_profile,
        'physics':       physics,
    }


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('cupy'):
            module = module.replace('cupy._core.core', 'numpy').replace('cupy', 'numpy')
        return super().find_class(module, name)


def _load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        with open(path, 'rb') as f:
            return _CpuUnpickler(f).load()


def find_results_dir(mouse, rep, agents_dir=None):
    """Find the results directory for a given mouse and rep, checking multiple locations."""
    candidates = []
    if agents_dir:
        candidates.append(os.path.join(agents_dir, f'results_{mouse}_r{rep}'))
    candidates += [
        os.path.join(BASE_DIR, 'data', 'agents', f'results_{mouse}_r{rep}'),
        os.path.join(BASE_DIR, 'data',    f'results_{mouse}_r{rep}'),
        os.path.join(BASE_DIR,            f'results_{mouse}_r{rep}'),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def load_best_agent(mouse, rep, gen=GEN, agents_dir=None):
    """Load best agent from completed run."""
    results_dir = find_results_dir(mouse, rep, agents_dir)
    if results_dir is None:
        print(f'  WARNING: results dir not found for {mouse} r{rep}')
        return None, None
    summary_path = os.path.join(results_dir, f'gen_{gen}', 'summary.pkl')
    if not os.path.exists(summary_path):
        print(f'  WARNING: {summary_path} not found')
        return None, None
    pop  = _load_pickle(summary_path)
    best = min(pop, key=lambda r: r['fitness'])
    return best['agent'], best['fitness']


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_holdout_eval(tf_dir='data/raw', agents_dir=None,
                     split_fraction=0.5, gen=GEN):
    """
    Full holdout evaluation pipeline.

    Returns:
        dict with holdout_matrix (9×9), train_ratio, holdout_ratio, per_mouse data
    """
    from config import load_config
    from core.simulation import Simulation
    from core.fitness import evaluate_batch
    from utils.backend import xp
    from utils.maze import create_maze

    config_path = os.path.join(BASE_DIR, 'config.yaml')
    config      = load_config(config_path)
    simulation  = Simulation(config.physics)
    maze        = create_maze(6)

    # Noise matrix (same seed as all other evals)
    noise = np.zeros((config.simulation.n_bouts, config.simulation.max_frames), dtype=np.float64)
    for k in range(config.simulation.n_bouts):
        local_rng = np.random.RandomState(k)
        noise[k]  = local_rng.uniform(-1, 1, size=config.simulation.max_frames)
    noise_xp = xp.array(noise)

    # ── Step 1: Load all -tf files and split bouts ────────────────────────────
    print('\n=== Step 1: Loading -tf files and splitting bouts ===')
    tf_dir_path = Path(tf_dir) if os.path.isabs(tf_dir) else Path(BASE_DIR) / tf_dir

    train_baselines  = {}  # mouse → metrics from first half (verify match to saved pkl)
    holdout_baselines = {} # mouse → metrics from second half (never seen)
    bout_counts       = {}

    for mouse in MICE:
        tf_path = tf_dir_path / f'{mouse}-tf'
        if not tf_path.exists():
            tf_path = tf_dir_path / f'{mouse}_tf'
        if not tf_path.exists():
            print(f'  WARNING: -tf file not found for {mouse} in {tf_dir_path}')
            # Fall back to saved baselines for both halves (degrade gracefully)
            saved = os.path.join(BASE_DIR, 'data', f'mouse_{mouse}_metrics.pkl')
            if os.path.exists(saved):
                bl = _load_pickle(saved)
                train_baselines[mouse]   = bl
                holdout_baselines[mouse] = bl
            bout_counts[mouse] = 0
            continue

        tr    = load_tf_file(str(tf_path))
        n     = len(tr.no)
        split = max(1, int(n * split_fraction))
        bout_counts[mouse] = n

        print(f'  {mouse}: {n} bouts → first {split} / last {n - split}')

        # Slice bouts
        no_h1 = tr.no[:split];       no_h2 = tr.no[split:]
        ce_h1 = tr.ce[:split] if tr.ce else None
        ce_h2 = tr.ce[split:] if tr.ce else None
        ke_h1 = tr.ke[:split] if tr.ke else None
        ke_h2 = tr.ke[split:] if tr.ke else None

        print(f'    Computing train baseline (bouts 0–{split-1})...')
        train_baselines[mouse]   = compute_metrics_from_bouts(no_h1, ce_h1, ke_h1, maze)
        print(f'    Computing holdout baseline (bouts {split}–{n-1})...')
        holdout_baselines[mouse] = compute_metrics_from_bouts(no_h2, ce_h2, ke_h2, maze)

    # ── Step 2: Load all best agents ──────────────────────────────────────────
    print('\n=== Step 2: Loading best agents ===')
    agents_by_mouse = {}
    for mouse in MICE:
        agents_by_mouse[mouse] = []
        for rep in range(1, N_REPS + 1):
            agent, fitness = load_best_agent(mouse, rep, gen=gen, agents_dir=agents_dir)
            if agent is not None:
                agents_by_mouse[mouse].append((agent, fitness))
        print(f'  {mouse}: {len(agents_by_mouse[mouse])}/{N_REPS} agents loaded')

    # ── Step 3: Build held-out 9×9 matrix ────────────────────────────────────
    print('\n=== Step 3: Evaluating held-out 9×9 matrix ===')
    # Structure: matrix[i, j] = mean fitness of mouse_i's agents on mouse_j's HELD-OUT baseline
    holdout_matrix = np.zeros((9, 9))
    batch_size     = 18

    for j, target_mouse in enumerate(MICE):
        bl = holdout_baselines[target_mouse]
        print(f'\n  Evaluating all agents on {target_mouse} holdout baseline...')

        all_agents = []
        agent_labels = []  # (mouse_i, rep)
        for i, eval_mouse in enumerate(MICE):
            for agent, _ in agents_by_mouse[eval_mouse]:
                all_agents.append(agent)
                agent_labels.append(i)

        if not all_agents:
            print('    No agents loaded — skipping')
            continue

        # Evaluate in batches
        all_fitness = []
        for b_start in range(0, len(all_agents), batch_size):
            batch  = all_agents[b_start: b_start + batch_size]
            results = evaluate_batch(batch, simulation, config, bl, noise_xp)
            all_fitness.extend([r.total for r in results])

        # Accumulate into matrix
        counts = np.zeros(9)
        for k, (i_mouse, fit) in enumerate(zip(agent_labels, all_fitness)):
            holdout_matrix[i_mouse, j] += fit
            counts[i_mouse]            += 1
        for i in range(9):
            if counts[i] > 0:
                holdout_matrix[i, j] /= counts[i]

        diag_val = holdout_matrix[j, j]
        print(f'    Own-mouse holdout fitness ({target_mouse}): {diag_val:.4f}')

    print('\n=== Holdout matrix ===')
    print(np.round(holdout_matrix, 4))

    # ── Step 4: Compare specialization ratios ────────────────────────────────
    print('\n=== Step 4: Specialization ratios ===')
    diag         = np.diag(holdout_matrix)
    off_diag_all = holdout_matrix[~np.eye(9, dtype=bool)]

    holdout_ratio  = float(np.mean(diag) / np.mean(off_diag_all))
    import numpy as _np; _mat = _np.load(os.path.join(BASE_DIR, 'analysis', 'generalization_matrix.npy')); _n = _mat.shape[0]; training_ratio = float(_np.diag(_mat).mean() / _mat[~_np.eye(_n,dtype=bool)].mean())

    # Per-mouse holdout ratios
    per_mouse_holdout = {}
    for i, mouse in enumerate(MICE):
        own   = holdout_matrix[i, i]
        other = np.mean([holdout_matrix[i, j] for j in range(9) if j != i])
        per_mouse_holdout[mouse] = float(own / other) if other > 0 else 1.0

    print(f'\n  Training ratio (colab_4):  {training_ratio:.4f}')
    print(f'  Holdout ratio  (held-out): {holdout_ratio:.4f}')
    print(f'\n  Per-mouse holdout ratios:')
    for m, r in per_mouse_holdout.items():
        print(f'    {m}: {r:.4f}')

    degradation = (holdout_ratio - training_ratio) / training_ratio * 100
    print(f'\n  Ratio change: {degradation:+.1f}%  '
          f'(< 10% = robust; > 20% = possible overfitting)')

    # ── Step 5: Figure ───────────────────────────────────────────────────────
    os.makedirs(os.path.join(BASE_DIR, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'analysis'), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1: Heatmap
    ax = axes[0]
    im = ax.imshow(holdout_matrix, cmap='RdYlGn_r', aspect='auto',
                   vmin=np.min(holdout_matrix), vmax=np.max(holdout_matrix))
    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.set_xticklabels(MICE, rotation=45, fontsize=7)
    ax.set_yticklabels(MICE, fontsize=7)
    ax.set_xlabel('Target mouse (held-out baseline)'); ax.set_ylabel('Agent mouse')
    ax.set_title('Held-out 9×9 fitness matrix')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: Ratio comparison bars
    ax2 = axes[1]
    labels   = ['Training ratio\n(colab_4)', 'Held-out ratio\n(R9 test)']
    vals     = [training_ratio, holdout_ratio]
    colors   = ['#4e8cd4', '#e07b39' if abs(degradation) < 10 else '#c0392b']
    ax2.bar(range(2), vals, color=colors, alpha=0.85, width=0.5)
    ax2.axhline(1.0, color='black', lw=1, ls='--', alpha=0.5, label='No specialization (1.0)')
    ax2.set_xticks(range(2)); ax2.set_xticklabels(labels)
    ax2.set_ylabel('Specialization ratio (own / other)')
    ax2.set_title('Specialization: training vs held-out')
    ax2.set_ylim(0, 1.2)
    ax2.legend(fontsize=8)
    verdict = f'Δ = {degradation:+.1f}%\n{"Robust" if abs(degradation) < 10 else "Overfitting?"}'
    ax2.text(0.5, max(vals) * 1.07, verdict, ha='center', fontsize=9, color='black')

    # Panel 3: Per-mouse holdout vs training ratios
    ax3 = axes[2]
    # Load training ratios from generalization_matrix.npy if available
    gm_path = os.path.join(BASE_DIR, 'analysis', 'generalization_matrix.npy')
    if os.path.exists(gm_path):
        gm = np.load(gm_path)
        train_per_mouse = {}
        for i, m in enumerate(MICE):
            own   = gm[i, i]
            other = np.mean([gm[i, j] for j in range(9) if j != i])
            train_per_mouse[m] = float(own / other) if other > 0 else 1.0
    else:
        train_per_mouse = {m: training_ratio for m in MICE}

    x = np.arange(len(MICE))
    w = 0.38
    train_vals  = [train_per_mouse[m]   for m in MICE]
    holdout_vals = [per_mouse_holdout[m] for m in MICE]
    ax3.bar(x - w/2, train_vals,   width=w, label='Training',  color='#4e8cd4', alpha=0.85)
    ax3.bar(x + w/2, holdout_vals, width=w, label='Held-out',  color='#e07b39', alpha=0.85)
    ax3.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.4)
    ax3.set_xticks(x); ax3.set_xticklabels(MICE, rotation=45, fontsize=8)
    ax3.set_ylabel('Specialization ratio')
    ax3.set_title('Per-mouse: training vs held-out')
    ax3.legend(fontsize=8)

    plt.tight_layout()
    fig_out = os.path.join(BASE_DIR, 'figures', 'supp_R9_holdout.pdf')
    plt.savefig(fig_out, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(fig_out.replace('.pdf', '.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f'\n  Figure → {fig_out}')

    # ── Save results ─────────────────────────────────────────────────────────
    result = {
        'holdout_matrix':    holdout_matrix,
        'training_ratio':    training_ratio,
        'holdout_ratio':     holdout_ratio,
        'degradation_pct':   degradation,
        'per_mouse_holdout': per_mouse_holdout,
        'bout_counts':       bout_counts,
        'split_fraction':    split_fraction,
    }

    np.save(os.path.join(BASE_DIR, 'analysis', 'R9_holdout_matrix.npy'), holdout_matrix)
    with open(os.path.join(BASE_DIR, 'analysis', 'R9_holdout_results.pkl'), 'wb') as f:
        pickle.dump(result, f)
    print(f'  Results → analysis/R9_holdout_results.pkl')

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='R9 holdout evaluation')
    parser.add_argument('--tf_dir',     default='data/raw',
                        help='Directory containing -tf files')
    parser.add_argument('--agents_dir', default=None,
                        help='Directory containing results_*_r* subdirs (default: auto-detect)')
    parser.add_argument('--split',      type=float, default=0.5,
                        help='Fraction of bouts used as training split (default 0.5)')
    parser.add_argument('--gen',        type=int,   default=150,
                        help='Generation to load agents from (default 150)')
    args = parser.parse_args()

    run_holdout_eval(
        tf_dir=args.tf_dir,
        agents_dir=args.agents_dir,
        split_fraction=args.split,
        gen=args.gen,
    )
