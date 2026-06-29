#!/usr/bin/env python
"""
Preprocess mouse trajectory data to generate baseline metrics.

Usage:
    python preprocess_mouse.py <tf_file>

Examples:
    python preprocess_mouse.py D3-tf            # File in current directory
    python preprocess_mouse.py ./data/B1-tf     # File in data folder

Output (saved to ./data/):
    - mouse_<name>.pkl: Trajectory object
    - mouse_metrics.pkl: Baseline metrics (node_pdf, straightness)
"""
import argparse
import os
import sys

# Make project root (utils, config, core) importable when invoked as
# `python scripts/preprocess_mouse.py` from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pickle
import numpy as np
from pathlib import Path
from dataclasses import make_dataclass
from utils.maze import create_maze
from utils.markov import compute_bias_profile
from utils.metrics import compute_node_pdf, compute_straightness, compute_turn_bias

# Recreate the Traj dataclass to match Rosenberg format
Traj = make_dataclass('Traj', ['fr', 'ce', 'ke', 'no', 're'])
Traj.__module__ = __name__


def load_tf_file(path: str):
    """Load a Rosenberg -tf trajectory file."""
    print(f"Loading: {path}")
    
    class TrajUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'Traj':
                return Traj
            return super().find_class(module, name)
    
    with open(path, 'rb') as f:
        tr = TrajUnpickler(f).load()
    
    print(f"  Bouts: {len(tr.ce) if tr.ce else 0}")
    print(f"  Has keypoints: {tr.ke is not None}")
    print(f"  Has nodes: {tr.no is not None}")
    
    return tr


def extract_trajectories(tr, maze=None):
    """
    Extract (x, y) trajectories scaled to maze coordinates.
    Returns:
        raw_trajectories: List of (N, 2) arrays (nose keypoints, may contain NaNs)
        logical_trajectories: List of (N, 2) arrays (repaired cell centers, clean)
    """
    raw_trajectories = []
    logical_trajectories = []
    
    # 1. Repaired cell data (ce) - zero NaNs, Snapped to grid
    if tr.ce is not None and maze is not None:
        for ce_bout in tr.ce:
            x = maze.xc[ce_bout]
            y = maze.yc[ce_bout]
            logical_trajectories.append(np.column_stack([x, y]))
    
    # 2. Raw keypoints (ke) - contains NaNs, continuous
    if tr.ke is not None:
        for ke in tr.ke:
            if len(ke.shape) == 3 and ke.shape[1] == 1:
                xy = ke[:, 0, :]
            else:
                xy = ke
            scaled = -0.5 + 15.0 * xy
            raw_trajectories.append(scaled)
            
    return raw_trajectories, logical_trajectories


# Metric implementations now reside in utils/metrics.py


def compute_momentum_params(trajectories):
    """
    Compute max mobility, median cruising, and inertia (alpha) from trajectories.
    Uses temporal smoothing to remove tracking jitter from turn analysis.
    """
    all_speeds = []
    all_turns = []
    all_v_corr = []
    all_t_corr = []
    
    from scipy.ndimage import gaussian_filter1d
    
    for traj in trajectories:
        if len(traj) < 20: continue
        
        # 1. Smooth the trajectory to remove jitter
        # Only smooth finite segments
        mask = np.isfinite(traj).all(axis=1)
        if not np.any(mask): continue
        
        # Sub-select finite segments or handle NaNs? 
        # For physics, let's just use segments between NaNs.
        for seg in np.split(traj, np.where(~mask)[0]):
            seg = seg[np.isfinite(seg).all(axis=1)] # remove the NaN divider
            if len(seg) < 10: continue
            
            # Smooth mildly (sigma=1.0)
            smooth_seg = gaussian_filter1d(seg, sigma=1.0, axis=0)
            
            # Velocity and Speed
            vel = np.diff(smooth_seg, axis=0)
            speeds = np.linalg.norm(vel, axis=1)
            
            # Heading and Turns
            headings = np.arctan2(vel[:, 1], vel[:, 0])
            unwrapped = np.unwrap(headings)
            turns = np.diff(unwrapped)
            
            # Filter for "stable" movement (Speed > 0.1)
            stable_mask = speeds > 0.1
            robust_turn_mask = stable_mask[:-1] & stable_mask[1:]
            
            all_speeds.extend(speeds[stable_mask])
            if np.any(robust_turn_mask):
                physical_turns = turns[robust_turn_mask]
                physical_turns = physical_turns[np.abs(physical_turns) < np.pi/2]
                all_turns.extend(physical_turns)
                
                # Compute Correlations
                if len(speeds[stable_mask]) > 10:
                    s_valid = speeds[stable_mask]
                    c_v = np.corrcoef(s_valid[:-1], s_valid[1:])[0, 1]
                    if not np.isnan(c_v): all_v_corr.append(c_v)
                
                if len(physical_turns) > 10:
                    c_t = np.corrcoef(physical_turns[:-1], physical_turns[1:])[0, 1]
                    if not np.isnan(c_t): all_t_corr.append(c_t)

    # Aggregate with robust nan-aware functions
    res = {
        'max_speed': float(np.percentile(all_speeds, 99)) if all_speeds else 1.0,
        'median_speed': float(np.median(all_speeds)) if all_speeds else 0.1,
        'max_turn_rate': float(np.percentile(np.abs(all_turns), 99)) if all_turns else 1.1,
        'alpha_speed': float(1.0 - np.median(all_v_corr)) if all_v_corr else 0.5,
        'alpha_turn': float(1.0 - np.median(all_t_corr)) if all_t_corr else 0.5
    }
    
    # Clip alphas to [0.01, 1.0]
    res['alpha_speed'] = max(0.01, min(1.0, res['alpha_speed']))
    res['alpha_turn'] = max(0.01, min(1.0, res['alpha_turn']))
    
    return res


def main():
    parser = argparse.ArgumentParser(description="Preprocess mouse -tf file(s)")
    parser.add_argument('path', help="Path to -tf file or directory containing -tf files")
    parser.add_argument('--output', '-o', default='data', help="Output directory")
    
    args = parser.parse_args()
    
    # Identify files to process
    input_path = Path(args.path)
    if input_path.is_dir():
        # Look for both -tf and _tf suffixes
        files = list(input_path.glob('*-tf')) + list(input_path.glob('*_tf'))
        if not files:
            # Try to just get all files if no suffixes match
            files = [f for f in input_path.iterdir() if f.is_file()]
    else:
        files = [input_path]

    if not files:
        print(f"No files found at {args.path}")
        return

    print(f"Found {len(files)} subjects to process.")
    
    all_metrics = []
    processed_subjects = []
    
    for f_path in files:
        print(f"\nProcessing Subject: {f_path.name}")
        try:
            tr = load_tf_file(str(f_path))
            # Create maze for context
            maze = create_maze(6)
            
            # Extract trajectories
            raw_trajectories, logical_trajectories = extract_trajectories(tr, maze=maze)
            
            # Downsample trajectories (Consistency with simulation downsampling)
            # DOWNSAMPLE_FACTOR = 5 from constants
            ds_factor = 5
            # Use logical (repaired) for behavior metrics
            ds_logical = [t[::ds_factor] for t in logical_trajectories]
            # Use raw for physics extraction (with smoothing inside momentum function)
            
            print(f"\nExtracted {len(logical_trajectories)} bouts")
            print(f"Applying {ds_factor}x downsampling for behavioral metrics.")
            
            # Compute metrics on downsampled data
            print("\nComputing metrics...")
            
            # Node PDF (Occupancy) - Rosenberg's convention: bout_no[n, 1] is the
            # START frame of node n within the bout. Dwell time of node n is the
            # gap to the next entry's start frame. The final node of each bout has
            # no successor and contributes zero dwell (matches Rosenberg's PDF code).
            visit_frames = []
            if tr.no is not None:
                n_runs = len(maze.runs)
                for bout_no in tr.no:
                    for n in range(len(bout_no) - 1):
                        node_id = int(bout_no[n, 0])
                        duration = int(bout_no[n + 1, 1]) - int(bout_no[n, 1])
                        if duration > 0 and 0 <= node_id < n_runs:
                            visit_frames.extend([node_id] * duration)
            node_pdf = compute_node_pdf(visit_frames, len(maze.runs))
            
            straightness = compute_straightness(ds_logical, window=20 // ds_factor)
            
            # Turn Bias - Extract transition sequence (runs)
            # tr.no already stores run IDs (0-126), not cell IDs.
            # Do NOT apply maze.run_lookup here — that maps cell IDs to run IDs
            # and would corrupt values that are already run IDs.
            n_runs = len(maze.runs)
            node_transitions = []
            if tr.no is not None:
                for bout_no in tr.no:
                    prev_run = -1
                    for node_entry in bout_no:
                        run_id = int(node_entry[0])
                        if 0 <= run_id < n_runs and run_id != prev_run:
                            node_transitions.append(run_id)
                            prev_run = run_id
                                
            turn_bias = compute_turn_bias(node_transitions, maze)
            
            # PHYSICS Extraction from RAW (smoothed)
            momentum = compute_momentum_params(raw_trajectories) 
            
            # Compute Markov Profile (Bias Profile)
            # Extract bouts as list of integer lists
            bouts = []
            n_runs = len(maze.runs)
            if tr.no is not None:
                for bout_no in tr.no:
                    b = [int(entry[0]) for entry in bout_no]
                    # Prepend/Append virtual exit node to capture start/end transitions
                    b_aug = [n_runs] + b + [n_runs]
                    bouts.append(b_aug)
            
            try:
                markov_profile = compute_bias_profile(bouts, maze)
            except Exception as e:
                print(f"  Warning: Failed to compute Markov profile for {f_path.name}: {e}")
                markov_profile = np.zeros((6, 3, 2))
            
            per_mouse_metrics = {
                'node_pdf': node_pdf,
                'straightness': straightness,
                'turn_bias': turn_bias,
                'markov_profile': markov_profile,
                'physics': momentum
            }
            all_metrics.append(per_mouse_metrics)
            processed_subjects.append(f_path.name)

            # Save per-mouse metrics dict (same format as aggregate, loadable by run.py --mouse)
            mouse_name = f_path.stem.replace('-tf', '').replace('_tf', '')
            os.makedirs(args.output, exist_ok=True)
            per_mouse_path = os.path.join(args.output, f"mouse_{mouse_name}_metrics.pkl")
            with open(per_mouse_path, 'wb') as f:
                pickle.dump(per_mouse_metrics, f)
            print(f"  Saved per-mouse metrics: {per_mouse_path}")

            # Also save individual trajectory for separate analysis if single file
            if len(files) == 1:
                traj_path = os.path.join(args.output, f"mouse_{mouse_name}.pkl")
                with open(traj_path, 'wb') as f:
                    pickle.dump(tr, f)
                print(f"  Saved individual trajectory: {traj_path}")
                
        except Exception as e:
            print(f"  Error processing {f_path.name}: {e}")

    if not all_metrics:
        print("No valid metrics collected from any subject.")
        return

    # Aggregate across subjects
    print("\n" + "="*50)
    print(f"AGGREGATING RESULTS ACROSS {len(all_metrics)} SUBJECTS")
    print("="*50)
    
    # Behavior metrics (Mean)
    agg_node_pdf = np.mean([m['node_pdf'] for m in all_metrics], axis=0)
    agg_straightness = np.mean([m['straightness'] for m in all_metrics])
    agg_turn_bias = np.mean([m['turn_bias'] for m in all_metrics])
    agg_markov = np.mean([m['markov_profile'] for m in all_metrics], axis=0)
    
    # Characterize Spread
    std_straightness = np.std([m['straightness'] for m in all_metrics])
    std_turn_bias = np.std([m['turn_bias'] for m in all_metrics])
    
    # Average Distance from Mean (Characterizing "Typicality")
    occ_distances = [np.sum((m['node_pdf'] - agg_node_pdf)**2) for m in all_metrics]
    markov_distances = [np.linalg.norm(m['markov_profile'] - agg_markov) for m in all_metrics]
    
    avg_occ_dist = np.mean(occ_distances)
    std_occ_dist = np.std(occ_distances)
    avg_mar_dist = np.mean(markov_distances)
    std_mar_dist = np.std(markov_distances)

    print(f"  Straightness:  {agg_straightness:.3f} (± {std_straightness:.3f})")
    print(f"  Turn Bias:     {agg_turn_bias:.3f} (± {std_turn_bias:.3f})")
    print(f"  Occupancy (SSE from Mean):      Avg={avg_occ_dist:.6f} | Std={std_occ_dist:.6f}")
    print(f"  Markov (Euclidean from Mean):   Avg={avg_mar_dist:.4f} | Std={std_mar_dist:.4f}")
    
    print("-" * 50)
    # Physics parameters (Median)
    phys_keys = ['max_speed', 'median_speed', 'max_turn_rate', 'alpha_speed', 'alpha_turn']
    agg_physics = {}
    for k in phys_keys:
        agg_physics[k] = np.median([m['physics'][k] for m in all_metrics])

    print(f"  Straightness (Mean):  {agg_straightness:.3f}")
    print(f"  Turn Bias (Mean):     {agg_turn_bias:.3f}")
    print(f"  Max Speed (99th):     {agg_physics['max_speed']:.3f} units/frame")
    print(f"  Median Speed (Cruise): {agg_physics['median_speed']:.3f} units/frame")
    print(f"  Max Turn (Median):    {agg_physics['max_turn_rate']:.3f} rad/frame")
    print(f"  Alpha Speed (Median): {agg_physics['alpha_speed']:.3f}")
    print(f"  Alpha Turn (Median):  {agg_physics['alpha_turn']:.3f}")

    # Final Package
    final_metrics = {
        'node_pdf': agg_node_pdf, 
        'straightness': agg_straightness, 
        'turn_bias': agg_turn_bias,
        'markov_profile': agg_markov,
        'physics': agg_physics,
        'subjects': processed_subjects
    }
    
    # Save
    os.makedirs(args.output, exist_ok=True)
    metrics_path = os.path.join(args.output, "mouse_metrics.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(final_metrics, f)
    
    print(f"\nAggregated Metrics Saved to: {metrics_path}")
    print(f"Integrated {len(processed_subjects)} subjects into baseline.")
    print("\nRun experiment with: python run.py init")


if __name__ == '__main__':
    main()
