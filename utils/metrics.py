"""
Shared behavioral metric definitions for parity between mouse and agent analysis.
"""
import numpy as np
from core.constants import DOWNSAMPLE_FACTOR, TORTUOSITY_WINDOW

def compute_node_pdf(node_history, n_runs):
    """
    Compute time-weighted node occupancy PDF.
    node_history: List or array of node IDs (one entry per frame or weighted)
    """
    counts = np.zeros(n_runs)
    for n in node_history:
        if 0 <= n < n_runs:
            counts[n] += 1
    return counts / (np.sum(counts) + 1e-9)

def compute_straightness(trajectories, window=None):
    """
    Compute average path straightness.
    window: defaults to scaled TORTUOSITY_WINDOW
    """
    if window is None:
        window = max(2, TORTUOSITY_WINDOW // DOWNSAMPLE_FACTOR)
        
    values = []
    for traj in trajectories:
        if len(traj) <= window:
            continue
        # Displacement / Path Length
        disp = np.linalg.norm(traj[window:] - traj[:-window], axis=1)
        steps = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
        path_length = np.cumsum(np.insert(steps, 0, 0.0))[window:] - np.cumsum(np.insert(steps, 0, 0.0))[:-window]
        
        valid = path_length > 0.1
        if np.any(valid):
            values.append(np.mean(disp[valid] / (path_length[valid] + 1e-6)))
            
    return np.mean(values) if values else 0.5

def compute_turn_bias(node_seq, maze):
    """
    Compute signed turn bias (-1 left, +1 right) from a node sequence.
    """
    n_left = 0
    n_right = 0
    
    for i in range(len(node_seq) - 1):
        c_curr, c_next = node_seq[i], node_seq[i+1]
        
        if c_curr != c_next:
            if c_curr < maze.st.shape[0] and c_next < maze.st.shape[1]:
                s_type = maze.st[c_curr, c_next]
                if s_type in [0, 2]: n_left += 1
                elif s_type in [1, 3]: n_right += 1
                
    total = n_left + n_right
    return (n_right - n_left) / total if total > 0 else 0.0
