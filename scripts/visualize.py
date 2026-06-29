"""
Visualization module for neuroevolution experiments.
Matches original visualization style with dual-panel MP4 output.

Usage:
    python visualize.py best 1              # Animate best agent from gen 1
    python visualize.py fitness             # Plot fitness history
    python visualize.py maze                # Draw maze structure
"""

import os
import sys

# Make project root (config, utils, core) and scripts/ (sibling `run` module)
# importable when invoked as `python scripts/visualize.py` from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

        
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import networkx as nx
import os
import argparse

from config import load_config
from utils.maze import Maze, create_maze, OccupancyGrid
from utils.backend import xp, to_cpu
from core.agent import Agent
from core.simulation import Simulation
from run import load_mouse_baselines, generate_noise_matrix


# --- Layout Configuration ---
X_SPREAD = 2.5
Y_SPREAD_S = 2.0
Y_SPREAD_M = 1.0
Y_SPREAD_I = 2.5


def get_node_pos(G):
    """Deterministic node layout for the agent network."""
    S = [n for n in G.nodes if G.nodes[n]['type'] == 'sensory']
    M = [n for n in G.nodes if G.nodes[n]['type'] == 'motor']
    I = [n for n in G.nodes if G.nodes[n]['type'] == 'inter']
    
    S.sort(key=lambda x: G.nodes[x]['idx'])
    M.sort(key=lambda x: G.nodes[x]['idx'])
    I.sort(key=lambda x: G.nodes[x]['idx'])
    
    pos = {}
    
    # Sensory (Left)
    ys = np.linspace(Y_SPREAD_S/2, -Y_SPREAD_S/2, len(S)) if len(S) > 1 else [0.0]
    for i, n in enumerate(S): pos[n] = np.array([-X_SPREAD, ys[i]])
    
    # Interneurons (Center)
    ys = np.linspace(Y_SPREAD_I/2, -Y_SPREAD_I/2, len(I)) if len(I) > 1 else [0.0]
    for i, n in enumerate(I): pos[n] = np.array([0.0, ys[i]])
    
    # Motor (Right)
    ys = np.linspace(Y_SPREAD_M/2, -Y_SPREAD_M/2, len(M)) if len(M) > 1 else [0.0]
    for i, n in enumerate(M): pos[n] = np.array([X_SPREAD, ys[i]])
    
    return pos


def draw_network_base(G, pos, ax, with_labels=True):
    """Draws a premium scaffolding of the network."""
    edges = G.edges(data=True)
    
    exc_edges = [(u, v) for u, v, d in edges if d['weight'] > 0]
    inh_edges = [(u, v) for u, v, d in edges if d['weight'] < 0]
    exc_weights = [d['weight'] for u, v, d in edges if d['weight'] > 0]
    inh_weights = [abs(d['weight']) for u, v, d in edges if d['weight'] < 0]
    
    def scale_w(ws): return [1.0 + 4.0 * w for w in ws]
    
    # Draw Edges with better styling
    if exc_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=exc_edges, width=scale_w(exc_weights),
                               edge_color='#ff4444', alpha=0.3, arrowstyle='-|>', arrowsize=18, 
                               connectionstyle='arc3,rad=0.3')
    if inh_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=inh_edges, width=scale_w(inh_weights),
                               edge_color='#4444ff', alpha=0.3, arrowstyle='-[', arrowsize=18, 
                               connectionstyle='arc3,rad=0.3')
    
    # Draw Nodes with a "glass" or "glow" look
    nS = [n for n in G.nodes if G.nodes[n]['type'] == 'sensory']
    nI = [n for n in G.nodes if G.nodes[n]['type'] == 'inter']
    nM = [n for n in G.nodes if G.nodes[n]['type'] == 'motor']
    
    def get_color(nlist):
        return ['#ff8888' if G.nodes[n].get('node_type', 1) == 1 else '#8888ff' for n in nlist]

    # Draw shadows/glow
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nS + nI + nM, node_color='white', 
                           node_size=800, alpha=0.1, linewidths=0)
    
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nS, node_color=get_color(nS), 
                           node_shape='s', node_size=600, edgecolors='#333333', linewidths=2)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nI, node_color=get_color(nI), 
                           node_shape='o', node_size=500, edgecolors='#333333', linewidths=2)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nM, node_color=get_color(nM), 
                           node_shape='^', node_size=600, edgecolors='#333333', linewidths=2)
    
    # Headers with better typography
    font_props = {'color': '#333333', 'weight': 'bold', 'size': 14}
    ax.text(-X_SPREAD, 1.9, "SENSORY", ha='center', **font_props)
    ax.text(0.0, 1.9, "INTERNEURONS", ha='center', **font_props)
    ax.text(X_SPREAD, 1.9, "MOTOR", ha='center', **font_props)


def simulate_for_viz(agent, simulation, config, baselines, max_frames=2000, seed=42):
    """
    Simulate agent and record all data needed for visualization.
    
    Returns:
        dict with trajectory, headings, activations, rays
    """
    # Ensure agent is on the correct device and configured for single agent
    agent.weights = xp.array(agent.weights)
    agent.batch_size = 1
    agent.reset()
    
    pos = xp.array([[-0.5, 7.0]], dtype=xp.float64)
    heading = xp.array([0.0], dtype=xp.float64)
    
    prev_speed = xp.zeros(1, dtype=xp.float64)
    prev_turn = xp.zeros(1, dtype=xp.float64)
    actual_speed = xp.zeros(1, dtype=xp.float64)
    actual_turn = xp.zeros(1, dtype=xp.float64)
    
    # Physics override from mouse data
    physics = config.physics
    m_phys = baselines.get('physics', {})
    p_max_speed = m_phys.get('max_speed', physics.max_speed)
    p_max_turn = m_phys.get('max_turn_rate', physics.max_turn_rate)
    p_alpha_speed = m_phys.get('alpha_speed', physics.alpha_speed)
    p_alpha_turn = m_phys.get('alpha_turn', physics.alpha_turn)
    
    # Generate noise for the specified bout index (seed corresponds to bout index)
    # This ensures parity with training where bout i uses RandomState(i)
    noise = np.random.RandomState(seed).uniform(-1, 1, max_frames)
    noise = xp.array(noise)
    
    trajectory = []
    headings_list = []
    activations = []
    rays = []
    
    for t in range(max_frames):
        if pos[0, 0] < -1.0:
            break
        
        trajectory.append(to_cpu(pos[0]).copy())
        headings_list.append(float(to_cpu(heading)[0]))
        
        # Raycast
        d_f = simulation.raycast(pos, heading)
        d_l = simulation.raycast(pos, heading + np.pi / 2)
        d_r = simulation.raycast(pos, heading - np.pi / 2)
        
        rays.append([float(to_cpu(d_f)[0]), float(to_cpu(d_l)[0]), float(to_cpu(d_r)[0])])
        
        # Build inputs (consistent with fitness.py)
        # Using xp for inputs to maintain precision parity
        inputs = xp.zeros((1, 6), dtype=xp.float64)
        inputs[0, 0] = d_f[0] / 10.0
        inputs[0, 1] = d_l[0] / 10.0
        inputs[0, 2] = d_r[0] / 10.0
        inputs[0, 3] = prev_speed[0]
        inputs[0, 4] = prev_turn[0]
        inputs[0, 5] = noise[t]
        
        # Forward pass and record activations
        out = agent.forward(inputs)
        activations.append(to_cpu(agent.state[0]).copy())
        
        raw_speed = out[0, 0]
        raw_turn = out[0, 1]
        
        # Apply physics with split momentum
        target_turn = raw_turn * p_max_turn
        actual_turn = (1 - p_alpha_turn) * actual_turn + p_alpha_turn * target_turn
        heading = heading + actual_turn
        heading = (heading + xp.pi) % (2 * xp.pi) - xp.pi
        
        target_speed = raw_speed * p_max_speed
        actual_speed = (1 - p_alpha_speed) * actual_speed + p_alpha_speed * target_speed
        
        step = xp.zeros((1, 2), dtype=xp.float64)
        step[0, 0] = actual_speed[0] * xp.cos(heading[0])
        step[0, 1] = actual_speed[0] * xp.sin(heading[0])
        
        pos, _ = simulation.step(pos, step)
        
        prev_speed = actual_speed / p_max_speed
        prev_turn = actual_turn / p_max_turn
    
    return {
        'trajectory': np.array(trajectory),
        'headings': np.array(headings_list),
        'activations': np.array(activations),
        'rays': np.array(rays),
        'weights': to_cpu(agent.weights)
    }


def create_combined_animation(agent, simulation, config, baselines, output_path, limit_frames=2000, fitness_data=None, seed=42):
    """
    Generate side-by-side MP4 (Maze + Network Activity).
    
    Args:
        agent: Agent object
        simulation: Simulation environment
        config: Config object
        output_path: Output file path (.mp4)
        limit_frames: Max frames to animate
        fitness_data: Optional dict of fitness scores to display
    """
    print("Simulating trajectory...")
    data = simulate_for_viz(agent, simulation, config, baselines, limit_frames, seed=seed)
    
    pos_data = data['trajectory']
    activations = data['activations']
    headings = data['headings']
    ray_data = data['rays']
    
    # Build network graph
    G = agent.to_networkx()
    pos = get_node_pos(G)
    nodes = list(G.nodes())
    node_to_idx = {n: G.nodes[n]['idx'] for n in nodes}
    s_nodes = sorted([n for n in nodes if G.nodes[n]['type'] == 'sensory'], key=lambda x: G.nodes[x]['idx'])
    m_nodes = sorted([n for n in nodes if G.nodes[n]['type'] == 'motor'], key=lambda x: G.nodes[x]['idx'])
    
    # Maze walls
    walls = simulation.maze.walls
    
    N_FRAMES = min(limit_frames, len(activations))
    print(f"Creating animation ({N_FRAMES} frames)...")
    
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    # AX1: Maze
    ax_maze = fig.add_subplot(gs[0])
    ax_maze.set_title("Maze Simulation")
    ax_maze.set_aspect('equal')
    ax_maze.plot(walls[:, 0], walls[:, 1], 'k-', linewidth=1.5, alpha=0.5, label='Walls')
    
    if fitness_data:
        # Display fitness metrics
        metrics_text = (
            f"Fitness: {fitness_data.get('fitness', 0):.2f}\n"
            f"Markov: {fitness_data.get('markov_score', 0):.2f}\n"
            f"Occupancy: {fitness_data.get('occupancy_score', 0):.2f}\n"
            f"Tortuosity: {fitness_data.get('tortuosity_score', 0):.2f}\n"
            f"Turn Bias: {fitness_data.get('turn_bias_score', 0):.2f}"
        )
        ax_maze.text(0.02, 0.98, metrics_text, transform=ax_maze.transAxes,
                     verticalalignment='top', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    agent_dot, = ax_maze.plot([], [], 'ro', markersize=8, label='Agent')
    ray_lines = [ax_maze.plot([], [], 'g-', linewidth=1, alpha=0.6)[0] for _ in range(3)]
    
    ax_maze.set_xlim(walls[:, 0].min() - 2, walls[:, 0].max() + 2)
    ax_maze.set_ylim(walls[:, 1].min() - 2, walls[:, 1].max() + 2)
    ax_maze.legend(loc='upper right')
    
    # AX2: Network
    ax_net = fig.add_subplot(gs[1])
    ax_net.set_title("Network Activity")
    ax_net.axis('off')
    ax_net.set_xlim(-3.5, 3.5)
    ax_net.set_ylim(-2.0, 2.0)
    
    draw_network_base(G, pos, ax_net, with_labels=True)
    
    # Dynamic nodes overlay
    xy = np.array([pos[n] for n in nodes])
    init_colors = np.zeros(len(nodes))
    nodes_scatter = ax_net.scatter(xy[:, 0], xy[:, 1], s=400, c=init_colors, 
                                    cmap='coolwarm', vmin=-1, vmax=1, edgecolors='k', zorder=10)
    
    # Labels
    custom_labels = {
        'S0': 'F', 'S1': 'L', 'S2': 'R', 'S3': 'P_Spd', 'S4': 'P_Trn', 'S5': 'N',
        'M0': 'Speed', 'M1': 'Turn'
    }
    
    for n in nodes:
        label_text = custom_labels.get(n, n)
        ntype = "E" if G.nodes[n].get('node_type', 1) == 1 else "I"
        ax_net.text(pos[n][0], pos[n][1] + 0.35, f"{label_text}({ntype})", ha='center', va='bottom', 
                   fontsize=8, color='black', fontweight='bold')
    
    val_texts = {}
    for n in s_nodes:
        val_texts[n] = ax_net.text(pos[n][0] - 0.2, pos[n][1], '0.00', ha='right', 
                                   va='center', fontsize=9, color='green')
    for n in m_nodes:
        val_texts[n] = ax_net.text(pos[n][0] + 0.2, pos[n][1], '0.00', ha='left', 
                                   va='center', fontsize=9, color='purple')
    
    def update(frame):
        p = pos_data[frame]
        agent_dot.set_data([p[0]], [p[1]])
        
        h = headings[frame]
        dists = ray_data[frame]
        angles = [h, h + np.pi / 2, h - np.pi / 2]
        for i in range(3):
            end_x = p[0] + np.cos(angles[i]) * dists[i]
            end_y = p[1] + np.sin(angles[i]) * dists[i]
            ray_lines[i].set_data([p[0], end_x], [p[1], end_y])
        
        st = activations[frame]
        colors = [st[node_to_idx[n]] for n in nodes]
        nodes_scatter.set_array(np.array(colors))
        
        for n in s_nodes:
            val_texts[n].set_text(f"{st[node_to_idx[n]]:.2f}")
        for n in m_nodes:
            val_texts[n].set_text(f"{st[node_to_idx[n]]:.2f}")
        
        if frame % 100 == 0:
            print(f"Rendering frame {frame}/{N_FRAMES}", flush=True)
        
        return [agent_dot, nodes_scatter] + ray_lines + list(val_texts.values())
    
    ani = animation.FuncAnimation(fig, update, frames=range(0, N_FRAMES), interval=20, blit=True)
    
    # Use FFmpeg for MP4
    if output_path.endswith('.gif'):
        output_path = output_path.replace('.gif', '.mp4')
        print(f"Switched to MP4 for performance: {output_path}")
    
    writer = animation.FFMpegWriter(fps=30)
    print(f"Saving to {output_path}...")
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print("Done!")


def _mouse_id_from_meta(results_dir, generation):
    """Read config.mouse_id from meta.pkl for a given generation, or None."""
    meta_path = os.path.join(results_dir, f'gen_{generation}', 'meta.pkl')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return getattr(meta.get('config'), 'mouse_id', None)


def visualize_best(generation, results_dir='results', limit_frames=2000, seed=42):
    """Animate the best agent from a given generation."""
    config = load_config()

    summary_path = os.path.join(results_dir, f'gen_{generation}', 'summary.pkl')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No results found at {summary_path}")

    with open(summary_path, 'rb') as f:
        results = pickle.load(f)

    best = min(results, key=lambda x: x['fitness'])
    print(f"Best agent: ID={best['id']}, fitness={best['fitness']:.4f}")

    mouse_id = _mouse_id_from_meta(results_dir, generation)
    baselines = load_mouse_baselines(mouse_id)
    agent = best['agent']
    simulation = Simulation(config.physics)
    
    output_path = os.path.join(results_dir, f'gen_{generation}', 'best_agent.mp4')
    create_combined_animation(agent, simulation, config, baselines, output_path, limit_frames=limit_frames, fitness_data=best, seed=seed)
    print(f"Saved: {output_path}")


def plot_fitness_history(results_dir='results', output_path=None):
    """Plot fitness evolution over generations."""
    generations = []
    best_fitness = []
    mean_fitness = []
    
    gen_dirs = [d for d in os.listdir(results_dir) if d.startswith('gen_')]
    # Sort numerically by generation number
    gen_dirs.sort(key=lambda x: int(x.split('_')[1]))
    
    num_dirs = len(gen_dirs)
    print(f"Found {num_dirs} generations to load...")
    
    for i, gen_dir in enumerate(gen_dirs):
        if i % 5 == 0 or i == num_dirs - 1:
            print(f"\rLoading generation {i+1}/{num_dirs}...", end="", flush=True)
        gen_num = int(gen_dir.split('_')[1])
        summary_path = os.path.join(results_dir, gen_dir, 'summary.pkl')
        
        if os.path.exists(summary_path):
            with open(summary_path, 'rb') as f:
                results = pickle.load(f)
            
            fitnesses = [r['fitness'] for r in results]
            generations.append(gen_num)
            best_fitness.append(min(fitnesses))
            mean_fitness.append(np.mean(fitnesses))
    
    if not generations:
        print("\nNo generation data found!")
        return

    print("\nPlotting results...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, best_fitness, 'b-', linewidth=2, label='Best')
    ax.plot(generations, mean_fitness, 'r--', linewidth=1, alpha=0.7, label='Mean')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (lower is better)')
    ax.set_title('Fitness Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_path is None:
        output_path = os.path.join(results_dir, 'fitness_history.png')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_top(generation, results_dir='results', top_k=5, limit_frames=2000, seed=42):
    """Animate the top K agents from a given generation."""
    config = load_config()
    
    summary_path = os.path.join(results_dir, f'gen_{generation}', 'summary.pkl')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No results found at {summary_path}")
    
    with open(summary_path, 'rb') as f:
        results = pickle.load(f)
    
    # Sort by fitness (lower is better for this problem?)
    # Assuming lower fitness is better based on previous context (min(fitnesses))
    top_agents = sorted(results, key=lambda x: x['fitness'])[:top_k]

    mouse_id = _mouse_id_from_meta(results_dir, generation)
    baselines = load_mouse_baselines(mouse_id)
    simulation = Simulation(config.physics)
    
    for i, data in enumerate(top_agents):
        rank = i + 1
        agent_id = data['id']
        fitness = data['fitness']
        print(f"  Rank {rank}: Agent {agent_id} (Fitness: {fitness:.4f})")
        
        agent = data['agent']
        output_path = os.path.join(results_dir, f'gen_{generation}', f'top_{rank}_agent_{agent_id}.mp4')
        
        create_combined_animation(agent, simulation, config, baselines, output_path, limit_frames=limit_frames, fitness_data=data, seed=seed)
        print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualization tools")
    parser.add_argument('command', choices=['best', 'fitness', 'maze', 'top'],
                        help="Command: best, fitness, maze, or top")
    parser.add_argument('generation', type=int, nargs='?', default=1,
                        help="Generation number (for 'best'/'top' command)")
    parser.add_argument('--results', default='results', help="Results directory")
    parser.add_argument('--dir', default=None, help="Results directory (alias for --results)")
    parser.add_argument('--output', '-o', default=None, help="Output path")
    parser.add_argument('--frames', type=int, default=2000, help="Max frames to animate")
    parser.add_argument('--k', type=int, default=5, help="Number of top agents to visualize (for 'top' command)")
    parser.add_argument('--seed', type=int, default=42, help="Noise seed for simulation (default: 42)")
    
    args = parser.parse_args()
    
    # Handle alias
    if args.dir:
        args.results = args.dir
        
    config = load_config()
    
    # Centralized global seeding based on experimental seed
    from utils.backend import set_seed
    set_seed(config.seed)
    
    if args.command == 'best':
        visualize_best(args.generation, args.results, args.frames, seed=args.seed)

    elif args.command == 'top':
        visualize_top(args.generation, args.results, args.k, args.frames, seed=args.seed)
    
    elif args.command == 'fitness':
        plot_fitness_history(args.results, args.output)
    
    elif args.command == 'maze':
        maze = create_maze(6)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(maze.walls[:, 0], maze.walls[:, 1], 'k-', linewidth=1.5)
        ax.set_xlim(-2, 15)
        ax.set_ylim(-1, 15)
        ax.set_aspect('equal')
        ax.set_title('6-Level Binary Maze')
        
        output_path = args.output or 'maze.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
