"""
Neuroevolution Mouse Navigation Experiment

A minimal, publication-ready implementation for evolving neural network agents
to navigate a binary maze in a mouse-like manner.

Usage:
    python run.py init              # Initialize generation 1
    python run.py evolve --gen N    # Run generation N
    python run.py loop --start 2 --end 100  # Run multiple generations
"""
# Enable cuBLAS deterministic mode BEFORE any CUDA imports
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
# Add project root to sys.path so `config`, `utils`, `core` resolve when this
# script is invoked as `python scripts/run.py` (sys.path[0] would otherwise be
# scripts/). Mirrors the bootstrap used in every other script in this folder.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import pickle
import numpy as np
from datetime import datetime

from config import load_config
from utils.backend import xp, free_memory, HAS_GPU
from core.agent import Agent
from core.simulation import Simulation
from core.evolution import (
    initialize_population, 
    select_elites, 
    create_next_generation
)
from core.fitness import evaluate_batch


def generate_noise_matrix(n_bouts: int, max_frames: int):
    """Generate deterministic noise for reproducibility."""
    noise = np.zeros((n_bouts, max_frames), dtype=np.float64)
    
    for seed_idx in range(n_bouts):
        # Local random state seeded with seed_idx
        local_rng = np.random.RandomState(seed_idx)
        noise[seed_idx] = local_rng.uniform(-1, 1, size=max_frames)
    
    return xp.array(noise)


def load_mouse_baselines(mouse_id=None):
    """Load precomputed mouse behavior baselines.

    Args:
        mouse_id: If set (e.g. 'D3', 'B5'), loads per-mouse metrics from
                  data/mouse_{mouse_id}_metrics.pkl. Otherwise loads the
                  population-aggregated data/mouse_metrics.pkl.
    """
    baselines = {
        'node_pdf': np.zeros(128),
        'revisit_rate': 0.38,
        'straightness': 0.8,
        'turn_bias': 0.5,
        'markov_profile': None
    }

    # Try to load from data directory (same directory as run.py)
    if mouse_id:
        metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', f'mouse_{mouse_id}_metrics.pkl')
    else:
        metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'mouse_metrics.pkl')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            data = pickle.load(f)
            baselines['node_pdf'] = data.get('node_pdf', baselines['node_pdf'])
            baselines['revisit_rate'] = data.get('reversal_rate', baselines['revisit_rate'])
            # Load additional metrics if available
            baselines['straightness'] = data.get('straightness', baselines['straightness'])
            baselines['turn_bias'] = data.get('turn_bias', baselines['turn_bias'])
            baselines['markov_profile'] = data.get('markov_profile', baselines['markov_profile'])
            baselines['physics'] = data.get('physics', {})
            
            print(f"Loaded baselines from {metrics_path}")
            print(f"  - Turn Bias: {baselines['turn_bias']:.3f}")
            if baselines['markov_profile'] is not None:
                print(f"  - Markov Profile: {baselines['markov_profile'].shape}")
    
    return baselines


def save_generation(population, fitness_results, generation, config):
    """Save generation results."""
    output_dir = os.path.join(config.output_dir, f'gen_{generation}')
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for i, (agent, fitness) in enumerate(zip(population, fitness_results)):
        results.append({
            'id': i + 1,
            'fitness': fitness.total,
            'markov_score': fitness.markov,
            'occupancy_score': fitness.occupancy,
            'tortuosity_score': fitness.tortuosity,
            'turn_bias_score': fitness.turn_bias,
            'agent': agent
        })
    
    summary_path = os.path.join(output_dir, 'summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Log metadata
    meta = {
        'generation': generation,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'gpu_available': HAS_GPU,
        'best_fitness': min(r['fitness'] for r in results),
        'mean_fitness': np.mean([r['fitness'] for r in results])
    }
    
    meta_path = os.path.join(output_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Generation {generation} saved to {output_dir}")
    return results


def load_generation(generation, config):
    """Load previous generation."""
    summary_path = os.path.join(config.output_dir, f'gen_{generation}', 'summary.pkl')
    with open(summary_path, 'rb') as f:
        results = pickle.load(f)
    
    population = [r['agent'] for r in results]
    fitness_scores = [r['fitness'] for r in results]
    return population, fitness_scores


def run_init(config):
    """Initialize generation 1 with random population."""
    print(f"=== Initializing Generation 1 ===")
    print(f"Population size: {config.population.size}")
    print(f"GPU available: {HAS_GPU}")
    
    # Global seed already set in main()
    
    # Create environment
    simulation = Simulation(config.physics)
    noise_matrix = generate_noise_matrix(config.simulation.n_bouts, config.simulation.max_frames)
    baselines = load_mouse_baselines(config.mouse_id)

    # Create random population
    population = initialize_population(config)
    print(f"Created {len(population)} random agents")

    # Evaluate
    print("Evaluating population...")
    fitness_results = evaluate_batch(
        population, simulation, config, baselines, noise_matrix
    )

    # Save
    results = save_generation(population, fitness_results, 1, config)

    # Report
    fitnesses = [r.total for r in fitness_results]
    print(f"\nGeneration 1 Complete")
    print(f"Best fitness: {min(fitnesses):.4f}")
    print(f"Mean fitness: {np.mean(fitnesses):.4f}")

    free_memory()


def run_evolve(generation, config):
    """Run one generation of evolution."""
    print(f"=== Generation {generation} ===")
    
    # Load previous generation
    prev_population, prev_fitness = load_generation(generation - 1, config)
    print(f"Loaded {len(prev_population)} agents from generation {generation - 1}")
    
    # DEBUG: Track the best agent from previous generation
    best_prev_idx = np.argmin(prev_fitness)
    best_prev_fitness = prev_fitness[best_prev_idx]
    best_prev_agent = prev_population[best_prev_idx]
    best_prev_weights_hash = hash(best_prev_agent.weights.tobytes())
    print(f"  [DEBUG] Previous best: idx={best_prev_idx}, fitness={best_prev_fitness:.4f}, weights_hash={best_prev_weights_hash}")
    
    # Select elites
    elites = select_elites(prev_population, prev_fitness, config)
    print(f"Selected {len(elites)} elites")
    
    # DEBUG: Verify best agent is in elites
    elite_hashes = [hash(e.weights.tobytes()) for e in elites]
    if best_prev_weights_hash in elite_hashes:
        elite_pos = elite_hashes.index(best_prev_weights_hash)
        print(f"  [DEBUG] Previous best IS in elites at position {elite_pos}")
    else:
        print(f"  [DEBUG] WARNING: Previous best NOT in elites!")
    
    # Create next generation
    population = create_next_generation(elites, config.population.size, config)
    print(f"Created population of {len(population)}")
    
    # DEBUG: Track where best agent ended up in new population
    pop_hashes = [hash(a.weights.tobytes()) for a in population]
    if best_prev_weights_hash in pop_hashes:
        pop_pos = pop_hashes.index(best_prev_weights_hash)
        print(f"  [DEBUG] Previous best is at population index {pop_pos}")
    else:
        print(f"  [DEBUG] WARNING: Previous best NOT in new population!")
    
    # Evaluate
    simulation = Simulation(config.physics)
    noise_matrix = generate_noise_matrix(config.simulation.n_bouts, config.simulation.max_frames)
    baselines = load_mouse_baselines(config.mouse_id)

    print("Evaluating population...")
    fitness_results = evaluate_batch(
        population, simulation, config, baselines, noise_matrix
    )
    
    # DEBUG: Check re-evaluated fitness of previous best
    if best_prev_weights_hash in pop_hashes:
        pop_pos = pop_hashes.index(best_prev_weights_hash)
        reeval_fitness = fitness_results[pop_pos].total
        fitness_delta = reeval_fitness - best_prev_fitness
        print(f"  [DEBUG] Previous best re-evaluated: {reeval_fitness:.4f} (delta={fitness_delta:+.4f})")
        if abs(fitness_delta) > 0.0001:
            print(f"  [DEBUG] *** NON-DETERMINISM DETECTED! ***")
    
    # Save
    results = save_generation(population, fitness_results, generation, config)
    
    # Report
    fitnesses = [r.total for r in fitness_results]
    print(f"\nGeneration {generation} Complete")
    print(f"Best fitness: {min(fitnesses):.4f}")
    print(f"Mean fitness: {np.mean(fitnesses):.4f}")
    
    free_memory()


def calculate_mutation_rate(gen, config):
    """Calculate linear scheduled mutation rate."""
    schedule = config.mutation.schedule
    if not schedule or not schedule.enabled:
        return None
    
    if gen >= schedule.end_gen:
        return schedule.end_rate
    
    # Linear interpolation: y = mx + c
    # (2, start) -> (end_gen, end)
    # Because mutation actually starts being applied when creating Gen 2
    start_decay_gen = 2
    
    if gen < start_decay_gen:
        return schedule.start_rate
        
    slope = (schedule.end_rate - schedule.start_rate) / (schedule.end_gen - start_decay_gen)
    rate = slope * (gen - start_decay_gen) + schedule.start_rate
    return max(0.0, rate)


def run_loop(start_gen, end_gen, config):
    """Run multiple generations in a loop."""
    print(f"=== Running Generations {start_gen} to {end_gen} ===")
    
    base_weight_rate = config.mutation.weight_rate
    base_struct_rate = config.mutation.structure_rate
    base_node_rate = config.mutation.node_type_rate

    for gen in range(start_gen, end_gen + 1):
        # Apply mutation scheduling if enabled
        scheduled_rate = calculate_mutation_rate(gen, config)
        if scheduled_rate is not None:
            config.mutation.weight_rate = scheduled_rate
            config.mutation.structure_rate = scheduled_rate
            config.mutation.node_type_rate = scheduled_rate
            print(f"  [Schedule] Gen {gen}: Mutation Rate set to {scheduled_rate:.4f} ({scheduled_rate*100:.1f}%)")
            
        run_evolve(gen, config)
        print()
    
    # Restore base rates
    config.mutation.weight_rate = base_weight_rate
    config.mutation.structure_rate = base_struct_rate
    config.mutation.node_type_rate = base_node_rate


def main():
    parser = argparse.ArgumentParser(
        description="Neuroevolution Mouse Navigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py init                        # Initialize generation 1
  python run.py evolve --gen 5              # Run generation 5
  python run.py loop --start 2 --end 100    # Run generations 2-100
  python run.py init --ablate markov        # Ablation: disable markov metric
  python run.py loop --start 1 --end 50 --pop 200  # Custom population size

Ablation Metrics:
  markov, occupancy, tortuosity, turn_bias
        """
    )
    parser.add_argument('command', choices=['init', 'evolve', 'loop'],
                        help="Command to run: init, evolve, or loop")
    parser.add_argument('--config', default='config.yaml',
                        help="Path to config YAML file (default: config.yaml)")
    parser.add_argument('--gen', type=int, default=2,
                        help="Generation number for 'evolve' command (default: 2)")
    parser.add_argument('--start', type=int, default=2,
                        help="Start generation for 'loop' command (default: 2)")
    parser.add_argument('--end', type=int, default=100,
                        help="End generation for 'loop' command (default: 100)")
    parser.add_argument('--pop', type=int, default=None,
                        help="Override population size from config")
    parser.add_argument('--ablate', type=str, nargs='+', default=[],
                        help="Metrics to ablate (disable): markov, occupancy, spin, etc.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Override random seed from config")
    parser.add_argument('--dir', type=str, default=None,
                        help="Override output directory (default: from config)")
    parser.add_argument('--mouse', type=str, default=None,
                        help="Mouse ID (e.g. D3, B5). Loads data/mouse_{ID}_metrics.pkl")

    args = parser.parse_args()

    # Load and modify config
    from config import load_config, apply_ablations
    config = load_config(args.config)

    # Apply CLI overrides
    if args.mouse:
        config.mouse_id = args.mouse
        print(f"Override: mouse_id = {args.mouse}")

    if args.dir:
        config.output_dir = args.dir
        print(f"Override: output_dir = {args.dir}")
    
    # Apply CLI overrides
    if args.pop:
        config.population.size = args.pop
        print(f"Override: population.size = {args.pop}")
    
    if args.seed is not None:
        config.seed = args.seed
        print(f"Override: seed = {args.seed}")
    
    # Set global seed for all operations
    from utils.backend import set_seed
    set_seed(config.seed)
    
    # Apply ablations
    if args.ablate:
        config = apply_ablations(config, args.ablate)
    
    # Print active metrics
    print("\nActive fitness metrics:")
    for name in ['markov', 'occupancy', 'tortuosity', 'turn_bias']:
        metric = getattr(config.fitness, name)
        status = "[ON]" if metric.enabled else "[OFF]"
        print(f"  {status} {name}: weight={metric.weight}")
    print()
    
    if args.command == 'init':
        run_init(config)
    elif args.command == 'evolve':
        run_evolve(args.gen, config)
    elif args.command == 'loop':
        if args.start == 2 and args.gen != 2:
            # If --gen specified but --start wasn't, use --gen as start
            args.start = args.gen
        run_loop(args.start, args.end, config)


if __name__ == '__main__':
    main()
