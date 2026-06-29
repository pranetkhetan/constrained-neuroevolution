"""
Generalist Neuroevolution — Evolve agents on ALL 9 mice simultaneously.

Fitness = mean FitnessResult across all 9 mouse baselines.
Same architecture constraints and training settings as per-mouse runs.
6 replicates × 150 generations; saves to data/generalist/results_r{rep}/.

Usage:
    python run_generalist.py init --rep 0
    python run_generalist.py loop --start 2 --end 150 --rep 0
    python run_generalist.py loop --start 1 --end 150 --rep 0  # from scratch

Motivation (from colab_11):
    A true generalist should show NO own-mouse bias when we run Option A
    (colab_10) on its evolved agents. This provides a definitive control
    for the specificity result: specialized agents show 2.66 vs 2.27 Δfit
    (own vs other); generalist agents should be ~flat across all 9 mice.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys

# Make project root (config, utils, core) importable when invoked as
# `python scripts/run_generalist.py` from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import pickle
import numpy as np
from dataclasses import dataclass
from datetime import datetime

from config import load_config
from utils.backend import xp, free_memory, HAS_GPU, set_seed
from core.agent import Agent
from core.simulation import Simulation
from core.evolution import initialize_population, select_elites, create_next_generation
from core.fitness import evaluate_batch, FitnessResult

# All 9 mice used in the paper
MICE = ['B5', 'B6', 'B7', 'D3', 'D4', 'D5', 'D7', 'D8', 'D9']


def generate_noise_matrix(n_bouts: int, max_frames: int):
    """Generate deterministic noise matrix (same seed as per-mouse runs)."""
    noise = np.zeros((n_bouts, max_frames), dtype=np.float64)
    for seed_idx in range(n_bouts):
        local_rng = np.random.RandomState(seed_idx)
        noise[seed_idx] = local_rng.uniform(-1, 1, size=max_frames)
    return xp.array(noise)


def load_all_mouse_baselines():
    """Load behavioral baselines for all 9 mice."""
    defaults = {
        'node_pdf': np.zeros(128),
        'revisit_rate': 0.38,
        'straightness': 0.8,
        'turn_bias': 0.5,
        'markov_profile': None,
    }
    all_baselines = {}
    for mouse_id in MICE:
        bl = dict(defaults)
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', f'mouse_{mouse_id}_metrics.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            bl['node_pdf']       = data.get('node_pdf',       bl['node_pdf'])
            bl['revisit_rate']   = data.get('reversal_rate',  bl['revisit_rate'])
            bl['straightness']   = data.get('straightness',   bl['straightness'])
            bl['turn_bias']      = data.get('turn_bias',      bl['turn_bias'])
            bl['markov_profile'] = data.get('markov_profile', bl['markov_profile'])
            bl['physics']        = data.get('physics', {})
            print(f'  [{mouse_id}] loaded from {path}')
        else:
            print(f'  [{mouse_id}] WARNING: {path} not found, using defaults')
        all_baselines[mouse_id] = bl
    return all_baselines


def mean_fitness_result(results_list):
    """Average a list of FitnessResult across mice into a single FitnessResult.

    The .total field is the mean of per-mouse totals.
    Individual metric fields are also averaged for logging.
    """
    n = len(results_list)
    return FitnessResult(
        total=np.mean([r.total     for r in results_list]),
        markov=np.mean([r.markov   for r in results_list]),
        occupancy=np.mean([r.occupancy for r in results_list]),
        tortuosity=np.mean([r.tortuosity for r in results_list]),
        turn_bias=np.mean([r.turn_bias for r in results_list]),
    )


def evaluate_batch_generalist(agents, simulation, config, all_baselines, noise_matrix):
    """Evaluate agents against ALL 9 mice and return mean FitnessResult per agent.

    For each mouse, runs evaluate_batch() and collects per-agent FitnessResult.
    Then averages across mice for each agent.

    Args:
        agents: List of Agent objects (length P).
        simulation: Simulation environment.
        config: Config object.
        all_baselines: Dict[mouse_id → baselines dict].
        noise_matrix: (n_bouts, max_frames) noise array.

    Returns:
        List of FitnessResult (length P), each the mean across all 9 mice.
    """
    n_agents = len(agents)
    # per_mouse_results[mouse_id] = List[FitnessResult] length n_agents
    per_mouse_results = {}

    for mouse_id in MICE:
        print(f'  Evaluating vs {mouse_id}...', end=' ', flush=True)
        baselines = all_baselines[mouse_id]
        results = evaluate_batch(agents, simulation, config, baselines, noise_matrix)
        per_mouse_results[mouse_id] = results
        mean_total = np.mean([r.total for r in results])
        print(f'mean_fitness={mean_total:.4f}')

    # Average across mice per agent
    generalist_results = []
    for i in range(n_agents):
        per_agent = [per_mouse_results[m][i] for m in MICE]
        generalist_results.append(mean_fitness_result(per_agent))

    return generalist_results


def save_generation(population, fitness_results, generation, output_dir):
    """Save generation results to output_dir/gen_{generation}/summary.pkl."""
    gen_dir = os.path.join(output_dir, f'gen_{generation}')
    os.makedirs(gen_dir, exist_ok=True)

    results = []
    for i, (agent, fitness) in enumerate(zip(population, fitness_results)):
        results.append({
            'id': i + 1,
            'fitness': fitness.total,
            'markov_score': fitness.markov,
            'occupancy_score': fitness.occupancy,
            'tortuosity_score': fitness.tortuosity,
            'turn_bias_score': fitness.turn_bias,
            'agent': agent,
        })

    summary_path = os.path.join(gen_dir, 'summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(results, f)

    meta = {
        'generation': generation,
        'timestamp': datetime.now().isoformat(),
        'output_dir': output_dir,
        'gpu_available': HAS_GPU,
        'best_fitness': min(r['fitness'] for r in results),
        'mean_fitness': np.mean([r['fitness'] for r in results]),
        'n_mice': len(MICE),
        'mice': MICE,
    }
    with open(os.path.join(gen_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f'Generation {generation} saved to {gen_dir}')
    return results


def load_generation(generation, output_dir):
    """Load previous generation from output_dir/gen_{generation}/summary.pkl."""
    summary_path = os.path.join(output_dir, f'gen_{generation}', 'summary.pkl')
    with open(summary_path, 'rb') as f:
        results = pickle.load(f)
    population = [r['agent'] for r in results]
    fitness_scores = [r['fitness'] for r in results]
    return population, fitness_scores


def run_init(config, output_dir, all_baselines):
    """Initialize generation 1 with random population."""
    print(f'=== Generalist Init — Generation 1 ===')
    print(f'Population size: {config.population.size}')
    print(f'Mice: {MICE}')
    print(f'GPU available: {HAS_GPU}')

    simulation = Simulation(config.physics)
    noise_matrix = generate_noise_matrix(config.simulation.n_bouts, config.simulation.max_frames)

    population = initialize_population(config)
    print(f'Created {len(population)} random agents')

    print('Evaluating population (all 9 mice)...')
    fitness_results = evaluate_batch_generalist(
        population, simulation, config, all_baselines, noise_matrix
    )

    save_generation(population, fitness_results, 1, output_dir)

    fitnesses = [r.total for r in fitness_results]
    print(f'\nGeneration 1 Complete')
    print(f'Best fitness: {min(fitnesses):.4f}')
    print(f'Mean fitness: {np.mean(fitnesses):.4f}')
    free_memory()


def run_evolve(generation, config, output_dir, all_baselines):
    """Run one generation of generalist evolution."""
    print(f'=== Generalist Generation {generation} ===')

    prev_population, prev_fitness = load_generation(generation - 1, output_dir)
    print(f'Loaded {len(prev_population)} agents from generation {generation - 1}')

    elites = select_elites(prev_population, prev_fitness, config)
    print(f'Selected {len(elites)} elites')

    population = create_next_generation(elites, config.population.size, config)
    print(f'Created population of {len(population)}')

    simulation = Simulation(config.physics)
    noise_matrix = generate_noise_matrix(config.simulation.n_bouts, config.simulation.max_frames)

    print('Evaluating population (all 9 mice)...')
    fitness_results = evaluate_batch_generalist(
        population, simulation, config, all_baselines, noise_matrix
    )

    save_generation(population, fitness_results, generation, output_dir)

    fitnesses = [r.total for r in fitness_results]
    print(f'\nGeneration {generation} Complete')
    print(f'Best fitness: {min(fitnesses):.4f}')
    print(f'Mean fitness: {np.mean(fitnesses):.4f}')
    free_memory()


def run_loop(start_gen, end_gen, config, output_dir, all_baselines):
    """Run multiple generations of generalist evolution."""
    print(f'=== Running Generalist Generations {start_gen} to {end_gen} ===')
    for gen in range(start_gen, end_gen + 1):
        run_evolve(gen, config, output_dir, all_baselines)
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Generalist Neuroevolution — evolve on all 9 mice simultaneously.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_generalist.py init --rep 0
  python run_generalist.py loop --start 2 --end 150 --rep 0
  python run_generalist.py loop --start 1 --end 150 --rep 0  # init+run together

Output: data/generalist/results_r{rep}/gen_{gen}/summary.pkl
        """
    )
    parser.add_argument('command', choices=['init', 'evolve', 'loop'])
    parser.add_argument('--rep',   type=int, default=0, help='Replicate index (0-5)')
    parser.add_argument('--gen',   type=int, default=2, help='Generation for evolve')
    parser.add_argument('--start', type=int, default=2, help='Start gen for loop')
    parser.add_argument('--end',   type=int, default=150, help='End gen for loop')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--pop',   type=int, default=None, help='Override population size')
    parser.add_argument('--seed',  type=int, default=None, help='Override random seed')
    args = parser.parse_args()

    config = load_config(args.config)

    # Output dir: data/generalist/results_r{rep}/
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'data', 'generalist', f'results_r{args.rep}'
    )
    print(f'Output directory: {output_dir}')

    if args.pop:
        config.population.size = args.pop
        print(f'Override: population.size = {args.pop}')

    if args.seed is not None:
        config.seed = args.seed
        print(f'Override: seed = {args.seed}')

    set_seed(config.seed)

    print('\nActive fitness metrics:')
    for name in ['markov', 'occupancy', 'tortuosity', 'turn_bias']:
        metric = getattr(config.fitness, name)
        status = '[ON]' if metric.enabled else '[OFF]'
        print(f'  {status} {name}: weight={metric.weight}')
    print()

    print('Loading baselines for all 9 mice...')
    all_baselines = load_all_mouse_baselines()
    print()

    if args.command == 'init':
        run_init(config, output_dir, all_baselines)
    elif args.command == 'evolve':
        run_evolve(args.gen, config, output_dir, all_baselines)
    elif args.command == 'loop':
        if args.start == 1:
            # Start from scratch: init gen 1, then loop 2..end
            run_init(config, output_dir, all_baselines)
            if args.end >= 2:
                run_loop(2, args.end, config, output_dir, all_baselines)
        else:
            run_loop(args.start, args.end, config, output_dir, all_baselines)


if __name__ == '__main__':
    main()
