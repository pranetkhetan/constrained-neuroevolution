"""
Evolutionary operators: selection, mutation, and population management.

Implements:
- Elitist selection (top N% survive)
- Synaptic scaling (weight mutation to neighboring values)
- Structural plasticity (connection pruning and growth)
"""
import numpy as np
from typing import List
from utils.backend import to_cpu, xp
from core.agent import Agent


def initialize_population(config) -> List[Agent]:
    """
    Create initial random population.
    
    Args:
        config: Full Config object
        
    Returns:
        List of randomly initialized agents
    """
    population = []
    for _ in range(config.population.size):
        agent = Agent(config.network)
        population.append(agent)
    return population


def select_elites(population: List[Agent], 
                  fitness_scores: List[float], 
                  config) -> List[Agent]:
    """
    Select top-performing agents.
    
    Args:
        population: List of agents
        fitness_scores: Fitness for each agent (lower is better)
        config: Config with elite_fraction
        
    Returns:
        List of elite agents (copies)
    """
    n_elite = int(len(population) * config.population.elite_fraction)
    n_elite = max(1, n_elite)
    
    # Sort by fitness (lower = better)
    sorted_indices = np.argsort(fitness_scores)
    elite_indices = sorted_indices[:n_elite]
    
    return [population[i].copy() for i in elite_indices]


def mutate_weights(agent: Agent, mutation_rate: float):
    """
    Synaptic scaling: change weights to neighboring quantized values.
    
    Args:
        agent: Agent to mutate (modified in place)
        mutation_rate: Probability of mutating each connection
    """
    weights = to_cpu(agent.weights)
    rows, cols = np.nonzero(weights)
    
    for r, c in zip(rows, cols):
        if np.random.random() < mutation_rate:
            current_w = weights[r, c]
            mag = abs(current_w)
            sign = np.sign(current_w)
            
            # Swap between magnitudes [0.25, 1.0]
            new_mag = 1.0 if mag == 0.25 else 0.25
            weights[r, c] = new_mag * sign
    
    agent.weights = xp.array(weights)


def mutate_structure(agent: Agent, rewiring_rate: float):
    """
    Structural plasticity: prune and grow connections.
    
    Args:
        agent: Agent to mutate (modified in place)
        rewiring_rate: Probability of rewiring operations
    """
    weights = to_cpu(agent.weights)
    n_total = agent.n_total
    
    # Pruning
    rows, cols = np.nonzero(weights)
    for r, c in zip(rows, cols):
        if np.random.random() < rewiring_rate:
            weights[r, c] = 0.0
    
    # Growth
    n_attempts = int(n_total * n_total * rewiring_rate)
    # Efference copies: Sensory, Inter, AND Motor can be sources
    valid_sources = np.concatenate((agent.idx_sensory, agent.idx_inter, agent.idx_motor))
    valid_targets = np.concatenate((agent.idx_inter, agent.idx_motor))
    
    for _ in range(n_attempts):
        u = np.random.choice(valid_sources)
        v = np.random.choice(valid_targets)
        
        if weights[u, v] != 0:
            continue
        
        n_out = np.count_nonzero(weights[u, :])
        n_in = np.count_nonzero(weights[:, v])
        
        if n_out < agent.max_outgoing and n_in < agent.max_incoming:
            # New connections inherit sign from source node type
            mag = np.random.choice(agent.weight_magnitudes)
            weights[u, v] = mag * agent.node_types[u]
    
    agent.weights = xp.array(weights)


def mutate(agent: Agent, config) -> Agent:
    """
    Apply all mutations to create offspring.
    """
    offspring = agent.copy()
    
    # 1. Node Type Mutation (E/I flip)
    node_type_rate = getattr(config.mutation, 'node_type_rate', 0.05)
    # Only Interneurons and Motor neurons can flip
    mutable_nodes = np.concatenate((offspring.idx_inter, offspring.idx_motor))
    
    for node_idx in mutable_nodes:
        if np.random.random() < node_type_rate:
            # Flip E -> I or I -> E
            offspring.node_types[node_idx] *= -1
            # MUST flip all outgoing weights to maintain Dale's Law
            offspring.weights[node_idx, :] *= -1
    
    # 2. Weight Magnitude Mutation
    mutate_weights(offspring, config.mutation.weight_rate)
    
    # 3. Structural Rewiring
    mutate_structure(offspring, config.mutation.structure_rate)
    
    return offspring


def create_next_generation(elites: List[Agent], 
                           population_size: int, 
                           config) -> List[Agent]:
    """
    Create next generation from elites.
    
    Args:
        elites: List of elite agents
        population_size: Target population size
        config: Full config object
        
    Returns:
        New population (elites + mutated offspring)
    """
    next_gen = [e.copy() for e in elites]  # Elites survive unchanged
    
    n_offspring = population_size - len(elites)
    for _ in range(n_offspring):
        parent = np.random.choice(elites)
        offspring = mutate(parent, config)
        next_gen.append(offspring)
    
    return next_gen
