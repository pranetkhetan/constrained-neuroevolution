"""
Configuration loader and dataclass.
"""
import yaml
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path


@dataclass
class NetworkConfig:
    n_interneurons: int
    weights: list
    max_incoming: int
    max_outgoing: int


@dataclass
class PopulationConfig:
    size: int
    elite_fraction: float


@dataclass
class SimulationConfig:
    max_frames: int
    n_bouts: int


@dataclass
class MutationScheduleConfig:
    enabled: bool
    start_rate: float
    end_rate: float
    end_gen: int
    # Allow optional if user doesn't use schedule
    # But dataclasses require defs. load_config will handle defaults if missing in yaml?
    # Actually, simplistic load_config uses **data. We'll ensure yaml has it or handle safely.


@dataclass
class MutationConfig:
    weight_rate: float
    structure_rate: float
    node_type_rate: float = 0.05
    schedule: MutationScheduleConfig = None  # Optional


@dataclass
class FitnessMetricConfig:
    enabled: bool
    weight: float


@dataclass
class FitnessConfig:
    markov: 'FitnessMetricConfig'
    occupancy: 'FitnessMetricConfig'
    tortuosity: 'FitnessMetricConfig'
    turn_bias: 'FitnessMetricConfig'


@dataclass
class PhysicsConfig:
    max_speed: float
    max_turn_rate: float
    alpha_speed: float
    alpha_turn: float
    momentum_alpha: float = 0.1 # deprecated


@dataclass
class Config:
    name: str
    seed: int
    population: PopulationConfig
    simulation: SimulationConfig
    network: NetworkConfig
    mutation: MutationConfig
    physics: PhysicsConfig
    fitness: FitnessConfig
    output_dir: str
    mouse_id: str = None  # Per-mouse override (e.g. 'D3', 'B5')

def load_config(path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Parse fitness config with enabled flags
    fitness_data = data['fitness']
    fitness_config = FitnessConfig(
        markov=FitnessMetricConfig(**fitness_data['markov']),
        occupancy=FitnessMetricConfig(**fitness_data['occupancy']),
        tortuosity=FitnessMetricConfig(**fitness_data['tortuosity']),
        turn_bias=FitnessMetricConfig(**fitness_data['turn_bias']),
    )

    # Parse mutation schedule if present (backward compatibility)
    mut_data = data['mutation']
    schedule_config = None
    if 'schedule' in mut_data:
        schedule_config = MutationScheduleConfig(**mut_data['schedule'])
    else:
        # Default disabled schedule if missing from YAML
        schedule_config = MutationScheduleConfig(False, 0.0, 0.0, 0)
    
    mutation_config = MutationConfig(
        weight_rate=mut_data['weight_rate'],
        structure_rate=mut_data['structure_rate'],
        node_type_rate=mut_data.get('node_type_rate', 0.05),
        schedule=schedule_config
    )
    
    # Physics
    p_data = data['physics']
    p_alpha = p_data.get('momentum_alpha', 0.1)
    physics_config = PhysicsConfig(
        max_speed=p_data['max_speed'],
        max_turn_rate=p_data['max_turn_rate'],
        alpha_speed=p_data.get('alpha_speed', p_alpha),
        alpha_turn=p_data.get('alpha_turn', p_alpha),
        momentum_alpha=p_alpha
    )
    
    return Config(
        name=data['experiment']['name'],
        seed=data['experiment']['seed'],
        population=PopulationConfig(**data['population']),
        simulation=SimulationConfig(**data['simulation']),
        network=NetworkConfig(**data['network']),
        mutation=mutation_config,
        physics=physics_config,
        fitness=fitness_config,
        output_dir=data['output']['dir']
    )


def apply_ablations(config: Config, ablate_metrics: list) -> Config:
    """
    Disable specified metrics for ablation study.
    
    Args:
        config: Base configuration
        ablate_metrics: List of metric names to disable
        
    Returns:
        Modified config with metrics disabled
    """
    for metric in ablate_metrics:
        metric_lower = metric.lower()
        if hasattr(config.fitness, metric_lower):
            getattr(config.fitness, metric_lower).enabled = False
            print(f"Ablation: Disabled {metric_lower}")
    return config
