"""
Neural network agent for maze navigation.

Architecture:
- 6 Sensory neurons: F, L, R wall distances + Prev Speed, Prev Turn, Noise
- 6 Interneurons: Recurrently connected hidden layer  
- 2 Motor neurons: Speed and Turn rate outputs

Weights are quantized to 4 discrete values for interpretability.
Connectivity follows degree constraints (max 3 in, max 3 out).
"""
import numpy as np
import networkx as nx
from utils.backend import xp, to_cpu

# Quantized weight magnitudes for interpretability (Dale's Law will determine sign)
WEIGHT_MAGNITUDES = [0.25, 1.0]


class Agent:
    """
    Recurrent neural network agent with constrained connectivity.
    
    Args:
        config: NetworkConfig with n_interneurons, max_incoming, max_outgoing
        batch_size: Optional batch size for parallel simulation
    """
    
    def __init__(self, config, batch_size=None):
        self.n_sensory = 6
        self.n_motor = 2
        self.n_inter = config.n_interneurons
        self.n_total = self.n_sensory + self.n_motor + self.n_inter
        
        # Node indices
        self.idx_sensory = np.arange(self.n_sensory)
        self.idx_inter = np.arange(self.n_sensory, self.n_sensory + self.n_inter)
        self.idx_motor = np.arange(self.n_sensory + self.n_inter, self.n_total)
        
        self.max_incoming = config.max_incoming
        self.max_outgoing = config.max_outgoing
        # Magnitude set, signs are controlled by node_types
        self.weight_magnitudes = [0.25, 1.0]
        
        # Node types: 1 = Excitatory, -1 = Inhibitory
        # Sensory: always Excitatory
        # Inter/Motor: Randomly assigned
        self.node_types = np.ones(self.n_total, dtype=np.float64)
        self.node_types[self.idx_inter] = np.random.choice([1, -1], size=self.n_inter)
        self.node_types[self.idx_motor] = np.random.choice([1, -1], size=self.n_motor)
        
        self.batch_size = batch_size
        self.weights = np.zeros((self.n_total, self.n_total), dtype=np.float64)
        
        # State
        if batch_size:
            self.state = np.zeros((batch_size, self.n_total), dtype=np.float64)
        else:
            self.state = np.zeros(self.n_total, dtype=np.float64)
        
        self._randomize_network()
        
        # Move to device
        self.weights = xp.array(self.weights)
        self.state = xp.array(self.state)
    
    def _randomize_network(self, sparsity=0.5):
        """Initialize random sparse network respecting constraints."""
        # Random mask and magnitudes
        mask = np.random.rand(self.n_total, self.n_total) < sparsity
        rand_mags = np.random.choice(self.weight_magnitudes, size=(self.n_total, self.n_total))
        
        # Apply Dale's Law: Weight sign is determined by source node type
        # weights[i, j] = magnitude * source_type
        self.weights = rand_mags * mask * self.node_types[:, np.newaxis]
        
        # Ensure no direct connections to sensory neurons (sensors are inputs only)
        self.weights[:, self.idx_sensory] = 0
        
        # Enforce degree limits
        for j in range(self.n_total):
            incoming = np.nonzero(self.weights[:, j])[0]
            if len(incoming) > self.max_incoming:
                keep = np.random.choice(incoming, self.max_incoming, replace=False)
                new_col = np.zeros(self.n_total, dtype=np.float64)
                new_col[keep] = self.weights[keep, j]
                self.weights[:, j] = new_col
        
        for i in range(self.n_total):
            outgoing = np.nonzero(self.weights[i, :])[0]
            if len(outgoing) > self.max_outgoing:
                keep = np.random.choice(outgoing, self.max_outgoing, replace=False)
                new_row = np.zeros(self.n_total, dtype=np.float64)
                new_row[keep] = self.weights[i, keep]
                self.weights[i, :] = new_row
        
        # Ensure connectivity
        self._repair_connectivity()
    
    def _repair_connectivity(self):
        """Ensure all nodes have required connections."""
        valid_targets = np.concatenate((self.idx_inter, self.idx_motor))
        # Recurrence: Sensory, Inter, AND Motor can be sources
        valid_sources = np.concatenate((self.idx_sensory, self.idx_inter, self.idx_motor))
        
        # Sensory must have output
        for s in self.idx_sensory:
            if np.count_nonzero(self.weights[s, :]) == 0:
                v = np.random.choice(valid_targets)
                # Dale's law: sign is fixed by source
                self.weights[s, v] = np.random.choice(self.weight_magnitudes) * self.node_types[s]
        
        # Motor must have input
        for m in self.idx_motor:
            if np.count_nonzero(self.weights[:, m]) == 0:
                u = np.random.choice(valid_sources)
                if u == m: # Avoid simple self-loop during repair
                    u = np.random.choice(np.delete(valid_sources, np.where(valid_sources == m)))
                self.weights[u, m] = np.random.choice(self.weight_magnitudes) * self.node_types[u]
        
        # Interneurons must have both
        for i in self.idx_inter:
            if np.count_nonzero(self.weights[:, i]) == 0:
                u = np.random.choice(valid_sources)
                self.weights[u, i] = np.random.choice(self.weight_magnitudes) * self.node_types[u]
            if np.count_nonzero(self.weights[i, :]) == 0:
                v = np.random.choice(valid_targets)
                self.weights[i, v] = np.random.choice(self.weight_magnitudes) * self.node_types[i]
    
    def reset(self):
        """Reset network state to zeros."""
        if self.batch_size:
            self.state = xp.zeros((self.batch_size, self.n_total), dtype=xp.float64)
        else:
            self.state = xp.zeros(self.n_total, dtype=xp.float64)
    
    def forward(self, sensory_inputs):
        """
        Run one timestep of the network.
        
        Args:
            sensory_inputs: (6,) or (batch, 6) array of sensor values
            
        Returns:
            Motor outputs (speed, turn) as (2,) or (batch, 2) array
        """
        if self.batch_size:
            self.state[:, self.idx_sensory] = sensory_inputs
            
            if self.weights.ndim == 3:
                total_input = xp.matmul(self.state[:, None, :], self.weights).squeeze(1)
            else:
                total_input = self.state @ self.weights
            
            new_state = xp.tanh(total_input)
            self.state[:, self.idx_inter] = new_state[:, self.idx_inter]
            self.state[:, self.idx_motor] = new_state[:, self.idx_motor]
            
            return self.state[:, self.idx_motor]
        else:
            self.state[self.idx_sensory] = sensory_inputs
            total_input = self.state @ self.weights
            new_state = xp.tanh(total_input)
            
            self.state[self.idx_inter] = new_state[self.idx_inter]
            self.state[self.idx_motor] = new_state[self.idx_motor]
            
            return self.state[self.idx_motor]
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for visualization."""
        G = nx.DiGraph()
        
        for i in self.idx_sensory:
            G.add_node(f"S{i}", type='sensory', idx=i, node_type=self.node_types[i])
        for i in self.idx_inter:
            G.add_node(f"I{i - self.n_sensory}", type='inter', idx=i, node_type=self.node_types[i])
        for i in self.idx_motor:
            G.add_node(f"M{i - self.n_sensory - self.n_inter}", type='motor', idx=i, node_type=self.node_types[i])
        
        w_cpu = to_cpu(self.weights)
        rows, cols = np.nonzero(w_cpu)
        
        def node_name(idx):
            if idx in self.idx_sensory:
                return f"S{idx}"
            elif idx in self.idx_inter:
                return f"I{idx - self.n_sensory}"
            else:
                return f"M{idx - self.n_sensory - self.n_inter}"
        
        for i, j in zip(rows, cols):
            G.add_edge(node_name(i), node_name(j), weight=float(w_cpu[i, j]))
        
        return G
    
    def copy(self):
        """Create a copy of this agent."""
        new_agent = Agent.__new__(Agent)
        new_agent.n_sensory = self.n_sensory
        new_agent.n_motor = self.n_motor
        new_agent.n_inter = self.n_inter
        new_agent.n_total = self.n_total
        new_agent.idx_sensory = self.idx_sensory.copy()
        new_agent.idx_inter = self.idx_inter.copy()
        new_agent.idx_motor = self.idx_motor.copy()
        new_agent.max_incoming = self.max_incoming
        new_agent.max_outgoing = self.max_outgoing
        new_agent.weight_magnitudes = self.weight_magnitudes.copy()
        new_agent.node_types = self.node_types.copy()
        new_agent.batch_size = self.batch_size
        new_agent.weights = to_cpu(self.weights).copy()
        new_agent.state = to_cpu(self.state).copy()
        return new_agent
