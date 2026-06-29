"""
Simulation and fitness computation constants.

These are implementation-level constants, not user-configurable hyperparameters.
For experiment settings, see config.yaml instead.
"""

# =============================================================================
# Raycasting (simulation.py)
# =============================================================================

# Minimum raycast distance before snapping to 0 (prevents floating point noise)
MIN_WALL_DISTANCE = 0.1

# =============================================================================
# Collision Detection (simulation.py)
# =============================================================================

# Fraction of grid resolution per sub-step for collision detection
# Lower = more accurate but slower, Higher = faster but may miss collisions
SUB_STEP_MULTIPLIER = 0.4

# Fixed number of sub-steps per frame to ensure trajectory determinism.
# Decouples individual agent precision from the fastest agent in the batch.
# Value of 30 covers max mouse speed (0.52) at SUB_STEP_MULTIPLIER resolution.
FIXED_SUB_STEPS = 30

# =============================================================================
# Fitness Metrics (fitness.py)
# =============================================================================

# Position history downsampling factor for metric computation
# Reduces memory usage and computation time while preserving trajectory shape
DOWNSAMPLE_FACTOR = 5

# Sliding window size for path straightness (tortuosity) calculation
TORTUOSITY_WINDOW = 20
