"""
Backend abstraction for GPU/CPU operations.
Provides unified interface for NumPy and CuPy.
"""
import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
    xp = cp
except ImportError:
    HAS_GPU = False
    xp = np
    cp = None


import random

def set_seed(seed: int):
    """Set global seeds for all RNG modules."""
    if seed is None:
        return
    
    random.seed(seed)
    np.random.seed(seed)
    
    if HAS_GPU:
        import cupy as cp
        cp.random.seed(seed)


def get_array_module():
    """Returns xp (cupy if available, else numpy)."""
    return xp


def to_cpu(arr):
    """Convert array to NumPy on CPU."""
    if HAS_GPU and hasattr(arr, 'get'):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_device(arr):
    """Convert array to device (GPU if available)."""
    return xp.asarray(arr)


def free_memory():
    """Release GPU memory pools."""
    if HAS_GPU:
        try:
            mempool = cp.get_default_memory_pool()
            pinned = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned.free_all_blocks()
        except Exception:
            pass
