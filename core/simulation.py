"""
Maze simulation with physics, raycasting, and collision detection.

Provides environment for agent navigation using egocentric control
(speed + turn rate) with momentum-based physics.
"""
import numpy as np
import math
from utils.backend import xp, to_cpu, HAS_GPU
from utils.maze import create_maze, OccupancyGrid
from core.constants import MIN_WALL_DISTANCE, SUB_STEP_MULTIPLIER, FIXED_SUB_STEPS

# CUDA kernel for GPU-accelerated raycasting
DDA_KERNEL_CODE = r'''
extern "C" __global__
void dda_kernel(
    const double* positions, 
    const double* headings, 
    const bool* grid, 
    double* distances,
    int n_agents, int width, int height,
    double min_x, double min_y, double res, double max_dist
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_agents) return;

    double px = positions[i*2 + 0];
    double py = positions[i*2 + 1];
    double theta = headings[i];

    int map_x = (int)((px - min_x) / res);
    int map_y = (int)((py - min_y) / res);

    double ray_x = cos(theta);
    double ray_y = sin(theta);
    if (abs(ray_x) < 1e-10) ray_x = 1e-10;
    if (abs(ray_y) < 1e-10) ray_y = 1e-10;

    double delta_dist_x = abs(res / ray_x);
    double delta_dist_y = abs(res / ray_y);

    int step_x = (ray_x < 0) ? -1 : 1;
    int step_y = (ray_y < 0) ? -1 : 1;

    double diff_x = px - min_x;
    double diff_y = py - min_y;
    double off_x = diff_x - floor(diff_x / res) * res;
    double off_y = diff_y - floor(diff_y / res) * res;

    double side_dist_x, side_dist_y;
    if (ray_x < 0) side_dist_x = (off_x) * delta_dist_x / res;
    else           side_dist_x = (res - off_x) * delta_dist_x / res;
    if (ray_y < 0) side_dist_y = (off_y) * delta_dist_y / res;
    else           side_dist_y = (res - off_y) * delta_dist_y / res;

    if (map_x < 0 || map_x >= width || map_y < 0 || map_y >= height) {
         distances[i] = 0.0;
         return; 
    }
    if (grid[map_x * height + map_y]) { 
        distances[i] = 0.0;
        return;
    }

    int side = 0;
    int max_steps = (int)(max_dist / res * 1.5) + 10;
    bool hit = false;
    
    for (int s=0; s < max_steps; s++) {
        if (side_dist_x < side_dist_y) {
            side_dist_x += delta_dist_x;
            map_x += step_x;
            side = 0;
        } else {
            side_dist_y += delta_dist_y;
            map_y += step_y;
            side = 1;
        }
        
        if (map_x < 0 || map_x >= width || map_y < 0 || map_y >= height) {
            hit = true; break; 
        }
        if (grid[map_x * height + map_y]) {
            hit = true; break;
        }
    }
    
    if (hit) {
        double dist;
        if (side == 0) dist = side_dist_x - delta_dist_x;
        else           dist = side_dist_y - delta_dist_y;
        
        if (dist > max_dist) dist = max_dist;
        if (dist < 0.1) dist = 0.0;
        distances[i] = dist;
    } else {
        distances[i] = max_dist;
    }
}
'''


class Simulation:
    """
    Maze navigation environment with physics and sensing.
    
    Args:
        config: PhysicsConfig with max_speed, max_turn_rate, momentum_alpha
        maze_levels: Number of maze levels (default 6)
    """
    
    def __init__(self, config, maze_levels=6):
        self.config = config
        self.maze = create_maze(maze_levels)
        self.occ_grid = OccupancyGrid(self.maze.walls)
        
        # Upload grid to GPU if available
        self.grid_gpu = xp.array(self.occ_grid.grid)
        self.prox_grid_gpu = xp.array(self.occ_grid.prox_grid)
        
        # Compile CUDA kernel if available
        self._dda_kernel = None
        if HAS_GPU:
            import cupy as cp
            self._dda_kernel = cp.RawKernel(DDA_KERNEL_CODE, 'dda_kernel')
    
    def raycast(self, positions, headings, max_dist=10.0):
        """
        Compute wall distances using DDA raycasting.
        
        Args:
            positions: (N, 2) array of agent positions
            headings: (N,) array of agent headings in radians
            max_dist: Maximum raycast distance
            
        Returns:
            (N,) array of distances to nearest wall
        """
        n_agents = len(positions)
        res = self.occ_grid.resolution
        width, height = self.occ_grid.width, self.occ_grid.height
        
        # GPU path
        if self._dda_kernel is not None:
            import cupy as cp
            pos_gpu = positions.astype(cp.float64)
            head_gpu = headings.astype(cp.float64)
            dists_gpu = cp.zeros(n_agents, dtype=cp.float64)
            
            threads = 128
            blocks = (n_agents + threads - 1) // threads
            
            self._dda_kernel(
                (blocks,), (threads,),
                (pos_gpu, head_gpu, self.grid_gpu, dists_gpu,
                 np.int32(n_agents), np.int32(width), np.int32(height),
                 np.float64(self.occ_grid.min_x), np.float64(self.occ_grid.min_y),
                 np.float64(res), np.float64(max_dist))
            )
            return dists_gpu
        
        # CPU path - vectorized DDA
        return self._raycast_cpu(positions, headings, max_dist)
    
    def _raycast_cpu(self, positions, headings, max_dist):
        """CPU fallback for raycasting."""
        n_agents = len(positions)
        res = self.occ_grid.resolution
        width, height = self.occ_grid.width, self.occ_grid.height
        
        ray_dir_x = np.cos(headings)
        ray_dir_y = np.sin(headings)
        
        epsilon = 1e-10
        ray_dir_x = np.where(np.abs(ray_dir_x) < epsilon, epsilon, ray_dir_x)
        ray_dir_y = np.where(np.abs(ray_dir_y) < epsilon, epsilon, ray_dir_y)
        
        map_x = ((positions[:, 0] - self.occ_grid.min_x) / res).astype(int)
        map_y = ((positions[:, 1] - self.occ_grid.min_y) / res).astype(int)
        
        delta_dist_x = np.abs(res / ray_dir_x)
        delta_dist_y = np.abs(res / ray_dir_y)
        
        # Initialize distances
        distances = np.full(n_agents, max_dist)
        
        # Check start conditions
        safe_x = np.clip(map_x, 0, width - 1)
        safe_y = np.clip(map_y, 0, height - 1)
        start_is_wall = self.occ_grid.grid[safe_x, safe_y]
        start_oob = (map_x < 0) | (map_x >= width) | (map_y < 0) | (map_y >= height)
        
        bad_start = start_is_wall | start_oob
        distances[bad_start] = 0.0
        active = ~bad_start
        
        step_x = np.where(ray_dir_x < 0, -1, 1)
        step_y = np.where(ray_dir_y < 0, -1, 1)
        
        off_x = (positions[:, 0] - self.occ_grid.min_x) % res
        off_y = (positions[:, 1] - self.occ_grid.min_y) % res
        
        side_dist_x = np.where(ray_dir_x < 0, off_x, res - off_x) / np.abs(ray_dir_x)
        side_dist_y = np.where(ray_dir_y < 0, off_y, res - off_y) / np.abs(ray_dir_y)
        
        max_steps = int(max_dist / res * 1.5) + 10
        
        for _ in range(max_steps):
            if not np.any(active):
                break
            
            mask_x = (side_dist_x < side_dist_y) & active
            mask_y = (~mask_x) & active
            
            side_dist_x[mask_x] += delta_dist_x[mask_x]
            map_x[mask_x] += step_x[mask_x]
            
            side_dist_y[mask_y] += delta_dist_y[mask_y]
            map_y[mask_y] += step_y[mask_y]
            
            cur_map_x = map_x[active]
            cur_map_y = map_y[active]
            
            oob = (cur_map_x < 0) | (cur_map_x >= width) | (cur_map_y < 0) | (cur_map_y >= height)
            s_x = np.clip(cur_map_x, 0, width - 1)
            s_y = np.clip(cur_map_y, 0, height - 1)
            is_wall = self.occ_grid.grid[s_x, s_y]
            
            hit_now = oob | is_wall
            active_indices = np.where(active)[0]
            hit_indices = active_indices[hit_now]
            
            if len(hit_indices) > 0:
                is_x_hit = mask_x[hit_indices]
                hit_dists = np.zeros(len(hit_indices), dtype=np.float64)
                hit_dists[is_x_hit] = side_dist_x[hit_indices[is_x_hit]] - delta_dist_x[hit_indices[is_x_hit]]
                hit_dists[~is_x_hit] = side_dist_y[hit_indices[~is_x_hit]] - delta_dist_y[hit_indices[~is_x_hit]]
                
                distances[hit_indices] = hit_dists
                active[hit_indices] = False
        
        final_dists = np.minimum(distances, max_dist)
        final_dists[final_dists < MIN_WALL_DISTANCE] = 0.0
        return final_dists
    
    def step(self, positions, velocities):
        """
        Move agents with sub-stepping collision detection.
        
        Args:
            positions: (N, 2) current positions
            velocities: (N, 2) movement vectors
            
        Returns:
            new_positions: (N, 2) updated positions
            collisions: (N,) boolean array of collision flags
        """
        dists = xp.linalg.norm(velocities, axis=1)
        max_dist = xp.max(dists)
        
        if max_dist < 1e-5:
            return positions, xp.zeros(len(positions), dtype=bool)
        
        n_steps = FIXED_SUB_STEPS
        
        step_delta = velocities / n_steps
        pos = positions.copy()
        frame_collisions = xp.zeros(len(positions), dtype=bool)
        
        # Get grid parameters for in-line occupancy check
        res = self.occ_grid.resolution
        min_x, min_y = self.occ_grid.min_x, self.occ_grid.min_y
        width, height = self.occ_grid.width, self.occ_grid.height
        
        for _ in range(n_steps):
            proposed = pos + step_delta
            
            # Optimized in-line occupancy check (GPU only if xp=cupy)
            gx = ((proposed[:, 0] - min_x) / res).astype(xp.int32)
            gy = ((proposed[:, 1] - min_y) / res).astype(xp.int32)
            
            valid = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
            collisions = xp.ones(len(pos), dtype=bool)
            
            # Only update valid entries (GPU lookup)
            if xp.any(valid):
                # result[valid] = grid[gx[valid], gy[valid]]
                # Indexing must be done on the same device
                collisions[valid] = self.grid_gpu[gx[valid], gy[valid]]
            
            frame_collisions |= collisions
            pos[~collisions] = proposed[~collisions]
            step_delta[collisions] = 0.0
        
        return pos, frame_collisions
    
    def is_near_wall(self, positions):
        """Check if positions are within wall proximity."""
        return self.occ_grid.is_proximate(positions[:, 0], positions[:, 1], xp)
