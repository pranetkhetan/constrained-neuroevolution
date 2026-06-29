"""
Binary tree maze with occupancy grid for collision detection and raycasting.
Simplified from Rosenberg Lab utilities.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List
from scipy.ndimage import binary_dilation, distance_transform_edt


@dataclass
class Maze:
    """Binary maze structure."""
    levels: int
    runs: List[List[int]]           # Cell lists per run
    parent: np.ndarray              # Parent run indices
    children: np.ndarray            # Child run indices (n, 2)
    xc: np.ndarray                  # X coordinate per cell
    yc: np.ndarray                  # Y coordinate per cell
    cell_lookup: Dict[Tuple, int]   # (x,y) -> cell number
    run_lookup: np.ndarray          # cell -> run index
    walls: np.ndarray               # Wall coordinates for rendering
    st: np.ndarray                  # Step type matrix for turn bias


def create_maze(levels: int = 6) -> Maze:
    """
    Construct a binary maze with n levels.
    
    The maze is a binary tree where each junction splits into two corridors.
    Entry is from the left, exit is back through the entry.
    """
    runs = []
    parent = []
    
    for level in range(levels + 1):
        xd = (level + 1) % 2  # Step in x direction
        yd = level % 2        # Step in y direction
        run_length = 2 ** ((levels - level) // 2)
        
        for j in range(2 ** level):
            if level == 0:
                # Entry run
                parent.append(-1)
                x, y = int(2 ** (levels / 2) - 1), int(2 ** (levels / 2) - 1)
                runs.append([(x1, y) for x1 in range(0, x + 1)])
            else:
                parent_idx = 2 ** (level - 1) - 1 + j // 2
                parent.append(parent_idx)
                x0, y0 = runs[parent_idx][-1]
                xs = xd * (2 * (j % 2) - 1)
                ys = yd * (2 * (j % 2) - 1)
                x, y = x0 + xs * run_length, y0 + ys * run_length
                
                if xs == 0:
                    runs.append([(x, y1) for y1 in range(y0 + ys, y + ys, ys)])
                else:
                    runs.append([(x1, y) for x1 in range(x0 + xs, x + xs, xs)])
    
    # Build cell lookup
    cell_lookup = {}
    cell_to_xy = {}
    cell_id = 0
    for run in runs:
        for p in run:
            cell_lookup[p] = cell_id
            cell_to_xy[cell_id] = p
            cell_id += 1
    
    n_cells = cell_id
    runs_by_cell = [[cell_lookup[p] for p in run] for run in runs]
    
    parent = np.array(parent)
    children = np.full((len(runs), 2), -1, dtype=int)
    for i, p in enumerate(parent):
        if p >= 0:
            if children[p, 0] == -1:
                children[p, 0] = i
            else:
                children[p, 1] = i
    
    xc = np.array([cell_to_xy[c][0] for c in range(n_cells)])
    yc = np.array([cell_to_xy[c][1] for c in range(n_cells)])
    
    # Determine run for each cell
    run_lookup = np.zeros(n_cells, dtype=int)
    for i, run in enumerate(runs_by_cell):
        for c in run:
            run_lookup[c] = i
    
    # Generate wall coordinates
    walls = _generate_walls(runs_by_cell, xc, yc, parent, children, cell_lookup, levels)
    
    # Generate step type matrix
    st = _make_step_type(levels, runs, parent, children)
    
    return Maze(
        levels=levels,
        runs=runs_by_cell,
        parent=parent,
        children=children,
        xc=xc,
        yc=yc,
        cell_lookup=cell_lookup,
        run_lookup=run_lookup,
        walls=walls,
        st=st
    )


def _make_step_type(levels, runs, parent, children):
    """
    Generate step type matrix:
    in left=0, in right=1, out left=2, out right=3
    """
    n_runs = len(runs)
    exit_state = n_runs # 1 + highest node number
    st = np.full((n_runs + 1, n_runs + 1), -1, dtype=int)
    
    for i in range(levels + 1):
        for j in range(2**i - 1, 2**(i + 1) - 1):
            if j > 0: # not the first node
                if (i + j + parent[j]) % 2 == 0:
                    st[j, parent[j]] = 2 # out left
                else:
                    st[j, parent[j]] = 3 # out right
            
            if i < levels: # not the last level
                # Look at children (safe access)
                c0 = children[j, 0] if j < len(children) else -1
                c1 = children[j, 1] if j < len(children) else -1
                
                # We need to access children indices if they exist
                # In numpy, children is (n, 2). -1 is no child.
                
                # Check children of run j
                row_child = children[j]
                
                for c in row_child:
                    if c != -1:
                        if (i + j + c) % 2 == 0:
                            st[j, c] = 1 # in right
                        else:
                            st[j, c] = 0 # in left
                            
    st[0, exit_state] = 3 # special case
    return st


def _generate_walls(runs, xc, yc, parent, children, cell_lookup, levels):
    """
    Generate wall polygon for maze boundary.
    
    This implements the 'MazeWall' logic from Rosenberg Lab utilities,
    tracing the outline of the corridors rather than the centerline.
    """
    def acw(i): # recursive function that returns a path for the wall starting with run i
        r = runs[i]
        c0 = np.array([xc[r[0]], yc[r[0]]]) # first cell in this run
        c1 = np.array([xc[r[-1]], yc[r[-1]]]) # last cell in this run
        
        if i == 0:
            d = np.array([1, 0]) # direction of the entry run
        else:
            # Last cell of parent run
            p_idx = parent[i]
            p_run = runs[p_idx]
            p1 = np.array([xc[p_run[-1]], yc[p_run[-1]]])
            d = c0 - p1 # direction of this run
            
        # Diagonal and compass displacements
        # sw = south-west relative to direction d? (Logic copied from rosenberg_maze.py)
        sw = 0.5 * np.array([-d[0]-d[1], d[0]-d[1]]) 
        se = 0.5 * np.array([-d[0]+d[1], -d[0]-d[1]]) 
        nw = 0.5 * np.array([d[0]-d[1], d[0]+d[1]])              
        ne = 0.5 * np.array([d[0]+d[1], -d[0]+d[1]])
        
        if i == 0:
            p = [c0 + sw]
        else:
            p = []
            
        p.append(c1 + sw) # to end of this run on left side
        
        # Check for children
        # children array is (N, 2), -1 indicates no child
        current_children = [c for c in children[i] if c != -1]
        
        if len(current_children) > 0:
            # We assume a binary tree structure where if there are children, there are usually 2
            # We need to distinguish Left vs Right child to traverse in correct order
            
            # Direction to first child start
            child_0 = current_children[0]
            r_child_0 = runs[child_0]
            c_child_0_start = np.array([xc[r_child_0[0]], yc[r_child_0[0]]])
            e = c_child_0_start - c1
            
            # Check if e is 'left' relative to d (vector [-dy, dx])
            is_left = np.allclose(e, np.array([-d[1], d[0]]))
            
            if is_left:
                il = current_children[0]
                ir = current_children[1] if len(current_children) > 1 else None
            else:
                ir = current_children[0]
                il = current_children[1] if len(current_children) > 1 else None
            
            if il is not None:
                p += acw(il) # accumulate left path
            
            p.append(c1 + ne) # short connector on far side
            
            if ir is not None:
                p += acw(ir) # accumulate right path
                
            p.append(c0 + se)  # finish the reverse path  
            
        else: # End point
            p.append(c1 + nw)
            p.append(c1 + ne)
            p.append(c1 + se)
            
        return p
    
    path = acw(0)
    return np.array(path)


class OccupancyGrid:
    """Grid-based collision detection with raycasting support."""
    
    def __init__(self, walls: np.ndarray, resolution: float = 0.05):
        self.resolution = resolution
        
        x_coords = walls[:, 0]
        y_coords = walls[:, 1]
        
        self.min_x = np.min(x_coords) - 3.0  # Extra space for exit
        self.max_x = np.max(x_coords) + 0.1
        self.min_y = np.min(y_coords) - 0.1
        self.max_y = np.max(y_coords) + 0.1
        
        self.width = int(np.ceil((self.max_x - self.min_x) / resolution))
        self.height = int(np.ceil((self.max_y - self.min_y) / resolution))
        
        self.grid = np.zeros((self.width, self.height), dtype=bool)
        self._rasterize_walls(walls)
        self._add_boundaries()
        
        # Dilate for safety margin
        self.grid = binary_dilation(self.grid, iterations=2).astype(bool)
        
        # Proximity grid for wall contact detection
        dt = distance_transform_edt(~self.grid)
        self.prox_grid = (dt * resolution) < 0.25
    
    def _rasterize_walls(self, walls):
        """Rasterize wall polygon into grid."""
        for i in range(len(walls) - 1):
            self._rasterize_line(walls[i], walls[i + 1])
    
    def _rasterize_line(self, p1, p2):
        """Rasterize a line segment."""
        x0 = int((p1[0] - self.min_x) / self.resolution)
        y0 = int((p1[1] - self.min_y) / self.resolution)
        x1 = int((p2[0] - self.min_x) / self.resolution)
        y1 = int((p2[1] - self.min_y) / self.resolution)
        
        num_points = int(np.hypot(x1 - x0, y1 - y0) * 10) + 1
        if num_points == 0:
            return
        
        xs = np.linspace(x0, x1, num_points).astype(int)
        ys = np.linspace(y0, y1, num_points).astype(int)
        
        valid = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
        self.grid[xs[valid], ys[valid]] = True
    
    def _add_boundaries(self):
        """Add outer boundary walls."""
        boundaries = [
            ([-0.5, -0.5], [-0.5, 6.0]),
            ([-0.5, 8.0], [-0.5, 14.5]),
            ([-0.5, 14.5], [14.5, 14.5]),
            ([14.5, 14.5], [14.5, -0.5]),
            ([14.5, -0.5], [-0.5, -0.5])
        ]
        for p1, p2 in boundaries:
            self._rasterize_line(np.array(p1), np.array(p2))
    
    def to_grid(self, x, y):
        """World to grid coordinates."""
        gx = ((x - self.min_x) / self.resolution).astype(int)
        gy = ((y - self.min_y) / self.resolution).astype(int)
        return gx, gy
    
    def is_occupied(self, x, y, xp):
        """Check if positions are in walls."""
        gx, gy = self.to_grid(x, y)
        valid = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)
        result = xp.ones_like(gx, dtype=bool)
        
        if xp.any(valid):
            # Convert to CPU for NumPy grid indexing
            valid_cpu = valid.get() if hasattr(valid, 'get') else valid
            gx_cpu = gx.get() if hasattr(gx, 'get') else gx
            gy_cpu = gy.get() if hasattr(gy, 'get') else gy
            
            # Index NumPy grid on CPU
            values = self.grid[gx_cpu[valid_cpu], gy_cpu[valid_cpu]]
            
            # Convert back to GPU if needed
            if hasattr(result, 'get'):
                import cupy as cp
                result[valid] = cp.asarray(values)
            else:
                result[valid] = values
        return result
    
    def is_proximate(self, x, y, xp):
        """Check if positions are near walls."""
        gx, gy = self.to_grid(x, y)
        valid = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)
        result = xp.zeros_like(gx, dtype=bool)
        
        if xp.any(valid):
            # Convert to CPU for NumPy grid indexing
            valid_cpu = valid.get() if hasattr(valid, 'get') else valid
            gx_cpu = gx.get() if hasattr(gx, 'get') else gx
            gy_cpu = gy.get() if hasattr(gy, 'get') else gy
            
            # Index NumPy grid on CPU
            values = self.prox_grid[gx_cpu[valid_cpu], gy_cpu[valid_cpu]]
            
            # Convert back to GPU if needed
            if hasattr(result, 'get'):
                import cupy as cp
                result[valid] = cp.asarray(values)
            else:
                result[valid] = values
        return result
