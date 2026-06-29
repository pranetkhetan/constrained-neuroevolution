import numpy as np
from utils.maze import Maze

def get_transition_matrix(maze: Maze):
    """
    Returns array sta with connection matrix among nodes.
    sta[i,:] = [parent, left_child, right_child]
    """
    n_runs = len(maze.runs)
    sta = np.full((n_runs, 3), -1)
    
    for i in range(n_runs):
        if i == 0:
            sta[i, 0] = n_runs # Transition to/from Exit
        else:
            sta[i, 0] = maze.parent[i] # parent node
        
        # Check children
        c0 = maze.children[i, 0]
        c1 = maze.children[i, 1]
        
        if c0 == -1 and c1 == -1:
            pass # End node
        else:
            # Check step type to identify Left vs Right child
            # maze.st might not be fully populated for children, need to check logic
            # Simpler: Use the st matrix I implemented? 
            # maze.st[i, j] gives step type from i to j.
            # 0=InLeft, 1=InRight.
            
            # If c0 is Left child (step type 0)
            if c0 != -1 and maze.st[i, c0] == 0:
                sta[i, 1] = c0
                sta[i, 2] = c1
            elif c0 != -1 and maze.st[i, c0] == 1:
                sta[i, 1] = c1
                sta[i, 2] = c0
            else:
                # Fallback if st not fully populated (e.g. exit?)
                # Try c1
                if c1 != -1 and maze.st[i, c1] == 0:
                    sta[i, 1] = c1
                    sta[i, 2] = c0
                else:
                    # Default assignment if step types ambiguous (shouldn't happen in binary tree)
                    sta[i, 1] = c0
                    sta[i, 2] = c1
                    
    # Special exit node from node 0?
    # rosenberg code: sta[0,0]=len(m.ru) 
    # Our maze parent[0] is -1.
    # We can handle this or leave as -1.
    return sta

def tally_strings(bouts, m=5):
    """
    Produces m dictionaries tallying j-strings up to length m.
    bouts: List[List[int]] (node sequences)
    """
    se = [{} for _ in range(m)] # se[j] tallies (j+1)-strings
    
    n_runs = len(bouts[0]) if bouts and isinstance(bouts[0], list) else 127 # estimate
    # Note: We need the actual n_runs to define the virtual exit node
    # Let's pass it or assume it's the max node in bouts + 1?
    # Better: get_transition_matrix handles it. 
    # Here we just need to make sure strings are talliable.
    
    for b in bouts:
        # Prepend/Append virtual exit node (index 127 for 6 levels)
        # Using a large fixed value or identifying from maze.
        # Let's assume the user of bouts ensures they are complete with exits if needed.
        # Actually, let's just use the bouts as provided, but ensure they aren't ignored.
        b_arr = np.array(b)
        for i in range(len(b) - m + 1):
            if i < 0: continue
            
            for j in range(m): # j+1 = length of string
                # Check bounds
                if i + j + 1 <= len(b):
                    s = tuple(b_arr[i : i+j+1])
                    if s in se[j]:
                        se[j][s] += 1
                    else:
                        se[j][s] = 1
    return se

def second_trans_prob(bouts, maze: Maze):
    """
    Computes 2nd order transition probabilities from bouts.
    Returns:
        sta: (N, 3) connection matrix
        trb: (N, 3, 3) transition probability array
             trb[i, j, k] = P(next=sta[i,k] | curr=i, prev=sta[i,j])
    """
    sta = get_transition_matrix(maze)
    ta = tally_strings(bouts, 3) # Need strings up to length 3
    
    n_nodes = len(maze.runs)
    trb = np.zeros((n_nodes, 3, 3))
    
    for i in range(n_nodes): # i is current state
        for j, sj in enumerate(sta[i]): # sta[i,j] is preceding state
            for k, sk in enumerate(sta[i]): # sta[i,k] is next state
                
                # We need P(sk | i, sj) = Count(sj->i->sk) / Count(sj->i)
                # s3 = (sj, i, sk)
                # s2 = (sj, i)
                
                # Note: sj or sk might be -1 (illegal/exit)
                if sj == -1 or sk == -1: 
                    continue
                    
                s3 = (sj, i, sk)
                s2 = (sj, i)
                
                if s2 in ta[1]: # ta[1] has 2-strings
                    count_2 = ta[1][s2]
                    count_3 = ta[2].get(s3, 0)
                    
                    if count_2 > 0:
                        trb[i, j, k] = count_3 / count_2
    return sta, trb

def compute_bias_for_node(node_idx, maze, trb, alt=True):
    """
    Computes 6 biases for node i based on transition probs trb.
    Returns: (3, 2) array [[Bf, Bl], [Lf, Lo], [Rf, Ro]]
    """
    def normalize(x):
        s = sum(x)
        return x[0]/s if s > 0 else 0.0

    tr = trb[node_idx] # (3, 3) -> [prev, next]
    
    # prev indices: 0=Parent(B), 1=Left(L), 2=Right(R)
    # next indices: 0=Parent, 1=Left, 2=Right
    
    # Bf: From B, go fwd (L or R) vs back (B/Parent) -> No, vs back is tr[0,0]
    # "Bf = forward bias from B". P(fwd | B) / (P(fwd|B) + P(back|B))?
    # Rosenberg code: Norm([tr[0,1]+tr[0,2], tr[0,0]]) -> P(L)+P(R) vs P(P)
    # result is Fraction Forward.
    Bf = normalize([tr[0,1] + tr[0,2], tr[0,0]])
    
    # Bl: Left bias when stepping forward from B.
    # P(L) vs P(R) given came from B and went forward.
    # Rosenberg: Norm([tr[0,2], tr[0,1]]) if alt and parent->i is Left?
    # Simpler default: Norm([tr[0,1], tr[0,2]]) -> L over R.
    # Note: tr[0,1] is to 'Left Child' (index 1 of sta), tr[0,2] is 'Right Child'
    
    # Check "alt" logic for alternating turns
    # If Step(parent->i) was Left, then "Right" turn is Same Direction (Circle), "Left" is Alternating?
    # Wait, Rosenberg "alt=True" means "score this as alternating vs same".
    # If entered from Left (Run is Left Child of Parent), and go Right, that is "Out" or "Cross"?
    
    # Let's stick to simple Left/Right definitions first to avoid confusion.
    # tr[0,1] is transition to Left Child.
    
    # Bl: Alternating Turn Bias when stepping forward from B.
    # If alt=True, Bl should represent "Alternation" (turning opposite to previous turn).
    # If we entered from a Left branch, "Alternating" means going to the Right child.
    if alt and maze.st[maze.parent[node_idx], node_idx] == 0: # Entered via Left child
         # Swap: P(Right) occupies the first slot (the "success" slot for alternating)
         Bl = normalize([tr[0,2], tr[0,1]])
    else:
         # Either not alt, or entered from Right child (where Left is the alternating turn)
         Bl = normalize([tr[0,1], tr[0,2]])
         
    # Lf: Forward bias from Left Child (entering from Left)
    # Came from L (index 1). Go Forward (index 0=Parent? No, Parent is 'out' if we came from L?)
    # Wait, 'L' in "B, L, R" means "Preceding State was Left Child".
    # So we are at Node i. Previous was sta[i,1] (Left Child). We moved UP to i.
    # Forward from L means continuing to Parent (index 0) or Right Child (index 2)?
    # Usually "Forward" means "Into Maze". But we came from a child (Outward).
    # Rosenberg logic: Lf = Norm([tr[1,0]+tr[1,2], tr[1,1]])
    # tr[1,1] is "Return to Left Child" (Reverse).
    # so Forward is going to Parent (0) or Right Child (2).
    Lf = normalize([tr[1,0]+tr[1,2], tr[1,1]])
    
    # Lo: Outward bias if forward from L.
    # If we went forward (0 or 2), did we go Out (Parent 0) or In (Right 2)?
    # Rosenberg: Lo = Norm([tr[1,0], tr[1,2]]) -> 0 (Parent) vs 2 (Right)
    Lo = normalize([tr[1,0], tr[1,2]])
    
    # Rf: Forward bias from Right Child
    # Came from R (index 2).
    Rf = normalize([tr[2,0]+tr[2,1], tr[2,2]])
    
    # Ro: Outward bias if forward from R.
    Ro = normalize([tr[2,0], tr[2,1]])
    
    return np.array([[Bf, Bl], [Lf, Lo], [Rf, Ro]])

def compute_bias_profile(bouts, maze):
    """
    Computes the Level-Averaged Bias Vector (Size: Levels x 3 x 2).
    """
    sta, trb = second_trans_prob(bouts, maze)
    
    # 2. Compute Biases for every node
    # Only internal nodes (level 0 to le-1) have full 3-way connections?
    # Leaf nodes don't have children.
    # We iterate from 1 to 2^le - 1 (excluding leaves? or including?)
    # Rosenberg range: range(1, 2**ma.le - 1)
    
    biases = []
    max_node = 2**maze.levels - 1
    
    # Safe range: checking bounds
    valid_range_end = min(n_runs := len(maze.runs), max_node)
    
    for i in range(0, valid_range_end):
        if i >= len(trb): break
        biases.append(compute_bias_for_node(i, maze, trb, alt=True))
        
    bi_array = np.array(biases)
    if len(bi_array) == 0:
        raise ValueError(
            "No valid node biases could be computed. Trajectory data may be insufficient or corrupted."
        )

    profile = []
    for l in range(maze.levels):
        start = 2**l - 1
        end = 2**(l+1) - 1
        
        # Adjust for 0-indexing of biases array (starts at node 0)
        s_idx = max(0, start)
        e_idx = min(len(bi_array), end)
        
        if s_idx < e_idx:
            level_biases = bi_array[s_idx:e_idx]
            avg_bias = np.mean(level_biases, axis=0)
        else:
            avg_bias = np.zeros((3, 2))
            
        profile.append(avg_bias)
        
    return np.array(profile)

def calculate_markov_distance(agent_bouts, mouse_profile, maze):
    """
    Computes Euclidean distance between Agent Profile and Mouse Profile.
    """
    agent_profile = compute_bias_profile(agent_bouts, maze)
    
    sz = min(len(agent_profile), len(mouse_profile))
    if sz == 0: return 999.0
    
    diff = agent_profile[:sz] - mouse_profile[:sz]
    return np.linalg.norm(diff)
