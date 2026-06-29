"""
Compute circuit comparison statistics for the §2.8 structural degeneracy case study.

Identifies the most topologically similar and most dissimilar within-mouse circuit pairs
(by Jaccard distance on the binarised connectivity graph), loads the corresponding agent
weight matrices, and computes per-layer edge counts and shared-edge statistics.

Output: analysis/activity_embeddings/circuits_results.pkl

Run with:
    python scripts/compute_circuits_results.py
"""

import os
import pickle

import numpy as np


class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("cupy"):
            module = module.replace("cupy._core.core", "numpy").replace("cupy", "numpy")
        return super().find_class(module, name)


def _load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (ModuleNotFoundError, AttributeError):
        with open(path, "rb") as f:
            return _CpuUnpickler(f).load()

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(PROJECT_DIR, "analysis")
OUT_PATH = os.path.join(ANALYSIS_DIR, "activity_embeddings", "circuits_results.pkl")

MICE = ["B5", "B6", "B7", "D3", "D4", "D5", "D7", "D8", "D9"]
N_REPS = 6
GEN = 150

# Neuron index ranges
SENS_IDX  = slice(0, 6)
INTER_IDX = slice(6, 12)
MOTOR_IDX = slice(12, 14)


def load_best_agent(mouse: str, rep: int):
    path = os.path.join(
        PROJECT_DIR, "data", "agents",
        f"results_{mouse}_r{rep}", f"gen_{GEN}", "summary.pkl",
    )
    pop = _load(path)  # list of dicts: [{'fitness': ..., 'agent': <Agent>}, ...]
    # fitness is a loss metric — lower is better (matches analyze_circuits.py convention)
    best = min(pop, key=lambda d: d["fitness"])
    return best["agent"]


def get_edge_set(agent) -> set:
    W = np.asarray(agent.weights)
    return {(i, j) for i in range(14) for j in range(14) if W[i, j] != 0}


def layer_counts(edges: set) -> dict:
    s_idx = set(range(6))
    i_idx = set(range(6, 12))
    m_idx = set(range(12, 14))
    return {
        "SensInter":  sum(1 for (i, j) in edges if i in s_idx and j in i_idx),
        "InterInter": sum(1 for (i, j) in edges if i in i_idx and j in i_idx),
        "InterMotor": sum(1 for (i, j) in edges if i in i_idx and j in m_idx),
        "SensSens":   sum(1 for (i, j) in edges if i in s_idx and j in s_idx),
        "SensMotor":  sum(1 for (i, j) in edges if i in s_idx and j in m_idx),
        "total":      len(edges),
    }


def agent_index_to_mouse_rep(idx: int):
    mouse = MICE[idx // N_REPS]
    rep   = (idx % N_REPS) + 1
    return mouse, rep


def main():
    # ── Load A1 degeneracy results (Jaccard + behavioral distances) ──────────
    a1_path = os.path.join(ANALYSIS_DIR, "degeneracy_analyses", "A1_results.pkl")
    A1 = _load(a1_path)

    n_agents = 54
    pair_indices = [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]
    pair_types   = A1["pair_types"]       # (1431,) 'within'/'between'
    topo_dists   = A1["topo_dists"]       # (1431,) Jaccard distances
    beh_dists    = A1["beh_dists_upper"]  # (1431,) cosine behavioral distances

    within_mask = pair_types == "within"
    within_indices    = np.where(within_mask)[0]
    within_topo_dists = topo_dists[within_mask]
    within_beh_dists  = beh_dists[within_mask]

    # Most similar (min Jaccard) and most dissimilar (max Jaccard) within-mouse pairs
    sim_pos  = within_indices[np.argmin(within_topo_dists)]
    dis_pos  = within_indices[np.argmax(within_topo_dists)]

    sim_i, sim_j = pair_indices[sim_pos]
    dis_i, dis_j = pair_indices[dis_pos]

    sim_mouse_a, sim_rep_a = agent_index_to_mouse_rep(sim_i)
    sim_mouse_b, sim_rep_b = agent_index_to_mouse_rep(sim_j)
    dis_mouse_a, dis_mouse_rep_a = agent_index_to_mouse_rep(dis_i)
    dis_mouse_b, dis_mouse_rep_b = agent_index_to_mouse_rep(dis_j)

    assert sim_mouse_a == sim_mouse_b, "Similar pair must be same mouse"
    assert dis_mouse_a == dis_mouse_b, "Dissimilar pair must be same mouse"

    print(f"Most SIMILAR within-mouse pair:")
    print(f"  {sim_mouse_a} r{sim_rep_a} vs r{sim_rep_b}  "
          f"Jaccard={topo_dists[sim_pos]:.4f}  "
          f"BehDist={beh_dists[sim_pos]:.4f}")

    print(f"\nMost DISSIMILAR within-mouse pair:")
    print(f"  {dis_mouse_a} r{dis_mouse_rep_a} vs r{dis_mouse_rep_b}  "
          f"Jaccard={topo_dists[dis_pos]:.4f}  "
          f"BehDist={beh_dists[dis_pos]:.4f}")

    # ── Load agents and compute edge stats ───────────────────────────────────
    print("\nLoading agents...")
    sim_ag_a = load_best_agent(sim_mouse_a, sim_rep_a)
    sim_ag_b = load_best_agent(sim_mouse_b, sim_rep_b)
    dis_ag_a = load_best_agent(dis_mouse_a, dis_mouse_rep_a)
    dis_ag_b = load_best_agent(dis_mouse_b, dis_mouse_rep_b)

    sim_edges_a = get_edge_set(sim_ag_a)
    sim_edges_b = get_edge_set(sim_ag_b)
    dis_edges_a = get_edge_set(dis_ag_a)
    dis_edges_b = get_edge_set(dis_ag_b)

    sim_shared = sim_edges_a & sim_edges_b
    sim_union  = sim_edges_a | sim_edges_b
    dis_shared = dis_edges_a & dis_edges_b
    dis_union  = dis_edges_a | dis_edges_b

    sim_shared_frac = len(sim_shared) / len(sim_union)
    dis_shared_frac = len(dis_shared) / len(dis_union)

    print(f"\nSimilar pair edge stats:")
    print(f"  agent1 edges: {len(sim_edges_a)}  agent2 edges: {len(sim_edges_b)}")
    print(f"  shared: {len(sim_shared)}  union: {len(sim_union)}  frac: {sim_shared_frac:.3f}")

    print(f"\nDissimilar pair edge stats:")
    print(f"  agent1 edges: {len(dis_edges_a)}  agent2 edges: {len(dis_edges_b)}")
    print(f"  shared: {len(dis_shared)}  union: {len(dis_union)}  frac: {dis_shared_frac:.3f}")

    lc_dis_a = layer_counts(dis_edges_a)
    lc_dis_b = layer_counts(dis_edges_b)
    print(f"\nDissimilar r{dis_mouse_rep_a} layer counts: {lc_dis_a}")
    print(f"Dissimilar r{dis_mouse_rep_b} layer counts: {lc_dis_b}")

    # ── Build results dict ───────────────────────────────────────────────────
    results = {
        # similar pair
        "sim_mouse":          sim_mouse_a,
        "sim_rep_a":          sim_rep_a,
        "sim_rep_b":          sim_rep_b,
        "sim_jaccard":        float(topo_dists[sim_pos]),
        "sim_beh_dist":       float(beh_dists[sim_pos]),
        "sim_shared_edges":   len(sim_shared),
        "sim_union_edges":    len(sim_union),
        "sim_shared_frac":    sim_shared_frac,
        # dissimilar pair
        "dis_mouse":          dis_mouse_a,
        "dis_rep_a":          dis_mouse_rep_a,
        "dis_rep_b":          dis_mouse_rep_b,
        "dis_jaccard":        float(topo_dists[dis_pos]),
        "dis_beh_dist":       float(beh_dists[dis_pos]),
        "dis_shared_edges":   len(dis_shared),
        "dis_union_edges":    len(dis_union),
        "dis_shared_frac":    dis_shared_frac,
        # dissimilar pair layer breakdown
        "dis_rep_a_SensInter":  lc_dis_a["SensInter"],
        "dis_rep_a_InterInter": lc_dis_a["InterInter"],
        "dis_rep_a_InterMotor": lc_dis_a["InterMotor"],
        "dis_rep_b_SensInter":  lc_dis_b["SensInter"],
        "dis_rep_b_InterInter": lc_dis_b["InterInter"],
        "dis_rep_b_InterMotor": lc_dis_b["InterMotor"],
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved -> {OUT_PATH}")
    return results


if __name__ == "__main__":
    main()
