"""
Update A3_results.pkl with mouse-identity MI/NMI and save feature vectors.

The original nb15 A3 analysis computes MI(structural cluster, fitness_quintile).
This script adds MI(structural cluster, mouse_identity) using the same k-means
cluster labels, computes AMI robustness across k=3-9 for the mouse-identity
target, and saves feature vectors to A3_results.pkl for the PCA supplementary
figure (supp_s_mi_clustering.py).

Run from the repository root:
    python scripts/update_A3_mouse_identity.py

Prerequisites:
    - nb15 has been run (A3_results.pkl exists with cluster_labels per axis)
    - data/agents/results_{mouse}_r{rep}/gen_150/summary.pkl exist

Outputs:
    - analysis/degeneracy_analyses/A3_results.pkl  (updated in-place)

Next step:
    python scripts/build_paper_stats.py
"""

import os, sys, pickle
import numpy as np
from sklearn.metrics import (mutual_info_score, normalized_mutual_info_score,
                              adjusted_mutual_info_score)
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

DEGENERACY = os.path.join(_PROJECT, "analysis", "degeneracy_analyses")
AGENTS_DIR = os.path.join(_PROJECT, "data", "agents")
MICE       = ['B5', 'B6', 'B7', 'D3', 'D4', 'D5', 'D7', 'D8', 'D9']
N_REPS     = 6


class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith("cupy"):
            mod = mod.replace("cupy._core.core", "numpy").replace("cupy", "numpy")
        return super().find_class(mod, name)


def _load_pkl(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        with open(path, "rb") as f:
            return _CpuUnpickler(f).load()


def _load_weights() -> tuple[np.ndarray, np.ndarray]:
    """Return (W, mouse_labels) for all 54 best agents ordered MICE × REPS."""
    weights, labels = [], []
    for m in MICE:
        for r in range(1, N_REPS + 1):
            path = os.path.join(AGENTS_DIR, f"results_{m}_r{r}",
                                "gen_150", "summary.pkl")
            results = _load_pkl(path)
            best = min(results, key=lambda d: d["fitness"])
            weights.append(best["agent"].weights.copy())
            labels.append(m)
    return np.array(weights), np.array(labels)


def main() -> None:
    print("Loading A3_results.pkl …")
    a3_path = os.path.join(DEGENERACY, "A3_results.pkl")
    A3 = _load_pkl(a3_path)

    print("Loading agent weights …")
    W, mouse_labels = _load_weights()
    N = len(W)
    print(f"  {N} agents  |  mice: {np.unique(mouse_labels)}")

    topo_vecs = (W != 0).astype(float).reshape(N, -1)   # binary adjacency (N, 196)
    mag_vecs  = np.abs(W).reshape(N, -1)                 # {0, 0.25, 1.0}   (N, 196)
    sign_vecs = np.sign(W).reshape(N, -1)                # E/I sign          (N, 196)

    le = LabelEncoder()
    mouse_id_int = le.fit_transform(mouse_labels)        # 0–8 integer labels
    print(f"  Mouse classes: {list(le.classes_)}\n")

    K_RANGE = range(3, 10)

    print("MI(structural cluster, mouse identity):")
    print(f"  {'Axis':12s}  {'NMI':>6}  {'AMI':>6}  AMI range k=3-9")
    print("  " + "-" * 56)

    for axis_name, vecs in [("Topology",   topo_vecs),
                             ("Magnitude",  mag_vecs),
                             ("Sign (E/I)", sign_vecs)]:
        labels_km = np.array(A3[axis_name]["cluster_labels"])

        mi_m  = mutual_info_score(labels_km, mouse_id_int)
        nmi_m = normalized_mutual_info_score(labels_km, mouse_id_int)
        ami_m = adjusted_mutual_info_score(labels_km, mouse_id_int)

        ami_vals = [
            round(adjusted_mutual_info_score(
                KMeans(n_clusters=k, random_state=42, n_init=20).fit(vecs).labels_,
                mouse_id_int), 4)
            for k in K_RANGE
        ]

        A3[axis_name]["MI_mouse"]      = float(mi_m)
        A3[axis_name]["NMI_mouse"]     = float(nmi_m)
        A3[axis_name]["AMI_mouse"]     = float(ami_m)
        A3[axis_name]["ami_mouse_min"] = min(ami_vals)
        A3[axis_name]["ami_mouse_max"] = max(ami_vals)

        print(f"  {axis_name:12s}  {nmi_m:>6.4f}  {ami_m:>6.4f}  "
              f"{min(ami_vals):.2f} – {max(ami_vals):.2f}")

    # Save feature vectors + mouse labels for the PCA supplementary figure
    A3["topo_vecs"]          = topo_vecs
    A3["mag_vecs"]           = mag_vecs
    A3["sign_vecs"]          = sign_vecs
    A3["agent_mouse_labels"] = mouse_labels

    with open(a3_path, "wb") as f:
        pickle.dump(A3, f)

    print(f"\nSaved updated A3_results.pkl -> {a3_path}")
    print("Next: python scripts/build_paper_stats.py")


if __name__ == "__main__":
    main()
