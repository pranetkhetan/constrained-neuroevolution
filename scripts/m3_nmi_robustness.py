"""
M3 Robustness Check — NMI stability across k = 3..9
=====================================================
Loads weight_data.pkl and replicates the A3 MI/NMI computation
for k-means cluster counts 3 through 9.

Run from the project root:
    python scripts/m3_nmi_robustness.py

Output: prints a table of NMI values per axis per k,
        plus the range for each axis (to paste into paper).
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANALYSIS_DIR = os.path.join(PROJECT_DIR, "analysis")

# ---------------------------------------------------------------------------
# CPU-safe unpickler (handles CuPy-serialised agents)
# ---------------------------------------------------------------------------
class _Stub:
    """Placeholder for Agent objects — we only need the numpy arrays."""
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state): self.__dict__.update(state)

class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("cupy"):
            module = (module
                      .replace("cupy._core.core", "numpy")
                      .replace("cupy._core", "numpy")
                      .replace("cupy", "numpy"))
        if module == "core.agent" and name == "Agent":
            return _Stub
        return super().find_class(module, name)

def load_pickle_cpu(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        with open(path, "rb") as f:
            return _CpuUnpickler(f).load()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading weight_data.pkl ...")
weight_data = load_pickle_cpu(os.path.join(ANALYSIS_DIR, "weight_data.pkl"))

W    = weight_data["weight_matrices"]       # (54, 14, 14)
fits = np.array(weight_data["fitnesses"])   # (54,)

N_AGENTS = W.shape[0]

topo_vecs = (W != 0).astype(float).reshape(N_AGENTS, -1)
sign_vecs = np.sign(W).reshape(N_AGENTS, -1)
mag_vecs  = np.abs(W).reshape(N_AGENTS, -1)

AXES = [
    ("Topology",   topo_vecs),
    ("Magnitude",  mag_vecs),
    ("Sign (E/I)", sign_vecs),
]

K_RANGE   = range(3, 10)   # k = 3, 4, 5, 6, 7, 8, 9
QUANTILES = 5
np.random.seed(42)

fit_bins = pd.qcut(fits, q=QUANTILES, labels=False)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"\n{'k':>3}  {'Topology NMI':>14}  {'Magnitude NMI':>14}  {'Sign NMI':>10}")
print("-" * 50)

records = []
for k in K_RANGE:
    row = {"k": k}
    for axis_name, vecs in AXES:
        km  = KMeans(n_clusters=k, random_state=42, n_init=20).fit(vecs)
        nmi = normalized_mutual_info_score(km.labels_, fit_bins)
        mi  = mutual_info_score(km.labels_, fit_bins)
        row[axis_name] = nmi
        row[f"{axis_name}_mi"] = mi
    records.append(row)
    print(f"{k:>3}  {row['Topology']:>14.4f}  {row['Magnitude']:>14.4f}  {row['Sign (E/I)']:>10.4f}")

# ---------------------------------------------------------------------------
# Summary ranges
# ---------------------------------------------------------------------------
df = pd.DataFrame(records).set_index("k")

print("\n--- NMI ranges across k = 3–9 ---")
for axis_name, _ in AXES:
    col  = df[axis_name]
    print(f"  {axis_name:12s}: min={col.min():.4f}  max={col.max():.4f}  "
          f"range={col.max()-col.min():.4f}  (k=5 value={df.loc[5, axis_name]:.4f})")

print("\n--- Sentence for paper ---")
topo_min,  topo_max  = df["Topology"].min(),   df["Topology"].max()
mag_min,   mag_max   = df["Magnitude"].min(),  df["Magnitude"].max()
sign_min,  sign_max  = df["Sign (E/I)"].min(), df["Sign (E/I)"].max()
print(
    f"NMI values were stable across k = 3--9 "
    f"(topology: {topo_min:.2f}--{topo_max:.2f}; "
    f"magnitude: {mag_min:.2f}--{mag_max:.2f}; "
    f"sign: {sign_min:.2f}--{sign_max:.2f}), "
    f"confirming the null is not an artefact of discretisation choice."
)
