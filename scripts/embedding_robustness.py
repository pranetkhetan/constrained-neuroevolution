"""
Representational-geometry robustness across six similarity metrics (§2.7 / Supp S18).

Reproduces the multi-metric robustness check that establishes the activity-embedding
degeneracy null is not an artifact of the Procrustes/PCA choice. For each of six
similarity metrics, computes a 54x54 agent-pair distance matrix and tests whether
within-mouse pairs are closer than between-mouse pairs (two-sided Mann-Whitney U).

Six metrics (all coordinate/rotation considerations made explicit per metric):
  1. Proc-6PC   - Procrustes on 6-PC loading matrices (the main-text analysis; read
                  from B_results.pkl B4)
  2. Proc-raw   - Procrustes on raw 20x14 per-bout means (no PCA truncation)
  3. RSA        - Spearman of per-agent bout-by-bout RDMs (coordinate-free)
  4. CKA        - linear centred kernel alignment (rotation/scale invariant)
  5. Cov-Frob   - Frobenius distance between 14x14 neuron covariance matrices
  6. LogEuc-SPD - log-Euclidean (affine-invariant) distance on the SPD manifold

All six are null -> the representational degeneracy is robust to method choice.

Input  : analysis/activity_embeddings/B_results.pkl   (raw activations, mouse labels,
          and the colab_20 B4 Procrustes scale matrix)
Output : analysis/activity_embeddings/embedding_robustness.pkl
          {metric: {within_mean, between_mean, mw_p, within(np), between(np)}}

This module supersedes the previously hardcoded values in build_paper_stats.py
(statActEmb{Rsa,Cka,CovFrob,ProcRaw,LogEuc}MwPVal) and supp_emb_robustness.py.
Originally developed in colab_20b_sanity.ipynb; ported to a script for the public
reproduction pipeline.

References:
    Pezon, Schmutz & Gerstner (2026), Neuron 114:1682-1694 - neuronal embeddings.
    Kornblith et al. (2019), ICML - linear CKA.
    Arsigny et al. (2006), Magn. Reson. Med. - log-Euclidean SPD metric.

Run:
    python scripts/embedding_robustness.py
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orthogonal_procrustes, logm as matrix_log

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

ACT_EMB_DIR = PROJECT / "analysis" / "activity_embeddings"
B_PATH = ACT_EMB_DIR / "B_results.pkl"
OUT_PATH = ACT_EMB_DIR / "embedding_robustness.pkl"

N_NEURONS = 14
N_COMP = 6  # retained PCs in the main Procrustes analysis


# -- pickle loader (CuPy -> NumPy) --------------------------------------------
class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("cupy"):
            module = module.replace("cupy._core.core", "numpy").replace("cupy", "numpy")
        return super().find_class(module, name)


def _load_pickle_cpu(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        with open(path, "rb") as f:
            return _CpuUnpickler(f).load()


# -- per-metric distance computations -----------------------------------------
def _rsa_rdm_distance(per_bout_means, n_bouts):
    """Method RSA: 1 - Spearman of per-agent bout RDMs (coordinate-free)."""
    n = per_bout_means.shape[0]
    rdms = [squareform(pdist(per_bout_means[i], metric="euclidean")) for i in range(n)]
    iu = np.triu_indices(n_bouts, k=1)
    D = np.zeros((n, n))
    for i in range(n):
        ui = rdms[i][iu]
        for j in range(i + 1, n):
            r, _ = spearmanr(ui, rdms[j][iu])
            D[i, j] = D[j, i] = 1.0 - r
    return D


def _linear_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    K, L = X @ X.T, Y @ Y.T
    nK, nL = np.linalg.norm(K, "fro"), np.linalg.norm(L, "fro")
    if nK < 1e-12 or nL < 1e-12:
        return 0.0
    return float(np.sum(K * L) / (nK * nL))


def _cka_distance(per_bout_means):
    n = per_bout_means.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = 1.0 - _linear_cka(per_bout_means[i], per_bout_means[j])
    return D


def _cov_frobenius_distance(covs):
    n = len(covs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = np.linalg.norm(covs[i] - covs[j], "fro")
    return D


def _proc_raw_distance(per_bout_means):
    """Procrustes on raw centred+normalised per-bout means; dist = sqrt(2 - 2*scale)."""
    n = per_bout_means.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            X, Y = per_bout_means[i], per_bout_means[j]
            Xc = X - X.mean(axis=0)
            Yc = Y - Y.mean(axis=0)
            Xn = Xc / (np.linalg.norm(Xc, "fro") + 1e-8)
            Yn = Yc / (np.linalg.norm(Yc, "fro") + 1e-8)
            _, scale = orthogonal_procrustes(Xn, Yn)
            D[i, j] = D[j, i] = np.sqrt(max(0.0, 2.0 - 2.0 * scale))
    return D


def _log_euclidean_distance(covs, eps_frac=1e-6):
    n = len(covs)
    D = np.zeros((n, n))

    def _logm(A):
        eps = eps_frac * np.trace(A) / A.shape[0]
        return matrix_log(A + eps * np.eye(A.shape[0])).real

    logs = [_logm(c) for c in covs]
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = np.linalg.norm(logs[i] - logs[j], "fro")
    return D


def _within_between(D, labels):
    n = D.shape[0]
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if labels[i] == labels[j] else between).append(D[i, j])
    within, between = np.asarray(within), np.asarray(between)
    _, p = mannwhitneyu(within, between, alternative="two-sided")
    return within, between, float(p)


def compute_robustness(b_path=B_PATH):
    """Compute all six within/between distance distributions + MW p-values."""
    B = _load_pickle_cpu(str(b_path))
    raw_acts = B["raw_acts"]                       # list[54] of (20, 2000, 14)
    exit_frames = B["exit_frames"]                 # list[54] of (20,)
    labels = np.asarray(B["agent_mouse_labels"])   # (54,)
    n_agents = int(B["n_agents"])
    n_bouts = int(B["n_bouts"])
    D_proc_scale = np.asarray(B["B4"]["D_procrustes"])  # colab_20 scale (similarity)

    # Per-bout mean activations (n_agents, n_bouts, 14) and full concatenated traces.
    per_bout_means = np.zeros((n_agents, n_bouts, N_NEURONS))
    full_acts = []
    for i, (acts, ef) in enumerate(zip(raw_acts, exit_frames)):
        frames = []
        for b in range(n_bouts):
            t = int(ef[b]) if ef[b] > 0 else acts.shape[1]
            per_bout_means[i, b] = acts[b, :t, :].mean(axis=0)
            frames.append(acts[b, :t, :].astype(np.float64))
        full_acts.append(np.concatenate(frames, axis=0))
    covs = [np.cov(a.T) for a in full_acts]

    # Proc-6PC: convert colab_20 scale -> distance: sqrt(2*n_comp - 2*scale).
    D_proc6 = np.sqrt(np.maximum(0.0, 2.0 * N_COMP - 2.0 * D_proc_scale))

    matrices = {
        "Proc-6PC":   D_proc6,
        "Proc-raw":   _proc_raw_distance(per_bout_means),
        "RSA":        _rsa_rdm_distance(per_bout_means, n_bouts),
        "CKA":        _cka_distance(per_bout_means),
        "Cov-Frob":   _cov_frobenius_distance(covs),
        "LogEuc-SPD": _log_euclidean_distance(covs),
    }

    results = {}
    for name, D in matrices.items():
        within, between, p = _within_between(D, labels)
        results[name] = {
            "within_mean": float(within.mean()),
            "between_mean": float(between.mean()),
            "mw_p": p,
            "within": within,
            "between": between,
        }
    return results


def main():
    if not B_PATH.exists():
        sys.exit(f"ERROR: {B_PATH} not found (download analysis/ intermediates first).")
    results = compute_robustness()
    with open(OUT_PATH, "wb") as f:
        pickle.dump(results, f)

    print(f"{'Method':<14} {'within':>8} {'between':>8} {'MW p':>8}  verdict")
    print("-" * 50)
    for name, r in results.items():
        verdict = "ns" if r["mw_p"] >= 0.05 else "***"
        print(f"{name:<14} {r['within_mean']:>8.4f} {r['between_mean']:>8.4f} "
              f"{r['mw_p']:>8.4f}  {verdict}")
    print(f"\nWritten -> {OUT_PATH}")


if __name__ == "__main__":
    main()
