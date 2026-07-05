#!/usr/bin/env python
"""Real-mouse individuation test: are the 9 mice genuinely distinguishable in the
4 fitness metrics beyond within-mouse bout-to-bout sampling noise?

This is the foundation test for the behavioral-individuation claim (Results §2.1;
Supplementary S12). It uses the real mouse data only (no agents, no simulation):
each mouse's raw ``-tf`` trajectory is partitioned into folds, the paper's 4
behavioral metrics are computed per fold with the exact ``holdout_eval`` machinery,
and within- vs between-mouse fold distances plus a leave-one-fold-out nearest-mouse
classification are reported. If between distances exceed within and classification
exceeds chance (1/9), the individuation the agents reproduce reflects stable
individual identity rather than bout-specific noise, and the held-out attenuation
(Supplementary S12) is target-estimation noise rather than circularity.

Provenance: promoted from the referee-audit script ``review/scripts/r4_mouse_individuation.py``
(computation unchanged, seed=1, deterministic). Writes a small aggregate result
(no raw trajectories) so ``build_paper_stats.py`` can emit macros without the
Tier-3 raw ``-tf`` files.

Requires the raw Rosenberg ``-tf`` files (Tier 3, not redistributed). Point at the
same directory ``holdout_eval.py`` uses::

    python scripts/analyze_individuation.py --tf_dir data/raw --n_folds 6

Output:
    analysis/individuation_results.pkl

References:
    Rosenberg, M. et al. (2021). Mice in a labyrinth show rapid learning,
    sudden insight, and efficient exploration. eLife 10:e66175.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "analysis"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

MICE = ["B5", "B6", "B7", "D3", "D4", "D5", "D7", "D8", "D9"]
SEED = 1
N_PERM = 1000


def _js_div(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / (p.sum() + 1e-12); q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def kl(x, y):
        msk = x > 0
        return float(np.sum(x[msk] * np.log(x[msk] / (y[msk] + 1e-12))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def beh_components(a, b):
    """Normalized (markov, occupancy, tortuosity, turn_bias) distances in [0, 1],
    matching the paper's fitness normalization."""
    ma = np.asarray(a["markov_profile"], float).ravel()
    mb = np.asarray(b["markov_profile"], float).ravel()
    markov = min(1.0, float(np.linalg.norm(ma - mb)) / 6.0)
    occ = min(1.0, _js_div(a["node_pdf"], b["node_pdf"]) / np.log(2))
    sa, sb = float(a["straightness"]), float(b["straightness"])
    tort = min(1.0, abs(sa - sb) / (0.5 * (abs(sa) + abs(sb)) + 1e-12))
    tb = min(1.0, abs(float(a["turn_bias"]) - float(b["turn_bias"])))
    return np.array([markov, occ, tort, tb])


def fitness_distance(a, b):
    c = beh_components(a, b)
    return float(2 * c[0] + 2 * c[1] + c[2] + c[3])   # F = 2M + 2O + T + B


def make_folds(n, k, mode="stride"):
    idx = np.arange(n)
    if mode == "stride":
        return [idx[r::k] for r in range(k)]
    return [idx[r * n // k:(r + 1) * n // k] for r in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf_dir", default="data/raw",
                    help="dir with {mouse}-tf raw trajectory files (Tier-3, not in repo)")
    ap.add_argument("--n_folds", type=int, default=6)
    ap.add_argument("--fold_mode", choices=["stride", "contiguous"], default="stride")
    args = ap.parse_args()

    from holdout_eval import load_tf_file, compute_metrics_from_bouts
    from utils.maze import create_maze

    tf_dir = Path(args.tf_dir)
    if not tf_dir.is_absolute():
        tf_dir = REPO / tf_dir
    maze = create_maze(6)

    # 1. per-fold metrics for each mouse
    fold_metrics = {}
    for mouse in MICE:
        tf_path = tf_dir / f"{mouse}-tf"
        if not tf_path.exists():
            tf_path = tf_dir / f"{mouse}_tf"
        if not tf_path.exists():
            raise FileNotFoundError(
                f"Missing -tf for {mouse} in {tf_dir}. This analysis requires the raw "
                "Rosenberg -tf files (Tier 3, not in repo).")
        tr = load_tf_file(str(tf_path))
        n = len(tr.no)
        folds = make_folds(n, args.n_folds, args.fold_mode)
        mets = []
        for f_idx in folds:
            if len(f_idx) < 3:
                continue
            no = [tr.no[i] for i in f_idx]
            ce = [tr.ce[i] for i in f_idx] if tr.ce else None
            ke = [tr.ke[i] for i in f_idx] if tr.ke else None
            mets.append(compute_metrics_from_bouts(no, ce, ke, maze))
        fold_metrics[mouse] = mets
        print(f"  {mouse}: {n} bouts -> {len(mets)} folds")

    # 2. pairwise fold distances (combined + per component)
    items = [(m, i) for m in MICE for i in range(len(fold_metrics[m]))]
    comp_names = ["markov", "occupancy", "tortuosity", "turn_bias", "combined_F"]
    wi = {c: [] for c in comp_names}
    be = {c: [] for c in comp_names}
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            (ma, ia), (mb, ib) = items[a], items[b]
            A = fold_metrics[ma][ia]; B = fold_metrics[mb][ib]
            comp = beh_components(A, B)
            F = 2 * comp[0] + 2 * comp[1] + comp[2] + comp[3]
            tgt = wi if ma == mb else be
            for ci, cn in enumerate(comp_names[:4]):
                tgt[cn].append(float(comp[ci]))
            tgt["combined_F"].append(float(F))

    results = {}
    for cn in comp_names:
        w = np.array(wi[cn]); bt = np.array(be[cn])
        U = stats.mannwhitneyu(w, bt, alternative="less")   # H1: within < between
        p = float(U.pvalue)
        rbc = 1.0 - 2.0 * U.statistic / (len(w) * len(bt))
        ratio = w.mean() / bt.mean() if bt.mean() > 1e-12 else np.nan
        results[cn] = {"within": float(w.mean()), "between": float(bt.mean()),
                       "ratio": float(ratio), "mw_p": p, "rank_biserial": float(rbc)}

    # 3. leave-one-fold-out nearest-mouse classification
    all_items = items
    D = np.zeros((len(all_items), len(all_items)))
    for a in range(len(all_items)):
        for b in range(a + 1, len(all_items)):
            d = fitness_distance(fold_metrics[all_items[a][0]][all_items[a][1]],
                                 fold_metrics[all_items[b][0]][all_items[b][1]])
            D[a, b] = D[b, a] = d
    labels_arr = np.array([m for m, _ in all_items])

    def loo_acc(labs):
        c = 0
        for a in range(len(all_items)):
            best_m, best_d = None, np.inf
            for m in MICE:
                idx = [k for k in range(len(all_items)) if labs[k] == m and k != a]
                if not idx:
                    continue
                dd = D[a, idx].mean()
                if dd < best_d:
                    best_d, best_m = dd, m
            c += int(best_m == labs[a])
        return c / len(all_items)

    acc = loo_acc(labels_arr)
    rng = np.random.default_rng(SEED)
    null = np.array([loo_acc(rng.permutation(labels_arr)) for _ in range(N_PERM)])
    acc_p = float((null >= acc).mean())
    chance = 1.0 / 9

    out = {"per_component": results,
           "classification": {"acc": float(acc), "chance": chance, "p": acc_p},
           "n_folds": args.n_folds, "fold_mode": args.fold_mode, "seed": SEED}
    ANALYSIS.mkdir(exist_ok=True)
    (ANALYSIS / "individuation_results.pkl").write_bytes(pickle.dumps(out))
    print(f"\ncombined_F within {results['combined_F']['within']:.3f} vs "
          f"between {results['combined_F']['between']:.3f} (MW p={results['combined_F']['mw_p']:.2e})")
    print(f"classification acc {acc:.3f} vs chance {chance:.3f} (p={acc_p:.4f})")
    print("Saved: analysis/individuation_results.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
