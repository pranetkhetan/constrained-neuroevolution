"""
Produce a repo-shippable, slimmed B_results.pkl for the activity-embedding figures.

The full ``analysis/activity_embeddings/B_results.pkl`` (~116 MB) is dominated by
``raw_acts`` — the per-frame activations of all 54 agents (54 x 20 x 2000 x 14
float32). The shipped figure/stats pipeline only needs:

  * the derived results B2/B3/B4/B5 and metadata (a few MB), used by the embedding
    RSA / dimensionality / robustness figures and by build_paper_stats; and
  * raw_acts for the four agents highlighted in the circuit-comparison driven-
    trajectory PCA (Fig 7 panel H): D9 r2/r5 and B5 r1/r5 -> indices 0, 4, 49, 52.

This script keeps both and replaces every other raw_acts entry with ``None`` (the
list length and indexing are preserved, so ``raw_acts[idx]`` still works for the
needed agents). The result is a few MB and ships in the repository, making the full
figure pipeline reproducible from a clone. ``embedding_robustness.py`` must be run
against the *full* B_results once (Tier-2) to produce embedding_robustness.pkl,
which is also shipped.

Input : analysis/activity_embeddings/B_results.pkl              (full, ~116 MB)
Output: analysis/activity_embeddings/B_results_slim.pkl         (few MB)

Run (with the full B_results present):
    python scripts/slim_b_results.py
"""

import os
import sys
import pickle
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# Honour NEUROEVO_ANALYSIS_DIR so the full (116 MB) B_results can live outside the repo.
try:
    from core.paths import analysis_dir as _analysis_dir
    ACT_EMB_DIR = _analysis_dir() / "activity_embeddings"
except Exception:
    ACT_EMB_DIR = PROJECT / "analysis" / "activity_embeddings"
FULL = ACT_EMB_DIR / "B_results.pkl"
SLIM = PROJECT / "analysis" / "activity_embeddings" / "B_results_slim.pkl"

MICE_ORDER = ["B5", "B6", "B7", "D3", "D4", "D5", "D7", "D8", "D9"]


def _agent_idx(mouse, rep):
    return MICE_ORDER.index(mouse) * 6 + (rep - 1)


# Agents whose raw activations the circuit-comparison panel H needs.
KEEP_RAW = sorted({
    _agent_idx("D9", 2), _agent_idx("D9", 5),   # pair 1 (most similar)
    _agent_idx("B5", 1), _agent_idx("B5", 5),   # pair 2 (most dissimilar)
})  # -> [0, 4, 49, 52]


class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("cupy"):
            module = module.replace("cupy._core.core", "numpy").replace("cupy", "numpy")
        return super().find_class(module, name)


def _load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        with open(path, "rb") as f:
            return _CpuUnpickler(f).load()


def main():
    if not FULL.exists():
        sys.exit(f"ERROR: {FULL} not found (download the full B_results — Tier-2).")
    B = _load(str(FULL))

    raw = B.get("raw_acts")
    if raw is not None:
        n = len(raw)
        B["raw_acts"] = [raw[i] if i in KEEP_RAW else None for i in range(n)]
        B["raw_acts_kept_indices"] = KEEP_RAW  # provenance

    with open(SLIM, "wb") as f:
        pickle.dump(B, f, protocol=4)

    full_mb = FULL.stat().st_size / 1024 / 1024
    slim_mb = SLIM.stat().st_size / 1024 / 1024
    print(f"Kept raw_acts for indices {KEEP_RAW} (4 highlighted agents).")
    print(f"{FULL.name}: {full_mb:.1f} MB  ->  {SLIM.name}: {slim_mb:.1f} MB")
    print("Ship B_results_slim.pkl in the repo (rename to B_results.pkl, or point "
          "the loaders at it).")


if __name__ == "__main__":
    main()
