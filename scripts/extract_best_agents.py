"""
Extract the best gen-150 agents into a single compact pickle (Tier-2 -> Tier-1).

The full evolutionary record in ``data/agents/`` (54 runs x 150 generations x 500
agents, ~8.7 GB) is only ever queried by the analysis/figure pipeline for the
*single best agent per run at the final generation*. This script distils exactly
those agents (54 per-mouse specialists + all generalist replicates) into one
compact pickle.

Shipping ``best_agents.pkl`` in the repository lets every figure and analysis that
calls ``load_best_agent`` run without the 8.7 GB download. The full per-generation
data is needed only to re-derive the ``analysis/*.pkl`` intermediates from scratch
(Tier 2), and is archived separately on Zenodo.

Output: ``data/best_agents.pkl``
    {
      'specialists': {(mouse, rep): {'agent': Agent, 'fitness': float}},  # 54
      'generalists': {rep: {'agent': Agent, 'fitness': float}},           # all reps (auto-detected)
      'gen': 150,
      'mice': [...],
    }

Run (with the full data/agents present):
    python scripts/extract_best_agents.py
"""

import os
import sys
import pickle
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from analyze_circuits import load_best_agent  # noqa: E402

MICE = ["B5", "B6", "B7", "D3", "D4", "D5", "D7", "D8", "D9"]
N_REPS = 6
GEN = 150

# Honour NEUROEVO_DATA_DIR so the full Zenodo archive can live outside the repo.
try:
    from core.paths import agents_dir as _agents_dir, generalist_dir as _gen_dir
    AGENTS_DIR = _agents_dir()
    GENERALIST_DIR = _gen_dir()
except Exception:
    AGENTS_DIR = PROJECT / "data" / "agents"
    GENERALIST_DIR = PROJECT / "data" / "generalist"
OUT_PATH = PROJECT / "data" / "best_agents.pkl"


def _run_dir(base, name):
    """Return the run directory for a given results_* name, or None."""
    d = base / name
    return d if d.is_dir() else None


def extract():
    specialists, generalists = {}, {}

    # 54 per-mouse specialists: data/agents/results_{MOUSE}_r{1..6}
    for mouse in MICE:
        for rep in range(1, N_REPS + 1):
            d = _run_dir(AGENTS_DIR, f"results_{mouse}_r{rep}")
            if d is None:
                print(f"  [skip] missing {AGENTS_DIR}/results_{mouse}_r{rep}")
                continue
            agent, fitness = load_best_agent(str(d), GEN)
            specialists[(mouse, rep)] = {"agent": agent, "fitness": float(fitness)}

    # Generalists: data/generalist/results_r{0..N-1}. The replicate count is
    # auto-detected (contiguous results_r{rep} from 0) rather than fixed at 6, so
    # the keystone tracks however many generalist replicates were evolved (6 at
    # first submission; 15 after the referee-follow-up extension). This keeps
    # best_agents.pkl and the analysis/*.pkl intermediates on the same n.
    if GENERALIST_DIR.is_dir():
        rep = 0
        while True:
            d = _run_dir(GENERALIST_DIR, f"results_r{rep}")
            if d is None:
                break
            agent, fitness = load_best_agent(str(d), GEN)
            generalists[rep] = {"agent": agent, "fitness": float(fitness)}
            rep += 1

    return {
        "specialists": specialists,
        "generalists": generalists,
        "gen": GEN,
        "mice": MICE,
    }


def main():
    if not AGENTS_DIR.is_dir():
        sys.exit(f"ERROR: {AGENTS_DIR} not found. Download the full agent archive "
                 f"from Zenodo (Tier-2) before running this extraction.")
    data = extract()
    with open(OUT_PATH, "wb") as f:
        pickle.dump(data, f, protocol=4)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Extracted {len(data['specialists'])} specialists + "
          f"{len(data['generalists'])} generalists")
    print(f"Written -> {OUT_PATH}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
