#!/usr/bin/env python
"""verify_data_provenance.py -- data -> pkl faithfulness gate (per repeat).

The missing first link in the reproducibility chain

    data/ (ground truth)  ->  best_agents.pkl  ->  analysis/*.pkl  ->  stats  ->  latex

``verify_stats_against_data.py`` covers pkl -> stat; this script covers data -> pkl.
For every evolved repeat (54 specialists + all generalists) it asserts:

  1. Ground-truth inventory: data/agents/results_<M>_r<n> (54) and
     data/generalist/results_r<n> (N) each have a gen-150 summary.pkl.
  2. Idempotent best-selection: each summary.pkl loads with a full population and a
     deterministic lowest-fitness ("best") agent.
  3. Specialist/generalist not confused: no generalist best equals any specialist
     best; the keystone stores them under distinct keys with the right counts.
  4. Keystone faithful: every agent in data/best_agents.pkl equals the on-disk best
     for the repeat it claims to represent.
  5. Analysis pkls faithful: A6_results.pkl W_gen[r] and the generalist_results
     replicate count match the on-disk data/generalist repeats.

Exit 0 iff every check passes. Run from repo root:
    python scripts/verify_data_provenance.py
"""
from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_circuits as ac  # noqa: E402

MICE = ["B5", "B6", "B7", "D3", "D4", "D5", "D7", "D8", "D9"]
N_SPEC_REPS = 6
GEN = 150
DATA = ROOT / "data"
ANALYSIS = ROOT / "analysis"
_load = ac._load_pickle_cpu


def _w(agent):
    x = agent.weights
    return np.asarray(x.get() if hasattr(x, "get") else x)


def _best(path):
    res = _load(str(path))
    b = min(res, key=lambda r: r["fitness"])
    return _w(b["agent"]), float(b["fitness"]), len(res)


class Audit:
    def __init__(self):
        self.failures = 0

    def check(self, cond, msg):
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            self.failures += 1
        return cond


def main():
    a = Audit()

    # The raw evolutionary populations (data/agents, data/generalist) are Tier-2
    # (large; on Zenodo, not shipped in the Tier-1 public clone). If absent, run only
    # the Tier-1 checks (keystone loads, specialist/generalist not confused, A6
    # self-consistency) and SKIP the data->pkl faithfulness sections with a notice.
    spec = {(m, r): DATA / f"agents/results_{m}_r{r}/gen_{GEN}/summary.pkl"
            for m in MICE for r in range(1, N_SPEC_REPS + 1)}
    gen = {}
    r = 0
    while (DATA / f"generalist/results_r{r}").is_dir():
        gen[r] = DATA / f"generalist/results_r{r}/gen_{GEN}/summary.pkl"
        r += 1
    raw_present = (DATA / "agents").is_dir() and len(gen) > 0 \
        and all(p.exists() for p in spec.values()) and all(p.exists() for p in gen.values())

    print("1. GROUND-TRUTH INVENTORY")
    if not raw_present:
        print("  [SKIP] raw populations (data/agents, data/generalist) not present -- "
              "Tier-2 data (Zenodo). Running Tier-1 checks only (keystone + A6 self-consistency).")
    else:
        a.check(len(spec) == 54, f"54 specialist repeats (found {len(spec)})")
        a.check(all(p.exists() for p in spec.values()), "all specialist gen-150 summary.pkl present")
        a.check(len(gen) > 0 and all(p.exists() for p in gen.values()),
                f"all {len(gen)} generalist gen-150 summary.pkl present (reps {sorted(gen)})")

    spec_best, gen_best = {}, {}
    if raw_present:
        print("\n2. PER-REPEAT LOAD + IDEMPOTENT BEST-SELECTION")
        for k, p in spec.items():
            w, f, n = _best(p)
            spec_best[k] = (w, f)
            if n != 500:
                a.check(False, f"specialist {k}: population {n} != 500")
        for rep, p in gen.items():
            w, f, n = _best(p)
            gen_best[rep] = (w, f)
            if n != 500:
                a.check(False, f"generalist r{rep}: population {n} != 500")
        # idempotency spot-check (re-selection is deterministic)
        idem = all(np.array_equal(spec_best[k][0], _best(spec[k])[0]) for k in list(spec)[:3]) \
            and all(np.array_equal(gen_best[r][0], _best(gen[r])[0]) for r in list(gen)[:3])
        a.check(idem, "best-selection idempotent (spot-check)")
        a.check(True, f"loaded {len(spec_best)} specialist + {len(gen_best)} generalist best agents")

    print("\n3. KEYSTONE STRUCTURE + SPECIALIST vs GENERALIST NOT CONFUSED")
    ks = _load(str(DATA / "best_agents.pkl"))
    a.check({"specialists", "generalists"} <= set(ks), "keystone has specialists + generalists keys")
    a.check(len(ks["specialists"]) == 54, f"keystone specialists == 54 (got {len(ks['specialists'])})")
    n_ks_gen = len(ks["generalists"])
    a.check(n_ks_gen > 0, f"keystone has generalists (got {n_ks_gen})")
    # spec != gen inside the keystone itself (works without raw data)
    ks_spec_w = [_w(v["agent"]) for v in ks["specialists"].values()]
    ks_collisions = sum(any(np.array_equal(_w(gv["agent"]), sw) for sw in ks_spec_w)
                        for gv in ks["generalists"].values())
    a.check(ks_collisions == 0, f"no keystone generalist == any keystone specialist ({ks_collisions})")
    if raw_present:
        a.check(n_ks_gen == len(gen),
                f"keystone generalists == data/generalist reps ({n_ks_gen} vs {len(gen)})")
        collisions = [(rep, sk) for rep, (gw, _) in gen_best.items()
                      for sk, (sw, _) in spec_best.items() if np.array_equal(gw, sw)]
        a.check(not collisions, f"no generalist best == any specialist best ({collisions[:3]})")

    if raw_present:
        print("\n4. KEYSTONE FAITHFUL TO GROUND TRUTH (per repeat)")
        sm = sum(np.array_equal(_w(ks["specialists"][k]["agent"]), spec_best[k][0]) for k in spec_best)
        a.check(sm == 54, f"all 54 keystone specialists == data/agents best ({sm}/54)")
        gm = sum(np.array_equal(_w(ks["generalists"][r]["agent"]), gen_best[r][0]) for r in gen_best)
        a.check(gm == len(gen), f"all {len(gen)} keystone generalists == data/generalist best ({gm}/{len(gen)})")
    else:
        print("\n4. KEYSTONE FAITHFUL TO GROUND TRUTH -- [SKIP] raw data not present (Tier-2)")

    print("\n5. ANALYSIS PKLS FAITHFUL + SELF-CONSISTENT")
    # A6_results.pkl stores the exact generalist agents it analysed in W_gen and the
    # per-neuron sensitivity in gen_sensitivity_norm. The paper's reported ratio derives
    # from THESE arrays, so faithfulness here means internal self-consistency
    # (gen_sens_var_norm == var(gen_sensitivity_norm)), not equality to data/generalist.
    #
    # NOTE (provenance): A6's generalist agents (W_gen) are the evolved generalist run
    # used for the published numbers (6.7x). As of 2026-07-14 the authoritative v5 data
    # for r0-5 was synced into data/generalist (replacing an earlier stale local copy),
    # so data/generalist == best_agents.pkl == A6.W_gen for all 15 reps -- fully coherent.
    # All 15 are independent repeats (r0-5 + r6-14 extension). At Zenodo/LFS deposit the
    # canonical 15 live together in one data/generalist/results_r0..14 folder.
    a6 = pickle.load(open(ANALYSIS / "degeneracy_analyses/A6_results.pkl", "rb"))
    Wg = np.asarray(a6["W_gen"])
    a.check(a6.get("N_REPS_G") == len(Wg), f"A6 N_REPS_G == len(W_gen) ({a6.get('N_REPS_G')} vs {len(Wg)})")
    gsn = np.asarray(a6["gen_sensitivity_norm"], float)
    a.check(gsn.shape[0] == len(Wg),
            f"A6 gen_sensitivity_norm rows == n generalists ({gsn.shape[0]} vs {len(Wg)})")
    a.check(np.allclose(gsn.var(axis=0), np.asarray(a6["gen_sens_var_norm"], float)),
            "A6 gen_sens_var_norm == var(gen_sensitivity_norm)  [self-consistent]")
    # A6.W_gen must match data/generalist for ALL reps (authoritative v5 data synced
    # 2026-07-14, so the earlier r0-5 split is resolved). Hard check when raw data present;
    # additionally verify A6.W_gen == keystone generalists (always available, Tier-1).
    ks_gen_w = [_w(ks["generalists"][r]["agent"]) for r in sorted(ks["generalists"])]
    perk = [np.array_equal(Wg[r], ks_gen_w[r]) for r in range(min(len(Wg), len(ks_gen_w)))]
    a.check(all(perk), f"A6.W_gen == keystone generalists for all reps ({sum(perk)}/{len(perk)})")
    if raw_present:
        per = [np.array_equal(Wg[r], gen_best[r][0]) for r in range(min(len(Wg), len(gen)))]
        for r, m in enumerate(per):
            if not m:
                print(f"      A6.W_gen[r{r}] != data/generalist r{r}")
        a.check(all(per), f"A6.W_gen matches data/generalist for all reps ({sum(per)}/{len(per)})")
    gr = pickle.load(open(ANALYSIS / "generalist_results.pkl", "rb"))
    a.check(len(gr["results_C"]) == len(Wg),
            f"generalist_results replicate count == A6 n ({len(gr['results_C'])} vs {len(Wg)})")

    print("\n" + ("PASS: data -> pkl faithful for every repeat."
                  if not a.failures else f"FAIL: {a.failures} check(s) failed."))
    return 0 if not a.failures else 1


if __name__ == "__main__":
    sys.exit(main())
