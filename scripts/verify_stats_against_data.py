#!/usr/bin/env python3
"""
verify_stats_against_data.py -- stale/derivable/compatible verification for the
load-bearing activity-embedding macros in stats/paper_stats.tex.

For each contested macro it (1) recomputes the value directly from the source pkl
under the DE-PATCHED convention (pkl stores honest distances and correct-sign rho),
(2) compares to the emitted macro, and (3) checks the manuscript's interpretation
(positive/expected for topo; null/ns for beh & sens) remains compatible with the
recomputed value. Exit 1 if any macro is stale or any interpretation is broken.

Run from repo root:  python scripts/verify_stats_against_data.py
"""
import os
import re
import sys
import pickle
import pathlib
import numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATS = ROOT / "stats" / "paper_stats.tex"
EMB = ROOT / "analysis" / "activity_embeddings"


def macro(name):
    txt = STATS.read_text()
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{([^}]*)\}", txt)
    return m.group(1) if m else None


def main():
    B = pickle.load(open(EMB / "B_results.pkl", "rb"))["B4"]
    D = pickle.load(open(EMB / "D_results.pkl", "rb"))
    fail = False

    checks = []
    wp = float(np.mean(np.asarray(B["within_proc"])))
    bp = float(np.mean(np.asarray(B["between_proc"])))
    checks.append(("statActEmbWithinProc", wp, None))
    checks.append(("statActEmbBetweenProc", bp, None))
    # de-patched: emit +raw rho (pkl rho computed on distances)
    checks.append(("statActEmbTopoMantelRho", float(D["D2"]["rho"]), "positive"))
    checks.append(("statActEmbBehMantelRho", float(D["D1"]["rho"]), "null"))
    checks.append(("statActEmbSensMantelRho", float(D["D3"]["rho"]), "null"))

    print("macro                              emitted   recomputed   verdict")
    print("-" * 70)
    for name, val, interp in checks:
        em = macro(name)
        stale = em is None or abs(float(em) - val) > 0.005
        verdict = ("STALE -> %+.3f" % val) if stale else "MATCH"
        if stale:
            fail = True
        print("%-34s %8s %12.3f   %s" % (name, em, val, verdict))
        if interp == "positive" and val <= 0:
            print("    !! interpretation 'positive/expected' INCOMPATIBLE with %+.3f" % val)
            fail = True
        if interp == "null" and abs(val) >= 0.05:
            print("    !! interpretation 'null/ns' INCOMPATIBLE with %+.3f" % val)
            fail = True

    if fail:
        print(os.linesep + "FAIL: stale macros or broken interpretations. "
              "De-patch build_paper_stats.py (remove D1-D3 sign-flip and "
              "within/between/gen Procrustes sqrt-rescale), re-run, recompile.",
              file=sys.stderr)
        sys.exit(1)
    print(os.linesep + "PASS: all checked macros match data and interpretations hold.")
    sys.exit(0)


if __name__ == "__main__":
    main()
