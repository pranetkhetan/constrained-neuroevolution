#!/usr/bin/env python3
"""
detect_monkeypatches.py -- flag reporting-layer corrections that compensate for a
data-layer state, and may silently rot when the data layer is regenerated.

Motivation (2026-06-16 review): two "Session 21 fix" patches in build_paper_stats.py
(a sign-flip 'rho = -float(...)' and a scale->distance 'sqrt(2*N - 2*scale)' rescale)
were correct when the pkl stored *scale*, but the pkl was later regenerated to store
*distance / correct-sign* values. The patches then DOUBLE-applied their corrections,
emitting wrong macros (-0.273 instead of +0.273; 2.837 instead of 1.974) while the
prose stayed correct. The committed paper silently disagreed with its own data.

This is a static, heuristic linter. It does NOT prove a patch is wrong -- it surfaces
every value-mutating correction so a human can re-verify each against current data.
A patch is DANGEROUS when (1) it transforms a value loaded from a pkl/npy, AND
(2) a comment ties it to a specific past data state ("stored scale", "Session N fix").

Exit code 1 if any HIGH-confidence patch is found (use as a pre-submission gate).
"""
import os
import re
import sys
import pathlib

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent

# Patterns that mutate a value loaded from data, between pkl/npy and macro emit.
VALUE_MUTATION = [
    (r'=\s*-\s*(?:float|int|np\.[a-z]+)?\(?\s*[A-Za-z_]\w*\s*\[', "sign-flip on loaded value"),
    (r'np\.sqrt\(.*[0-9.]+\s*\*\s*np\.asarray\(\s*[A-Za-z_]\w*\[', "scale<->distance rescale of loaded value"),
    (r'np\.sqrt\(.*-\s*[0-9.]+\s*\*\s*[A-Za-z_]\w*\[', "scale<->distance rescale of loaded value"),
    (r'=\s*-?[0-9]+\.[0-9]+\s*#.*(?:should be|actually|true value|hardcod|override|manual)', "hardcoded numeric override"),
]

# Comment markers tying code to a PAST data state (rot risk is high).
STATE_COUPLED = re.compile(
    r'#.*(session\s*\d+\s*fix|stored?\s+(scale|distance|sign)|'
    r'rep_dists?\s*=\s*scale|was computed with|corrected:|'
    r'higher\s*=\s*more similar|lower\s*=\s*more similar|obsolete|stale)',
    re.IGNORECASE)

GENERIC_PATCH = re.compile(
    r'#.*(hack|workaround|kludge|FIXME|XXX|for now|temporar|invert|sign.?fix|flip|double.?conv)',
    re.IGNORECASE)


def scan(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = []
    for i, line in enumerate(lines):
        # Generous 6-line comment window: patches often have setup lines between
        # the explanatory comment and the mutating line.
        ctx = os.linesep.join(lines[max(0, i - 6):i + 1])
        for pat, desc in VALUE_MUTATION:
            if re.search(pat, line):
                if STATE_COUPLED.search(ctx):
                    conf = "HIGH"
                elif GENERIC_PATCH.search(ctx):
                    conf = "MED"
                else:
                    conf = "LOW"
                hits.append((i + 1, conf, desc, line.strip()))
    return hits


def main():
    any_high = False
    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        hits = scan(path)
        if not hits:
            continue
        print(os.linesep + "=== " + path.name + " ===")
        for ln, conf, desc, src in hits:
            flag = {"HIGH": "[!!]", "MED": "[! ]", "LOW": "[ ?]"}[conf]
            print("  " + flag + " L" + str(ln) + " [" + conf + "] " + desc)
            print("       " + src)
            if conf == "HIGH":
                any_high = True
    if any_high:
        msg = ("HIGH-confidence value-mutating patches found. Re-verify each "
               "against CURRENT pkl/npy before trusting emitted macros.")
        print(os.linesep + msg, file=sys.stderr)
        sys.exit(1)
    print(os.linesep + "No HIGH-confidence monkeypatches detected.")
    sys.exit(0)


if __name__ == "__main__":
    main()
