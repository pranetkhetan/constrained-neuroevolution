#!/usr/bin/env python
"""
Extract paper-quoted statistics from degeneracy analyses (A1-A6) and emit:
  - paper/latex/paper_stats_v2.tex   newcommand macros for paper_v2
  - paper/latex/paper_stats_v2.txt   human-readable key=value listing

All macros use the statDeg prefix to avoid clashes with the existing
stat macros in paper_stats.tex.

A3 and A4 pickles are truncated; their values are read from
analysis/degeneracy_analyses/consolidated_summary.txt (written by the
notebook at generation time). If the pickles are regenerated, restore
the direct load blocks below.

Usage:
    python scripts/extract_degeneracy_stats.py
"""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
ANALYSIS = PROJECT / "analysis" / "degeneracy_analyses"
OUT_TEX = PROJECT / "paper_v2" / "latex" / "paper_stats_v2_standalone.tex"
OUT_TXT = PROJECT / "paper_v2" / "latex" / "paper_stats_v2_standalone.txt"

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

MICE = ["B5", "B6", "B7", "D3", "D4", "D5", "D7", "D8", "D9"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        with open(path, "rb") as f:
            class _CpuUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module.startswith("cupy"):
                        module = module.replace("cupy._core.core", "numpy").replace("cupy", "numpy")
                    elif module == "core" or module.startswith("core."):
                        module = "numpy." + module
                    return super().find_class(module, name)
            return _CpuUnpickler(f).load()


def _fmt(x, prec=3):
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    f = float(x)
    if not np.isfinite(f):
        return "NA"
    return f"{f:.{prec}f}"


def _fmt_p(p):
    p = float(p)
    if not np.isfinite(p):
        return "NA"
    if p < 1e-3:
        return "<0.001"
    return f"{p:.3f}"


def _fmt_pct(x, prec=1):
    return f"{float(x):.{prec}f}"


def _parse_summary(path):
    """Parse consolidated_summary.txt into a flat key->value dict.

    Section headers are bare 'A1', 'A2', ... lines.
    Each data line is '  key: value' where key may contain spaces.
    Result keys are '{section}_{key}', e.g. 'A3_Sign (E/I)'.
    """
    result = {}
    current = None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        m = re.match(r"^(A\d+)$", stripped)
        if m:
            current = m.group(1)
            continue
        m = re.match(r"^(.+?):\s+(.+)$", stripped)
        if m and current:
            result[f"{current}_{m.group(1)}"] = m.group(2).strip()
    return result


# ---------------------------------------------------------------------------
# load data
# ---------------------------------------------------------------------------

summary = _parse_summary(ANALYSIS / "consolidated_summary.txt")

A1 = _load(ANALYSIS / "A1_results.pkl")
A2 = _load(ANALYSIS / "A2_results.pkl")
A5 = _load(ANALYSIS / "A5_results.pkl")
A6 = _load(ANALYSIS / "A6_results.pkl")

# A3 and A4: read from summary (pickles truncated)
A3_mi_topo  = float(summary["A3_Topology"].split("MI=")[1].split()[0])
A3_nmi_topo = float(summary["A3_Topology"].split("NMI=")[1])
A3_mi_mag   = float(summary["A3_Magnitude"].split("MI=")[1].split()[0])
A3_nmi_mag  = float(summary["A3_Magnitude"].split("NMI=")[1])
A3_mi_sign  = float(summary["A3_Sign (E/I)"].split("MI=")[1].split()[0])
A3_nmi_sign = float(summary["A3_Sign (E/I)"].split("NMI=")[1])

A4_mw_p           = float(summary["A4_mw_p"])
A4_mantel_rho     = float(summary["A4_mantel_rho"])
A4_mantel_p_param = float(summary["A4_mantel_p_parametric"])
A4_mantel_p_perm  = float(summary["A4_mantel_p_permutation"])


# ---------------------------------------------------------------------------
# compute derived statistics
# ---------------------------------------------------------------------------

# A1
A1_rho_all        = float(A1["rho_all"])
A1_p_all          = float(A1["p_all"])
A1_rho_within     = float(A1["rho_within"])
A1_p_within       = float(A1["p_within"])
A1_rho_between    = float(A1["rho_between"])
A1_p_between      = float(A1["p_between"])
A1_mw_beh_p       = float(A1["mw_beh_p"])
A1_n_pairs        = int(np.array(A1["topo_dists"]).shape[0])

# A2
A2_rho_topo       = float(A2["rho_topo"])
A2_p_topo         = float(A2["p_topo"])
A2_rho_mag        = float(A2["rho_mag"])
A2_p_mag          = float(A2["p_mag"])
A2_partial_rho_topo = float(A2["partial_rho_topo_given_mag"])
A2_partial_p_topo   = float(A2["partial_p_topo_given_mag"])
A2_partial_rho_mag  = float(A2["partial_rho_mag_given_topo"])
A2_partial_p_mag    = float(A2["partial_p_mag_given_topo"])
A2_n_pairs        = len(A2["df"])

# A5
A5_rho_rand       = float(A5["rho_rand"])
A5_p_rand         = float(A5["p_rand"])
A5_mw_within_p    = float(A5["mw_within_between_p"])

# A6
A6_mw_p_topo      = float(A6["mw_p_gen_spec_topo"])
A6_gen_cost_pct   = np.array(A6["gen_cost_pct"])
A6_mean_cost_pct  = float(np.mean(A6_gen_cost_pct))
A6_min_cost_pct   = float(np.min(A6_gen_cost_pct))
A6_max_cost_pct   = float(np.max(A6_gen_cost_pct))
A6_hardest_mouse  = MICE[int(np.argmax(A6_gen_cost_pct))]
A6_easiest_mouse  = MICE[int(np.argmin(A6_gen_cost_pct))]

A6_spec_sens_var  = np.array(A6["spec_sens_var"])
A6_gen_sens_var   = np.array(A6["gen_sens_var_norm"])  # normalized by full-circuit permutation baseline (apples-to-apples)
A6_spec_var_mean  = float(np.mean(A6_spec_sens_var))
A6_gen_var_mean   = float(np.mean(A6_gen_sens_var))
if A6_gen_var_mean > 1e-9:
    A6_var_ratio  = A6_spec_var_mean / A6_gen_var_mean
else:
    A6_var_ratio  = float("nan")

A6_gen_topo_sims  = np.array(A6["gen_topo_sims"])
A6_spec_topo_sims = np.array(A6["spec_topo_sims"])
A6_gen_topo_mean  = float(np.mean(A6_gen_topo_sims))
A6_spec_topo_mean = float(np.mean(A6_spec_topo_sims))

# max NMI across A3 axes
A3_nmi_max = max(A3_nmi_topo, A3_nmi_mag, A3_nmi_sign)


# ---------------------------------------------------------------------------
# emit macros
# ---------------------------------------------------------------------------

lines_tex = [
    "% paper_stats_v2.tex -- degeneracy analyses (A1-A6)",
    "% Auto-generated by scripts/extract_degeneracy_stats.py",
    "% Do NOT edit by hand -- re-run the script after any analysis update.",
    "",
]
lines_txt = [
    "paper_stats_v2.txt -- degeneracy analyses (A1-A6)",
    "Auto-generated by scripts/extract_degeneracy_stats.py",
    "",
]


def emit(name, value, comment=""):
    macro = "\\newcommand{\\%s}{%s}" % (name, value)
    if comment:
        macro += "  %% %s" % comment
    lines_tex.append(macro)
    lines_txt.append("%s = %s  # %s" % (name, value, comment))


# -- A1 -----------------------------------------------------------------------
lines_tex.append("")
lines_tex.append("% -- A1: Topology-behavior flatness --")

emit("statDegAoneRhoAll",     _fmt(A1_rho_all, 3),     "Spearman rho (topo dist vs beh dist), all pairs")
emit("statDegAonePAll",       _fmt_p(A1_p_all),         "p-value for rho_all")
emit("statDegAoneRhoWithin",  _fmt(A1_rho_within, 3),  "rho within-mouse pairs")
emit("statDegAonePWithin",    _fmt_p(A1_p_within),      "p within-mouse")
emit("statDegAoneRhoBetween", _fmt(A1_rho_between, 3), "rho between-mouse pairs")
emit("statDegAonePBetween",   _fmt_p(A1_p_between),     "p between-mouse")
emit("statDegAoneMwBehP",     _fmt_p(A1_mw_beh_p),     "MW p: beh dist within < between (specialisation confirmed)")
emit("statDegAoneNPairs",     str(A1_n_pairs),           "number of agent pairs in A1 correlation")

# -- A2 -----------------------------------------------------------------------
lines_tex.append("")
lines_tex.append("% -- A2: Within-mouse structural variation vs fitness --")

emit("statDegAtwoRhoTopo",        _fmt(A2_rho_topo, 3),        "Spearman rho: topo dist vs delta_fit (within-mouse)")
emit("statDegAtwoPTopo",          _fmt_p(A2_p_topo),            "p for rho_topo")
emit("statDegAtwoRhoMag",         _fmt(A2_rho_mag, 3),          "Spearman rho: mag dist vs delta_fit")
emit("statDegAtwoPMag",           _fmt_p(A2_p_mag),              "p for rho_mag")
emit("statDegAtwoPartialRhoTopo", _fmt(A2_partial_rho_topo, 3), "partial rho: topo given mag")
emit("statDegAtwoPartialPTopo",   _fmt_p(A2_partial_p_topo),    "partial p: topo given mag")
emit("statDegAtwoPartialRhoMag",  _fmt(A2_partial_rho_mag, 3),  "partial rho: mag given topo")
emit("statDegAtwoPartialPMag",    _fmt_p(A2_partial_p_mag),     "partial p: mag given topo")
emit("statDegAtwoNPairs",         str(A2_n_pairs),               "number of within-mouse replicate pairs")

# -- A3 -----------------------------------------------------------------------
lines_tex.append("")
lines_tex.append("% -- A3: Axis-specific MI (Tononi-style) -- values from consolidated_summary.txt")

emit("statDegAthreeMiTopo",  _fmt(A3_mi_topo, 4),  "MI: topology axis vs behavioral identity")
emit("statDegAthreeNmiTopo", _fmt(A3_nmi_topo, 4), "NMI: topology axis")
emit("statDegAthreeMiMag",   _fmt(A3_mi_mag, 4),   "MI: magnitude axis")
emit("statDegAthreeNmiMag",  _fmt(A3_nmi_mag, 4),  "NMI: magnitude axis")
emit("statDegAthreeMiSign",  _fmt(A3_mi_sign, 4),  "MI: sign (E/I) axis")
emit("statDegAthreeNmiSign", _fmt(A3_nmi_sign, 4), "NMI: sign axis")
emit("statDegAthreeNmiMax",  _fmt(A3_nmi_max, 4),  "maximum NMI across all three structural axes")

# -- A4 (Supplementary) -------------------------------------------------------
lines_tex.append("")
lines_tex.append("% -- A4: Sensitivity RSA (Supplementary) -- values from consolidated_summary.txt")

emit("statDegAfourMwP",         _fmt_p(A4_mw_p),           "MW p: sensitivity within vs between (Supp)")
emit("statDegAfourMantelRho",   _fmt(A4_mantel_rho, 3),    "Mantel rho: sensitivity RSA (Supp)")
emit("statDegAfourMantelPPerm", _fmt_p(A4_mantel_p_perm),  "Mantel p permutation: sensitivity RSA (Supp)")

# -- A5 -----------------------------------------------------------------------
lines_tex.append("")
lines_tex.append("% -- A5: Random agent null --")

emit("statDegAfiveRhoRand",   _fmt(A5_rho_rand, 3),   "Spearman rho: topo vs beh dist, random agents")
emit("statDegAfivePRand",     _fmt_p(A5_p_rand),       "p for rho_rand")
emit("statDegAfiveMwWithinP", _fmt_p(A5_mw_within_p), "MW p: within vs between beh dist, random agents")

# -- A6 -----------------------------------------------------------------------
lines_tex.append("")
lines_tex.append("% -- A6: Generalist vs specialist --")

emit("statDegAsixMwTopoP",         _fmt_p(A6_mw_p_topo),      "MW p: gen vs spec topology similarity")
emit("statDegAsixGenTopoMean",     _fmt(A6_gen_topo_mean, 3),  "mean topology cosine similarity, generalists")
emit("statDegAsixSpecTopoMean",    _fmt(A6_spec_topo_mean, 3), "mean topology cosine similarity, specialists")
emit("statDegAsixMeanCostPct",     _fmt_pct(A6_mean_cost_pct), "mean generalist fitness cost pct")
emit("statDegAsixMinCostPct",      _fmt_pct(A6_min_cost_pct),  "min per-mouse generalist fitness cost pct")
emit("statDegAsixMaxCostPct",      _fmt_pct(A6_max_cost_pct),  "max per-mouse generalist fitness cost pct")
emit("statDegAsixHardestMouse",    A6_hardest_mouse,            "mouse with highest generalist fitness cost")
emit("statDegAsixEasiestMouse",    A6_easiest_mouse,            "mouse with lowest generalist fitness cost")
emit("statDegAsixSpecSensVarMean", _fmt(A6_spec_var_mean, 3),  "mean sensitivity variance per neuron, specialists")
emit("statDegAsixGenSensVarMean",  _fmt(A6_gen_var_mean, 3),   "mean sensitivity variance per neuron, generalists (normalized)")
if np.isfinite(A6_var_ratio):
    emit("statDegAsixVarRatio", _fmt(A6_var_ratio, 1), "sensitivity variance ratio: specialist / generalist")
else:
    emit("statDegAsixVarRatio", ">9999", "sensitivity variance ratio (gen var near zero)")


# ---------------------------------------------------------------------------
# write files
# ---------------------------------------------------------------------------

OUT_TEX.write_text("\n".join(lines_tex) + "\n")
OUT_TXT.write_text("\n".join(lines_txt) + "\n")

n_macros = sum(1 for l in lines_tex if l.startswith("\\"))
print(f"Written {n_macros} macros to:")
print(f"  {OUT_TEX}")
print(f"  {OUT_TXT}")
