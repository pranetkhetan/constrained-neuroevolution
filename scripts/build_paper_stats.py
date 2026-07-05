#!/usr/bin/env python
r"""
Single source of truth for all paper-quoted statistics.

Reads from analysis pkl/csv files and writes:
  stats/paper_stats.tex   -- LaTeX \newcommand macros for both papers
  stats/paper_stats.txt   -- human-readable key=value listing

Both paper/latex/main_readable.tex and paper_v2/latex/main_v2.tex
\input this file. Each paper uses the subset of macros it needs;
unused \newcommand definitions are silent in LaTeX.

Re-run after any analysis update to keep both papers current:
    python scripts/build_paper_stats.py

Block 1  (\stat...)     -- core per-mouse, ANOVA, permutation, dynamics stats
Block 2  (\statDeg...)  -- degeneracy analyses A1-A6
"""
from __future__ import annotations

import csv
import pickle
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats as _scipy_stats

PROJECT  = Path(__file__).resolve().parent.parent
ANALYSIS = PROJECT / "analysis"
DEGENERACY = ANALYSIS / "degeneracy_analyses"
OUT_DIR  = PROJECT / "stats"
OUT_TEX  = OUT_DIR / "paper_stats.tex"
OUT_TXT  = OUT_DIR / "paper_stats.txt"

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

MICE   = ["B5", "B6", "B7", "D3", "D4", "D5", "D7", "D8", "D9"]
B_MICE = [m for m in MICE if m.startswith("B")]
D_MICE = [m for m in MICE if m.startswith("D")]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("cupy"):
            module = module.replace("cupy._core.core", "numpy").replace("cupy", "numpy")
        elif module == "core" or module.startswith("core."):
            module = "numpy." + module
        return super().find_class(module, name)


def _load(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (ModuleNotFoundError, AttributeError):
        with open(path, "rb") as f:
            return _CpuUnpickler(f).load()


def _fmt(x, prec: int = 3) -> str:
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    f = float(x)
    if not np.isfinite(f):
        return "NA"
    return f"{f:.{prec}f}"


def _fmt_p(p) -> str:
    p = float(p)
    if not np.isfinite(p):
        return "NA"
    if p < 1e-3:
        return "<0.001"
    return f"{p:.3f}"


def _fmt_p_exact(p, prec: int = 4) -> str:
    p = float(p)
    if not np.isfinite(p):
        return "NA"
    if p < 10 ** (-prec):
        s = f"{p:.1e}"
        mantissa, exp = s.split("e")
        return f"\\ensuremath{{{mantissa} \\times 10^{{{int(exp)}}}}}"
    return f"{p:.{prec}f}"


_DIGIT_WORDS = {
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
}


def _strip_digits(s: str) -> str:
    return "".join(_DIGIT_WORDS[c] if c.isdigit() else c for c in s)


def _macroname(section: str, key: str) -> str:
    def camel(s):
        parts = s.replace("-", "_").split("_")
        return "".join(p[0].upper() + p[1:].lower() for p in parts if p)
    return _strip_digits(camel(section) + camel(key))


# ---------------------------------------------------------------------------
# Block 1 — StatBag (core macros, \stat...)
# ---------------------------------------------------------------------------

class StatBag:
    def __init__(self):
        self.entries = []

    def add(self, section, key, raw, formatted, comment=""):
        self.entries.append((section, key, raw, formatted, comment))

    def tex_lines(self) -> list[str]:
        lines: list[str] = []
        current = None
        seen: set[str] = set()
        for section, key, _raw, formatted, comment in self.entries:
            if section != current:
                lines.append(f"% ----- {section} -----")
                current = section
            macro = _macroname(section, key)
            if macro in seen:
                lines.append(f"% (skipped duplicate: \\stat{macro})")
                continue
            seen.add(macro)
            cmt = f"  % {comment}" if comment else ""
            lines.append(f"\\newcommand{{\\stat{macro}}}{{{formatted}}}{cmt}")
        return lines

    def txt_lines(self) -> list[str]:
        lines: list[str] = []
        current = None
        col = max((len(k) for _, k, _, _, _ in self.entries), default=20) + 2
        for section, key, _raw, formatted, comment in self.entries:
            if section != current:
                lines.append("")
                lines.append(f"## {section}")
                current = section
            cmt = f"   # {comment}" if comment else ""
            lines.append(f"  {key.ljust(col)} = {formatted}{cmt}")
        return lines


def _build_core(bag: StatBag) -> None:
    """Populate StatBag with Block 1 (core) statistics."""

    # =========================================================================
    # cross_mouse generalisation
    # =========================================================================
    G = np.load(ANALYSIS / "generalization_matrix.npy")
    diag = np.diag(G)
    off_mask = ~np.eye(9, dtype=bool)
    off = G[off_mask]
    spec_ratio = diag.mean() / off.mean()

    bag.add("cross_mouse", "diag_mean",   diag.mean(),           _fmt(diag.mean(), 3),           "mean own-mouse fitness (diagonal)")
    bag.add("cross_mouse", "diag_std",    diag.std(ddof=1),      _fmt(diag.std(ddof=1), 3),      "SD across diagonal entries")
    bag.add("cross_mouse", "off_mean",    off.mean(),            _fmt(off.mean(), 3),             "mean cross-mouse fitness (off-diagonal)")
    bag.add("cross_mouse", "off_std",     off.std(ddof=1),       _fmt(off.std(ddof=1), 3),       "SD across off-diagonal entries")
    bag.add("cross_mouse", "spec_ratio",  spec_ratio,            _fmt(spec_ratio, 3),             "specialization ratio = mean diag / mean off-diag")
    bag.add("cross_mouse", "spec_ratio_pct", spec_ratio * 100,   _fmt(spec_ratio * 100, 1),       "spec_ratio as percent")
    bag.add("cross_mouse", "spec_gap_pct", (1 - spec_ratio)*100, _fmt((1 - spec_ratio)*100, 1),  "(1 - spec_ratio) as percent: own-mouse advantage")

    fitness_gap = float(off.mean() - diag.mean())
    bag.add("cross_mouse", "fitness_gap", fitness_gap, _fmt(fitness_gap, 3),
            "absolute fitness gap: mean cross-mouse minus mean own-mouse fitness")

    _a5_rf = _load(ANALYSIS / "A5_random_null.pkl")
    _rand_baseline = float(np.array(_a5_rf["fitness_matrix"]).mean())
    bag.add("cross_mouse", "gap_vs_baseline_pct",     fitness_gap / _rand_baseline * 100,
            _fmt(fitness_gap / _rand_baseline * 100, 1),
            "fitness_gap as pct of random baseline")
    bag.add("cross_mouse", "gap_vs_evo_improvement_pct",
            fitness_gap / (_rand_baseline - float(diag.mean())) * 100,
            _fmt(fitness_gap / (_rand_baseline - float(diag.mean())) * 100, 1),
            "fitness_gap as pct of evolutionary improvement")

    per_mouse_ratios = {}
    for i, m in enumerate(MICE):
        own = G[i, i]
        per_mouse_ratios[m] = own / np.delete(G[i, :], i).mean()

    most_spec  = min(per_mouse_ratios, key=per_mouse_ratios.get)
    least_spec = max(per_mouse_ratios, key=per_mouse_ratios.get)
    bag.add("cross_mouse", "most_specialized_mouse",  most_spec,  most_spec,
            f"smallest own/other ratio = {per_mouse_ratios[most_spec]:.3f}")
    bag.add("cross_mouse", "most_specialized_ratio",  per_mouse_ratios[most_spec],
            _fmt(per_mouse_ratios[most_spec], 3))
    bag.add("cross_mouse", "least_specialized_mouse", least_spec, least_spec,
            f"largest own/other ratio = {per_mouse_ratios[least_spec]:.3f}")
    bag.add("cross_mouse", "least_specialized_ratio", per_mouse_ratios[least_spec],
            _fmt(per_mouse_ratios[least_spec], 3))

    spec_index = 1.0 - spec_ratio
    bag.add("cross_mouse", "spec_index", spec_index, _fmt(spec_index, 3),
            "specialization index = 1 - spec_ratio")
    bag.add("cross_mouse", "most_specialized_index",  1.0 - per_mouse_ratios[most_spec],
            _fmt(1.0 - per_mouse_ratios[most_spec], 3),  f"= 1 - ratio ({most_spec})")
    bag.add("cross_mouse", "least_specialized_index", 1.0 - per_mouse_ratios[least_spec],
            _fmt(1.0 - per_mouse_ratios[least_spec], 3), f"= 1 - ratio ({least_spec})")

    # =========================================================================
    # per_metric component decomposition
    # =========================================================================
    cmp = _load(ANALYSIS / "cross_mouse_per_metric.pkl")
    for name, M in cmp["component_matrices"].items():
        d = np.diag(M); o = M[~np.eye(9, dtype=bool)]; r = d.mean() / o.mean()
        bag.add("per_metric", f"{name}_own",   d.mean(), _fmt(d.mean(), 4))
        bag.add("per_metric", f"{name}_other", o.mean(), _fmt(o.mean(), 4))
        bag.add("per_metric", f"{name}_ratio", r,        _fmt(r, 3),
                f"specialization ratio for {name} component")

    # =========================================================================
    # specificity / permutation Option A & B
    # =========================================================================
    sp = _load(ANALYSIS / "specificity_results.pkl")
    A = sp["option_A"]; B = sp["option_B"]
    spec_own   = np.array(A["delta_own"])
    spec_other = np.array(A["delta_other"])

    bag.add("permutation_optionA", "spec_own_mean",   spec_own.mean(),         _fmt(spec_own.mean(), 3),         "mean Δfit_own across 9 mice")
    bag.add("permutation_optionA", "spec_own_std",    spec_own.std(ddof=1),    _fmt(spec_own.std(ddof=1), 3))
    bag.add("permutation_optionA", "spec_other_mean", spec_other.mean(),        _fmt(spec_other.mean(), 3),       "mean Δfit_other across 9 mice")
    bag.add("permutation_optionA", "spec_other_std",  spec_other.std(ddof=1),  _fmt(spec_other.std(ddof=1), 3))
    bag.add("permutation_optionA", "wilcoxon_p",      A["wilcoxon_p"],          _fmt_p_exact(A["wilcoxon_p"], 4), "Wilcoxon signed-rank own > other (n=9)")
    bag.add("permutation_optionA", "mannwhitney_p",   A["mannwhitney_p"],       _fmt_p_exact(A["mannwhitney_p"], 4))
    bag.add("permutation_optionA", "n_perm",          A["N_PERM"],              str(A["N_PERM"]),                 "permutations per cell")

    bag.add("permutation_optionB", "ratio_topology",   B["ratio_topology"],   _fmt(B["ratio_topology"], 3),   "topology MSE / baseline MSE")
    bag.add("permutation_optionB", "ratio_magnitude",  B["ratio_magnitude"],  _fmt(B["ratio_magnitude"], 3),  "magnitude MSE / baseline MSE")
    bag.add("permutation_optionB", "p_topo_vs_baseline", B["p_topo_vs_baseline"], _fmt_p_exact(B["p_topo_vs_baseline"], 4))
    bag.add("permutation_optionB", "p_mag_vs_baseline",  B["p_mag_vs_baseline"],  _fmt_p_exact(B["p_mag_vs_baseline"], 4))
    bag.add("permutation_optionB", "p_topo_vs_mag",    B.get("p_topo_vs_mag", float("nan")),
            _fmt_p_exact(B.get("p_topo_vs_mag", float("nan")), 4))
    bag.add("permutation_optionB", "baseline_mse",    B["baseline_mse"],    _fmt(B["baseline_mse"], 4))
    bag.add("permutation_optionB", "mean_topo_mse",   B["mean_topo_mse"],   _fmt(B["mean_topo_mse"], 4))
    bag.add("permutation_optionB", "mean_mag_mse",    B["mean_mag_mse"],    _fmt(B["mean_mag_mse"], 4))
    bag.add("permutation_optionB", "n_perm",          B["N_PERM"],          str(B["N_PERM"]))

    # magnitude degeneracy fraction
    _wd = _load(ANALYSIS / "weight_data.pkl")
    _W_evo  = np.array(_wd["weight_vectors"]).reshape(54, 14, 14)
    _W_rand = np.array(_wd["random_weight_vectors"]).reshape(54, 14, 14)

    def _mag_degen_frac(W_all):
        fracs = []
        for _W in W_all:
            n_u = n_m = 0
            for _src in range(14):
                _nz = _W[_src, _W[_src] != 0]
                if len(_nz) == 0:
                    continue
                if len(np.unique(np.round(np.abs(_nz), 3))) == 1:
                    n_u += 1
                else:
                    n_m += 1
            nc = n_u + n_m
            fracs.append(n_u / nc if nc > 0 else 0.0)
        return float(np.mean(fracs)) * 100.0

    bag.add("magnitude_degeneracy", "evolved_frac",
            _mag_degen_frac(_W_evo),  f"{_mag_degen_frac(_W_evo):.1f}",
            "% connected sources with uniform magnitude (evolved)")
    bag.add("magnitude_degeneracy", "random_frac",
            _mag_degen_frac(_W_rand), f"{_mag_degen_frac(_W_rand):.1f}",
            "% connected sources with uniform magnitude (random)")

    # autapse (self-connection) prevalence: W[i,j] from i to j, so W[i,i] is autapse
    # Only interneuron/motor neurons (indices 6-13) can have incoming connections
    _n_with_autapse = sum(
        1 for _W in _W_evo if np.any(np.abs(np.diag(_W)[6:]) > 0)
    )
    _autapse_frac = _n_with_autapse / len(_W_evo)
    bag.add("network_arch", "autapse_n",    _n_with_autapse, str(_n_with_autapse),
            "number of evolved agents with ≥1 self-connection (interneuron/motor)")
    bag.add("network_arch", "autapse_total", len(_W_evo), str(len(_W_evo)),
            "total evolved agents")
    bag.add("network_arch", "autapse_pct",  _autapse_frac * 100,
            f"{round(_autapse_frac * 100)}", "% evolved agents with ≥1 autapse (integer %)")

    # dose-response Spearman
    from scipy.stats import spearmanr as _spearmanr
    _per_mouse_spec_index = np.array([1.0 - per_mouse_ratios[m] for m in MICE])
    _per_mouse_dfit_ratio = spec_own / spec_other
    _rho, _ = _spearmanr(_per_mouse_spec_index, _per_mouse_dfit_ratio)
    _n_sp   = len(MICE)
    _rng_sp = np.random.default_rng(1)
    _perm_count = sum(
        1 for _ in range(10000)
        if abs(_spearmanr(_per_mouse_spec_index[_rng_sp.permutation(_n_sp)],
                          _per_mouse_dfit_ratio)[0]) >= abs(_rho)
    )
    _perm_p = _perm_count / 10000
    _rng_boot = np.random.default_rng(1)
    _boot_rhos = [
        _spearmanr(
            _per_mouse_spec_index[_idx := _rng_boot.integers(0, _n_sp, _n_sp)],
            _per_mouse_dfit_ratio[_idx],
        )[0]
        for _ in range(10000)
    ]
    _ci_lo = float(np.percentile(_boot_rhos, 2.5))
    _ci_hi = float(np.percentile(_boot_rhos, 97.5))
    bag.add("permutation_optionA", "spearman_rho",        float(_rho),  _fmt(_rho, 2),
            "Spearman rho: per-mouse spec index vs own/other Δfit ratio (dose-response)")
    bag.add("permutation_optionA", "spearman_perm_p",     _perm_p,      _fmt_p(_perm_p),
            "permutation p for Spearman rho (10000 permutations, seed=1)")
    bag.add("permutation_optionA", "spearman_boot_ci_lo", _ci_lo,       _fmt(_ci_lo, 2),
            "bootstrap 95% CI lower bound (10000 resamples, seed=1)")
    bag.add("permutation_optionA", "spearman_boot_ci_hi", _ci_hi,       _fmt(_ci_hi, 2),
            "bootstrap 95% CI upper bound")

    # =========================================================================
    # ANOVA / connection-feature stats
    # =========================================================================
    st   = _load(ANALYSIS / "stats_results.pkl")
    anova = st["anova"]; perm = st["permutation"]; evr = st["evolved_vs_random"]

    rows = sorted(
        [(k, v) for k, v in anova.items() if np.isfinite(v.get("p_raw", float("nan")))],
        key=lambda kv: kv[1]["p_raw"],
    )
    if rows:
        top_name, top_v = rows[0]
        bag.add("anova", "top_feature",       top_name,              top_name.replace("_", "\\_"), "smallest raw ANOVA p-value")
        bag.add("anova", "top_feature_F",     top_v["F"],            _fmt(top_v["F"], 3))
        bag.add("anova", "top_feature_p",     top_v["p_raw"],        _fmt_p(top_v["p_raw"]))
        bag.add("anova", "top_feature_eta2",  top_v["eta_squared"],  _fmt(top_v["eta_squared"], 3))
        bag.add("anova", "top_feature_p_perm",perm[top_name]["p_perm"], _fmt_p(perm[top_name]["p_perm"]))
        bag.add("anova", "top_feature_p_fdr", top_v["p_fdr"],        _fmt_p(top_v["p_fdr"]))

    n_sig_fdr = sum(1 for v in anova.values() if v.get("significant", False))
    bag.add("anova", "n_features_tested",   len(anova),  str(len(anova)))
    bag.add("anova", "n_significant_fdr",   n_sig_fdr,   str(n_sig_fdr), "features surviving BH-FDR")

    eta2_rows = sorted(
        [(k, v) for k, v in anova.items() if np.isfinite(v.get("eta_squared", float("nan")))],
        key=lambda kv: -kv[1]["eta_squared"],
    )
    for rank, (fname, fv) in enumerate(eta2_rows[:3], start=1):
        bag.add("anova", f"rank{rank}_feature",      fname,              fname.replace("_", "\\_"))
        bag.add("anova", f"rank{rank}_feature_F",    fv["F"],            _fmt(fv["F"], 3))
        bag.add("anova", f"rank{rank}_feature_p",    fv["p_raw"],        _fmt_p(fv["p_raw"]))
        bag.add("anova", f"rank{rank}_feature_eta2", fv["eta_squared"],  _fmt(fv["eta_squared"], 3))

    for k in ["density", "n_connections", "ei_ratio", "sm_count", "im_count",
              "mm_count", "inter_in_mean", "w_mean_mag", "frac_strong"]:
        v = evr[k]
        bag.add("evolved_vs_random", f"{k}_evolved_mean", v["evolved_mean"], _fmt(v["evolved_mean"], 3))
        bag.add("evolved_vs_random", f"{k}_random_mean",  v["random_mean"],  _fmt(v["random_mean"], 3))
        bag.add("evolved_vs_random", f"{k}_t",            v["t"],            _fmt(v["t"], 2))
        bag.add("evolved_vs_random", f"{k}_p",            v["p"],            _fmt_p_exact(v["p"]))
        bag.add("evolved_vs_random", f"{k}_d",            v["cohens_d"],     _fmt(v["cohens_d"], 2))

    bag.add("evolved_vs_random", "n_evolved", st["n_evolved"], str(st["n_evolved"]))
    bag.add("evolved_vs_random", "n_random",  st["n_random"],  str(st["n_random"]))
    bag.add("evolved_vs_random", "frac_strong_evolved_pct",
            evr["frac_strong"]["evolved_mean"] * 100,
            f"{evr['frac_strong']['evolved_mean']*100:.0f}",
            "% evolved at max weight (integer pct for Table 1)")
    bag.add("evolved_vs_random", "frac_strong_random_pct",
            evr["frac_strong"]["random_mean"] * 100,
            f"{evr['frac_strong']['random_mean']*100:.0f}",
            "% random at max weight")

    # min p_FDR across all 18 features (floor to 2 d.p. so "> X" claim stays valid)
    import math as _math
    all_fdr = {k: v["p_fdr"] for k, v in anova.items() if "p_fdr" in v}
    min_fdr = min(all_fdr.values())
    min_fdr_floor = _math.floor(min_fdr * 100) / 100
    bag.add("anova", "min_p_fdr_floor", min_fdr_floor, _fmt(min_fdr_floor, 2),
            "floor(min p_FDR across all features) — safe lower bound for '> X' claim")
    # min p_FDR for weight magnitude features specifically
    mag_fdr = min(all_fdr.get(k, 1.0) for k in ["w_mean_mag", "frac_strong"])
    bag.add("anova", "min_p_fdr_mag", mag_fdr, _fmt(mag_fdr, 2),
            "min p_FDR for weight magnitude features (w_mean_mag, frac_strong)")

    # =========================================================================
    # deep_analysis: power, strain clustering, weight ANOVA
    # =========================================================================
    da = _load(ANALYSIS / "deep_analysis_results.pkl")
    pa = da["power_analysis"]
    if rows:
        top_name = rows[0][0]
        bag.add("power", "top_feature_power", pa[top_name]["power"],
                _fmt(pa[top_name]["power"], 3), f"attained power for ANOVA on {top_name}")
        bag.add("power", "top_feature_mde",   pa[top_name]["min_detectable_eta2"],
                _fmt(pa[top_name]["min_detectable_eta2"], 3), "minimum detectable eta^2 at 80% power")

    sc = da["strain_clustering"]
    bag.add("strain", "within_same_mean",    sc["within_same_mean"],                    _fmt(sc["within_same_mean"], 3))
    bag.add("strain", "within_B_mean",       sc["within_B_mean"],                       _fmt(sc["within_B_mean"], 3))
    bag.add("strain", "within_D_mean",       sc["within_D_mean"],                       _fmt(sc["within_D_mean"], 3))
    bag.add("strain", "between_mean",        sc["between_mean"],                        _fmt(sc["between_mean"], 3))
    bag.add("strain", "between_minus_within",sc["between_mean"] - sc["within_same_mean"],
            _fmt(sc["between_mean"] - sc["within_same_mean"], 3))
    bag.add("strain", "p_value",             sc["p_value"],                             _fmt_p_exact(sc["p_value"]))
    bag.add("strain", "u_stat",              sc["U_stat"],                              _fmt(sc["U_stat"], 1))

    wrows = sorted(
        [(k, v) for k, v in da["weight_anova"].items()
         if np.isfinite(v.get("p_raw", float("nan")))],
        key=lambda kv: kv[1]["p_raw"],
    )
    if wrows:
        wn, wv = wrows[0]
        bag.add("weight_anova", "top_feature",      wn,              wn.replace("_", "\\_"))
        bag.add("weight_anova", "top_feature_F",    wv["F"],         _fmt(wv["F"], 3))
        bag.add("weight_anova", "top_feature_p",    wv["p_raw"],     _fmt_p(wv["p_raw"]))
        bag.add("weight_anova", "top_feature_eta2", wv["eta_squared"], _fmt(wv["eta_squared"], 3))

    ws = da["weight_similarity"]
    bag.add("weight_similarity", "within_mean",  ws["within_mean"],  _fmt(ws["within_mean"], 4))
    bag.add("weight_similarity", "between_mean", ws["between_mean"], _fmt(ws["between_mean"], 4))

    # =========================================================================
    # cosine_evolved_random
    # =========================================================================
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim
    from scipy.stats import mannwhitneyu as _mw

    wd = _load(ANALYSIS / "weight_data.pkl")
    ev_vecs = np.asarray(wd["weight_vectors"])
    rd_vecs = np.asarray(wd["random_weight_vectors"])
    n_e_, n_r_ = len(ev_vecs), len(rd_vecs)
    triu_e = _cosine_sim(ev_vecs)[np.triu_indices(n_e_, k=1)]
    triu_r = _cosine_sim(rd_vecs)[np.triu_indices(n_r_, k=1)]
    _, mw_p_evol_rand = _mw(triu_e, triu_r, alternative="greater")

    bag.add("cosine_evolved_random", "evolved_mean",    float(triu_e.mean()),     _fmt(triu_e.mean(), 3))
    bag.add("cosine_evolved_random", "evolved_std",     float(triu_e.std(ddof=1)),_fmt(triu_e.std(ddof=1), 3))
    bag.add("cosine_evolved_random", "random_mean",     float(triu_r.mean()),     _fmt(triu_r.mean(), 3))
    bag.add("cosine_evolved_random", "random_std",      float(triu_r.std(ddof=1)),_fmt(triu_r.std(ddof=1), 3))
    bag.add("cosine_evolved_random", "mannwhitney_p",   float(mw_p_evol_rand),    _fmt_p_exact(mw_p_evol_rand))
    bag.add("cosine_evolved_random", "n_evolved",       n_e_,  str(n_e_))
    bag.add("cosine_evolved_random", "n_random",        n_r_,  str(n_r_))
    bag.add("cosine_evolved_random", "n_evolved_pairs", len(triu_e), str(len(triu_e)))
    bag.add("cosine_evolved_random", "n_random_pairs",  len(triu_r), str(len(triu_r)))

    # =========================================================================
    # E/I balance — computed from weight matrices (W[src, tgt]: sign = Dale's Law)
    # =========================================================================
    _W = np.array(wd["weight_matrices"])  # (54, 14, 14), W[src, tgt]
    for _tgt_name, _tgt_idx, _section_key in [
        ("speed_motor", 12, "speed"),
        ("turn_motor",  13, "turn"),
    ]:
        _fracs = []
        for _agent_W in _W:
            _inc = _agent_W[:, _tgt_idx]
            _nz  = _inc[_inc != 0]
            if len(_nz):
                _fracs.append(float(np.sum(_nz > 0) / len(_nz)))
        _fracs = np.array(_fracs)
        bag.add("ei_balance", f"{_section_key}_mean",
                float(_fracs.mean()), _fmt(_fracs.mean(), 2),
                f"E/(E+I) {_tgt_name}: mean across 54 agents")
        bag.add("ei_balance", f"{_section_key}_median",
                float(np.median(_fracs)), _fmt(np.median(_fracs), 3),
                f"E/(E+I) {_tgt_name}: median (discrete distribution: 0, 1/3, 2/3, 1)")

    # =========================================================================
    # dynamics
    # =========================================================================
    dyn     = _load(ANALYSIS / "dynamics_results.pkl")
    dyn_full= _load(ANALYSIS / "dynamics_results_full.pkl")
    traj = dyn["trajectory"]; fp = dyn["fixed_points"]; pm_d = dyn["permutation"]
    lya  = dyn_full["lyapunov"]

    bag.add("dynamics", "trajectory_within_mean",  traj["within_sims"].mean(),   _fmt(traj["within_sims"].mean(), 3))
    bag.add("dynamics", "trajectory_between_mean", traj["between_sims"].mean(),  _fmt(traj["between_sims"].mean(), 3))
    bag.add("dynamics", "trajectory_p",            traj["mannwhitney_p"],        _fmt_p_exact(traj["mannwhitney_p"]))
    bag.add("dynamics", "fp_F",   fp["F_fp"],  _fmt(fp["F_fp"], 3))
    bag.add("dynamics", "fp_p",   fp["p_fp"],  _fmt_p(fp["p_fp"]))
    bag.add("dynamics", "fp_eta2",fp["eta2_fp"],_fmt(fp["eta2_fp"], 3))
    bag.add("dynamics", "lam_F",  fp["F_lam"], _fmt(fp["F_lam"], 3))
    bag.add("dynamics", "lam_p",  fp["p_lam"], _fmt_p(fp["p_lam"]))
    bag.add("dynamics", "lam_eta2",fp["eta2_lam"],_fmt(fp["eta2_lam"], 3))
    bag.add("dynamics", "lya_F",    float(lya["F"]),          _fmt(float(lya["F"]), 3))
    bag.add("dynamics", "lya_p",    float(lya["p_anova"]),    _fmt_p(float(lya["p_anova"])))
    bag.add("dynamics", "lya_eta2", float(lya["eta2"]),       _fmt(float(lya["eta2"]), 3))
    bag.add("dynamics", "lya_kw_H", float(lya["H"]),          _fmt(float(lya["H"]), 3))
    bag.add("dynamics", "lya_kw_p", float(lya["p_kruskal"]),  _fmt_p(float(lya["p_kruskal"])))

    _lya_vals = np.array([r["lambda1"] for r in lya["lya_results"]])
    bag.add("dynamics", "lya_mean", float(_lya_vals.mean()), _fmt(_lya_vals.mean(), 3))
    bag.add("dynamics", "lya_std",  float(_lya_vals.std()),  _fmt(_lya_vals.std(), 3))
    bag.add("dynamics", "lya_min",  float(_lya_vals.min()),  _fmt(_lya_vals.min(), 3))
    bag.add("dynamics", "lya_max",  float(_lya_vals.max()),  _fmt(_lya_vals.max(), 3))
    bag.add("dynamics", "perm_p",   pm_d["mannwhitney_p"],   _fmt_p_exact(pm_d["mannwhitney_p"]))

    perm_means = [r["mean_mse"] for r in pm_d["perm_results"]]
    same = np.array(pm_d["same_mouse_mse"])
    bag.add("dynamics", "perm_mean_mse",       float(np.mean(perm_means)),           _fmt(np.mean(perm_means), 3))
    bag.add("dynamics", "replicate_mean_mse",  float(np.mean(same)),                 _fmt(np.mean(same), 3))
    bag.add("dynamics", "perm_over_replicate", float(np.mean(perm_means)/np.mean(same)),
            _fmt(np.mean(perm_means)/np.mean(same), 2), "MSE ratio: topology perm / replicate baseline")

    # =========================================================================
    # source_sensitivity
    # =========================================================================
    ss = _load(ANALYSIS / "source_sensitivity_results.pkl")
    bag.add("source_sensitivity", "within_r_mean",  float(np.mean(ss["within_r"])),  _fmt(np.mean(ss["within_r"]), 3))
    bag.add("source_sensitivity", "between_r_mean", float(np.mean(ss["between_r"])), _fmt(np.mean(ss["between_r"]), 3))
    bag.add("source_sensitivity", "mw_p",           ss["mw_p"],                       _fmt_p_exact(ss["mw_p"]))

    # =========================================================================
    # generalist (overall + per-mouse)
    # =========================================================================
    g = _load(ANALYSIS / "generalist_results.pkl")
    results_C = g["results_C"]; pkl_mice = g["MICE"]
    gen_delta = {
        m: [float(results_C[r][m]["delta_fit"]) for r in range(len(results_C))]
        for m in pkl_mice
    }
    gen_means  = np.array([np.mean(gen_delta[m]) for m in MICE])
    gen_grand  = float(np.mean(gen_means))

    bag.add("generalist", "grand_mean",    gen_grand,               _fmt(gen_grand, 3))
    bag.add("generalist", "min_per_mouse", float(np.min(gen_means)), _fmt(np.min(gen_means), 3),
            f"smallest per-mouse mean Δfit (mouse: {MICE[int(np.argmin(gen_means))]})")
    bag.add("generalist", "max_per_mouse", float(np.max(gen_means)), _fmt(np.max(gen_means), 3),
            f"largest per-mouse mean Δfit (mouse: {MICE[int(np.argmax(gen_means))]})")
    bag.add("generalist", "n_replicates", len(results_C), str(len(results_C)))
    bag.add("generalist", "n_perm",       g["N_PERM_C"],  str(g["N_PERM_C"]))
    bag.add("generalist", "kruskal_h",    g["kruskal"]["H"], _fmt(g["kruskal"]["H"], 3))
    bag.add("generalist", "kruskal_p",    g["kruskal"]["p"], _fmt_p_exact(g["kruskal"]["p"]))
    bag.add("generalist", "anova_f",      g["anova"]["F"],   _fmt(g["anova"]["F"], 3))
    bag.add("generalist", "anova_p",      g["anova"]["p"],   _fmt_p_exact(g["anova"]["p"]))

    a3 = _load(ANALYSIS / "A3_difficulty_correlation.pkl")
    gen_fits  = np.array([a3["gen_fit"][m] for m in MICE])
    spec_fits = np.array([a3["spec_own"][m] for m in MICE])
    spec_mean_fit = float(spec_fits.mean())

    bag.add("generalist", "fitness_min",            float(gen_fits.min()),   _fmt(gen_fits.min(), 3))
    bag.add("generalist", "fitness_max",            float(gen_fits.max()),   _fmt(gen_fits.max(), 3))
    bag.add("generalist", "fitness_mean",           float(gen_fits.mean()),  _fmt(gen_fits.mean(), 3))
    bag.add("generalist", "specialist_fitness_mean",spec_mean_fit,           _fmt(spec_mean_fit, 3))
    # Note: per-mouse generalist cost is reported from the A6 analysis
    # (\statDegAsixMinCostPct / \statDegAsixMaxCostPct / \statDegAsixCostRatioOfMeans),
    # which uses the per-mouse specialist reference. The old aggregate
    # gen_cost_min/max_pct macros (pooled spec mean) were stale and unused in v2.

    # =========================================================================
    # A2 deltafit decomposition
    # =========================================================================
    a2 = _load(ANALYSIS / "A2_deltafit_decomposition.pkl")
    bag.add("decomposition", "mean_own",              a2["mean_own"],              _fmt(a2["mean_own"], 3))
    bag.add("decomposition", "mean_gen",              a2["mean_gen"],              _fmt(a2["mean_gen"], 3))
    bag.add("decomposition", "mean_other",            a2["mean_other"],            _fmt(a2["mean_other"], 3))
    bag.add("decomposition", "individual_calibration",a2["individual_calibration"],_fmt(a2["individual_calibration"], 3))
    bag.add("decomposition", "generic_floor",         a2["generic_floor"],         _fmt(a2["generic_floor"], 3))
    bag.add("decomposition", "own_minus_gen",
            a2["mean_own"] - a2["mean_gen"], _fmt(a2["mean_own"] - a2["mean_gen"], 3))

    # =========================================================================
    # A1 cosine bootstrap
    # =========================================================================
    a1 = _load(ANALYSIS / "A1_cosine_bootstrap.pkl")
    bag.add("cosine", "spec_mean",    a1["spec_mean"], _fmt(a1["spec_mean"], 3))
    bag.add("cosine", "gen_mean",     a1["gen_mean"],  _fmt(a1["gen_mean"], 3))
    bag.add("cosine", "gen_std",      a1["gen_std"],   _fmt(a1["gen_std"], 3))
    bag.add("cosine", "p_val",        a1["p_val"],     _fmt_p_exact(a1["p_val"]))
    bag.add("cosine", "n_spec_pairs", a1["n_spec_pairs"], str(a1["n_spec_pairs"]))
    bag.add("cosine", "n_gen_pairs",  a1["n_gen_pairs"],  str(a1["n_gen_pairs"]))

    # =========================================================================
    # A3 difficulty correlation
    # =========================================================================
    bag.add("difficulty", "pearson_r", a3["pearson_r"], _fmt(a3["pearson_r"], 3))
    bag.add("difficulty", "pearson_p", a3["pearson_p"], _fmt_p_exact(a3["pearson_p"]))

    # =========================================================================
    # A4 strain confound
    # =========================================================================
    a4 = _load(ANALYSIS / "A4_strain_confound.pkl")
    bag.add("strain_confound", "mean_own",          a4["mean_own"],           _fmt(a4["mean_own"], 3))
    bag.add("strain_confound", "mean_within",       a4["mean_within"],        _fmt(a4["mean_within"], 3))
    bag.add("strain_confound", "mean_cross",        a4["mean_cross"],         _fmt(a4["mean_cross"], 3))
    bag.add("strain_confound", "p_within_vs_cross", a4["p_within_vs_cross"],  _fmt_p_exact(a4["p_within_vs_cross"]))
    bag.add("strain_confound", "p_mw",              a4["p_mw"],               _fmt_p_exact(a4["p_mw"]))
    bag.add("strain_confound", "b_ratio",           a4["b_ratio"],            _fmt(a4["b_ratio"], 3))
    bag.add("strain_confound", "d_ratio",           a4["d_ratio"],            _fmt(a4["d_ratio"], 3))

    # =========================================================================
    # A5 random null
    # =========================================================================
    a5 = _load(ANALYSIS / "A5_random_null.pkl")
    bag.add("random_null", "mean_ratio",  a5["mean_ratio"], _fmt(a5["mean_ratio"], 3))
    bag.add("random_null", "std_ratio",   a5["std_ratio"],  _fmt(a5["std_ratio"], 3))
    bag.add("random_null", "mean_index",  1.0 - float(a5["mean_ratio"]),
            _fmt(1.0 - float(a5["mean_ratio"]), 3))
    rand_fm = np.array(a5["fitness_matrix"])
    per_mouse_rf = rand_fm.mean(axis=0)
    bag.add("random_null", "baseline_mean", float(per_mouse_rf.mean()), _fmt(float(per_mouse_rf.mean()), 2))
    bag.add("random_null", "baseline_min",  float(per_mouse_rf.min()),  _fmt(float(per_mouse_rf.min()), 2))
    bag.add("random_null", "baseline_max",  float(per_mouse_rf.max()),  _fmt(float(per_mouse_rf.max()), 2))

    _cme = _load(ANALYSIS / "cross_mouse_evaluation.pkl")
    _G   = np.array(_cme["matrix"])
    _n_m = _G.shape[0]
    evo_ratios_r  = np.array([_G[i, i] / np.delete(_G[i, :], i).mean() for i in range(_n_m)])
    rand_ratios_r = np.array(a5["ratios"])
    _pooled_sd = np.sqrt((evo_ratios_r.std(ddof=1)**2 + rand_ratios_r.std(ddof=1)**2) / 2)
    cohen_d_evr = float((rand_ratios_r.mean() - evo_ratios_r.mean()) / _pooled_sd)
    _, mw_p_evr = _scipy_stats.mannwhitneyu(evo_ratios_r, rand_ratios_r, alternative="less")
    bag.add("random_null", "evolved_vs_rand_d", cohen_d_evr, _fmt(cohen_d_evr, 2))
    bag.add("random_null", "evolved_vs_rand_p", float(mw_p_evr), _fmt_p_exact(float(mw_p_evr)))

    # =========================================================================
    # sensitivity (equal-weight 1:1:1:1)
    # =========================================================================
    pm_s = _load(ANALYSIS / "cross_mouse_per_metric.pkl")
    _comps = pm_s["component_matrices"]
    _equal_mat = sum(_comps[m] for m in ["markov", "occupancy", "tortuosity", "turn_bias"])
    _n = _equal_mat.shape[0]
    _eq_diag = np.diag(_equal_mat)
    _eq_off  = _equal_mat[~np.eye(_n, dtype=bool)]
    eq_ratio  = float(_eq_diag.mean() / _eq_off.mean())
    eq_index  = 1.0 - eq_ratio
    main_index = float(1.0 - np.diag(pm_s["total_matrix"]).mean() /
                       pm_s["total_matrix"][~np.eye(_n, dtype=bool)].mean())
    bag.add("sensitivity", "equal_weight_index",      eq_index,              _fmt(eq_index, 3))
    bag.add("sensitivity", "equal_weight_index_diff", eq_index - main_index, _fmt(eq_index - main_index, 3))

    # =========================================================================
    # weighting sensitivity — 5 schemes for supplementary table + figure
    # =========================================================================
    _COMPS_ORDER = ["markov", "occupancy", "tortuosity", "turn_bias"]

    def _wt_index(mat: np.ndarray) -> float:
        d = np.diag(mat); o = mat[~np.eye(mat.shape[0], dtype=bool)]
        return float(1.0 - d.mean() / o.mean())

    def _wt_n_pos(mat: np.ndarray) -> int:
        n = mat.shape[0]
        return sum(
            1 for i in range(n)
            if (1.0 - float(mat[i, i]) / float(np.delete(mat[i, :], i).mean())) > 0
        )

    # Published 2:2:1:1 — n_pos (index values come from cross_mouse block)
    _pub_mat = pm_s["total_matrix"]
    bag.add("sensitivity", "published_n_pos", _wt_n_pos(_pub_mat), str(_wt_n_pos(_pub_mat)),
            "n mice positive under published 2:2:1:1")

    # Equal 1:1:1:1 — n_pos (index already emitted above)
    bag.add("sensitivity", "equal_weight_n_pos", _wt_n_pos(_equal_mat), str(_wt_n_pos(_equal_mat)),
            "n mice positive under equal 1:1:1:1")

    # Turn-dominant 1:1:1:3
    _td_mat = (_comps["markov"] + _comps["occupancy"] + _comps["tortuosity"]
               + 3.0 * _comps["turn_bias"]) / 6.0
    bag.add("sensitivity", "turn_dom_index", _wt_index(_td_mat), _fmt(_wt_index(_td_mat), 3),
            "specialisation index under turn-dominant 1:1:1:3")
    bag.add("sensitivity", "turn_dom_n_pos", _wt_n_pos(_td_mat), str(_wt_n_pos(_td_mat)),
            "n mice positive under turn-dominant 1:1:1:3")

    # Markov-only 1:0:0:0
    _mo_mat = _comps["markov"]
    bag.add("sensitivity", "markov_only_index", _wt_index(_mo_mat), _fmt(_wt_index(_mo_mat), 3),
            "specialisation index under markov-only 1:0:0:0")
    bag.add("sensitivity", "markov_only_n_pos", _wt_n_pos(_mo_mat), str(_wt_n_pos(_mo_mat)),
            "n mice positive under markov-only 1:0:0:0")

    # Turn+Occupancy 0:2:0:1
    _ot_mat = (2.0 * _comps["occupancy"] + _comps["turn_bias"]) / 3.0
    bag.add("sensitivity", "occ_turn_index", _wt_index(_ot_mat), _fmt(_wt_index(_ot_mat), 3),
            "specialisation index under turn+occupancy 0:2:0:1")
    bag.add("sensitivity", "occ_turn_n_pos", _wt_n_pos(_ot_mat), str(_wt_n_pos(_ot_mat)),
            "n mice positive under turn+occupancy 0:2:0:1")

    # =========================================================================
    # R9 holdout
    # =========================================================================
    r9 = _load(ANALYSIS / "R9_holdout_results.pkl")
    training_index = 1.0 - float(r9["training_ratio"])
    holdout_index  = 1.0 - float(r9["holdout_ratio"])
    total_bouts    = sum(r9["bout_counts"].values())
    n_mice_holdout = len(r9["bout_counts"])
    mean_bouts_total   = int(round(total_bouts / n_mice_holdout))
    mean_bouts_holdout = int(round(total_bouts * r9["split_fraction"] / n_mice_holdout))

    bag.add("holdout", "training_ratio",    r9["training_ratio"],  _fmt(r9["training_ratio"], 3))
    bag.add("holdout", "holdout_ratio",     r9["holdout_ratio"],   _fmt(r9["holdout_ratio"], 3))
    bag.add("holdout", "training_index",    training_index,        _fmt(training_index, 3))
    bag.add("holdout", "holdout_index",     holdout_index,         _fmt(holdout_index, 3))
    bag.add("holdout", "degradation_pct",   r9["degradation_pct"], _fmt(r9["degradation_pct"], 1))
    bag.add("holdout", "split_fraction",    r9["split_fraction"],  _fmt(r9["split_fraction"], 2))
    bag.add("holdout", "total_bouts",       total_bouts,           str(total_bouts))
    bag.add("holdout", "mean_bouts_total",   mean_bouts_total,     str(mean_bouts_total))
    bag.add("holdout", "mean_bouts_holdout", mean_bouts_holdout,   str(mean_bouts_holdout))

    # =========================================================================
    # individuation foundation test (real-mouse bout-fold distinguishability)
    # analysis/individuation_results.pkl, from scripts/analyze_individuation.py
    # =========================================================================
    indiv_path = ANALYSIS / "individuation_results.pkl"
    if indiv_path.exists():
        iv = _load(indiv_path)
        iv_cF = iv["per_component"]["combined_F"]
        iv_cls = iv["classification"]
        bag.add("individuation", "within_dist",  iv_cF["within"],  _fmt(iv_cF["within"], 3))
        bag.add("individuation", "between_dist", iv_cF["between"], _fmt(iv_cF["between"], 3))
        bag.add("individuation", "mw_p",         iv_cF["mw_p"],    _fmt_p(iv_cF["mw_p"]))
        bag.add("individuation", "rank_biserial",iv_cF["rank_biserial"], _fmt(iv_cF["rank_biserial"], 3))
        bag.add("individuation", "class_acc_pct", iv_cls["acc"] * 100.0, _fmt(iv_cls["acc"] * 100.0, 0))
        bag.add("individuation", "chance_pct",    iv_cls["chance"] * 100.0, _fmt(iv_cls["chance"] * 100.0, 0))
        bag.add("individuation", "class_p",       iv_cls["p"],      _fmt_p(iv_cls["p"]))
        bag.add("individuation", "n_folds",       iv["n_folds"],    str(iv["n_folds"]))

    # =========================================================================
    # phase3: random permutation, generalist formal, power
    # =========================================================================
    p3a = _load(ANALYSIS / "phase3a_random_permutation.pkl")
    bag.add("phase3a", "evol_ratio",   p3a["evol_ratio"],   _fmt(p3a["evol_ratio"], 3))
    bag.add("phase3a", "rand_ratio",   p3a["rand_ratio"],   _fmt(p3a["rand_ratio"], 3))
    bag.add("phase3a", "mannwhitney_p",p3a["mannwhitney_p"],_fmt_p_exact(p3a["mannwhitney_p"]))

    p3b = _load(ANALYSIS / "phase3b_generalist_formal.pkl")
    bag.add("phase3b", "kw_h",       p3b["kruskal_wallis_H"],    _fmt(p3b["kruskal_wallis_H"], 3))
    bag.add("phase3b", "kw_p",       p3b["kruskal_wallis_p"],    _fmt_p_exact(p3b["kruskal_wallis_p"]))
    bag.add("phase3b", "spread_mean",p3b["generalist_spread_mean"],_fmt(p3b["generalist_spread_mean"], 3))
    bag.add("phase3b", "spread_std", p3b["generalist_spread_std"], _fmt(p3b["generalist_spread_std"], 3))
    bag.add("phase3b", "wilcoxon_p", p3b["specialist_wilcoxon_p"],_fmt_p_exact(p3b["specialist_wilcoxon_p"]))
    bag.add("phase3b", "wilcoxon_w", p3b["specialist_wilcoxon_W"],_fmt(p3b["specialist_wilcoxon_W"], 1))

    p3c = _load(ANALYSIS / "phase3c_power_analysis.pkl")
    bag.add("phase3c", "n",               p3c["n"],               str(p3c["n"]))
    bag.add("phase3c", "cohen_dz",        p3c["cohen_dz"],        _fmt(p3c["cohen_dz"], 3))
    bag.add("phase3c", "attained_power",  p3c["attained_power"],  _fmt(p3c["attained_power"], 3))
    bag.add("phase3c", "mde_d",           p3c["mde_d"],           _fmt(p3c["mde_d"], 3))
    bag.add("phase3c", "mde_raw",         p3c["mde_raw"],         _fmt(p3c["mde_raw"], 3))
    bag.add("phase3c", "required_n_80pct",p3c["required_n_80pct"],str(p3c["required_n_80pct"]))

    # Power analysis: required N per group for medium-effect η² range (0.06–0.14)
    # Used in Supplementary S4 text.
    from scipy.stats import ncf as _ncf
    def _required_n(eta2, k=9, alpha=0.05, target=0.80):
        df1 = k - 1
        for n in range(2, 300):
            df2  = k * (n - 1)
            lam  = eta2 / (1 - eta2) * k * n
            fcrit = _ncf.ppf(1 - alpha, df1, df2, 0)
            if 1 - _ncf.cdf(fcrit, df1, df2, lam) >= target:
                return n
        return 300
    _eta2_med_low  = 0.06   # lower bound of medium-effect range in S4 text
    _eta2_med_high = 0.14   # upper bound
    _req_n_low  = _required_n(_eta2_med_high)   # fewer replicates needed for larger η²
    _req_n_high = _required_n(_eta2_med_low)    # more replicates needed for smaller η²
    bag.add("power", "med_eta_two_low",    _eta2_med_low,  _fmt(_eta2_med_low, 2),
            "lower bound of medium-effect η² range cited in S4")
    bag.add("power", "med_eta_two_high",   _eta2_med_high, _fmt(_eta2_med_high, 2),
            "upper bound of medium-effect η² range cited in S4")
    bag.add("power", "required_n_med_low", _req_n_low,  str(_req_n_low),
            f"required N per group at 80% power for η²={_eta2_med_high} (upper bound → fewer n)")
    bag.add("power", "required_n_med_high",_req_n_high, str(_req_n_high),
            f"required N per group at 80% power for η²={_eta2_med_low} (lower bound → more n)")

    # =========================================================================
    # permutation space (nb19 — Methods characterisation)
    # =========================================================================
    ps = _load(ANALYSIS / "permutation_space.pkl")

    bag.add("perm_space", "log_configs_mean", ps["log10_configs_mean"],
            _fmt(ps["log10_configs_mean"], 1), "mean log10(distinct configs) across 54 agents")
    bag.add("perm_space", "log_configs_min",  ps["log10_configs_min"],
            _fmt(ps["log10_configs_min"], 1),  "min log10(distinct configs)")
    bag.add("perm_space", "log_configs_max",  ps["log10_configs_max"],
            _fmt(ps["log10_configs_max"], 1),  "max log10(distinct configs)")
    bag.add("perm_space", "frac_nontrivial_pct",
            ps["frac_nontrivial_mean"] * 100,
            f"{ps['frac_nontrivial_mean']*100:.0f}",
            "% source-pathway pairs with >1 valid target arrangement")

    # per-block stats — mean is per-agent mean; std is std of per-agent means (n=54)
    for _bname in ["SI", "SM", "II", "IM", "MI", "MM"]:
        _per_agent = [float(np.mean(r["block_choices"][_bname]))
                      for r in ps["per_agent"]]
        _arr = np.array(_per_agent)
        _key = _bname.lower()
        bag.add("perm_space", f"choices_{_key}_mean",
                float(_arr.mean()), _fmt(_arr.mean(), 1),
                f"mean choices per source in {_bname} block (per-agent mean, n=54)")
        bag.add("perm_space", f"choices_{_key}_std",
                float(_arr.std()),  _fmt(_arr.std(), 1),
                f"std of per-agent mean choices, {_bname} block")

    # =========================================================================
    # per-mouse table
    # =========================================================================
    with open(ANALYSIS / "per_mouse_summary.csv") as f:
        pms_rows = list(csv.DictReader(f))
    fitnesses = [float(r["fitness"]) for r in pms_rows]
    n_conn    = [float(r["n_connections"]) for r in pms_rows]

    bag.add("per_mouse", "fitness_mean",      float(np.mean(fitnesses)),  _fmt(np.mean(fitnesses), 3))
    bag.add("per_mouse", "fitness_std",       float(np.std(fitnesses,ddof=1)), _fmt(np.std(fitnesses,ddof=1), 3))
    bag.add("per_mouse", "fitness_min",       float(np.min(fitnesses)),   _fmt(np.min(fitnesses), 3))
    bag.add("per_mouse", "fitness_max",       float(np.max(fitnesses)),   _fmt(np.max(fitnesses), 3))
    bag.add("per_mouse", "n_connections_mean",float(np.mean(n_conn)),     _fmt(np.mean(n_conn), 2))
    bag.add("per_mouse", "density_mean",
            float(np.mean([float(r["density"]) for r in pms_rows])),
            _fmt(np.mean([float(r["density"]) for r in pms_rows]), 3))

    # =========================================================================
    # derived quantities
    # =========================================================================
    cohen_dz  = float(p3c["cohen_dz"]); mde_dz = float(p3c["mde_d"])
    bag.add("derived", "cohen_dz_over_mde",
            cohen_dz / mde_dz, _fmt(cohen_dz / mde_dz, 2),
            "phase3c.cohen_dz / phase3c.mde_d")
    rnd_mean_r = float(a5["mean_ratio"]); rnd_std_r = float(a5["std_ratio"])
    bag.add("derived", "evolved_z_vs_random_null",
            (rnd_mean_r - float(spec_ratio)) / rnd_std_r,
            _fmt((rnd_mean_r - float(spec_ratio)) / rnd_std_r, 2),
            "(random mean_ratio - evolved spec_ratio) / random std_ratio")

    # =========================================================================
    # experimental constants
    # =========================================================================
    bag.add("setup", "n_mice",             len(MICE),      str(len(MICE)))
    bag.add("setup", "n_reps",             6,              "6",  "evolution replicates per mouse")
    bag.add("setup", "n_runs",             9 * 6,          "54", "total per-mouse evolution runs")
    bag.add("setup", "n_generalist_reps",  g["N_REPS_G"],  str(g["N_REPS_G"]))


# ---------------------------------------------------------------------------
# Block 2 — Degeneracy analyses A1-A6 (explicit \statDeg... macro names)
# ---------------------------------------------------------------------------

def _parse_summary(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    current = None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        m = re.match(r"^(A\d+)$", stripped)
        if m:
            current = m.group(1); continue
        m = re.match(r"^(.+?):\s+(.+)$", stripped)
        if m and current:
            result[f"{current}_{m.group(1)}"] = m.group(2).strip()
    return result


def _build_degeneracy() -> tuple[list[str], list[str]]:
    """Return (tex_lines, txt_lines) for Block 2 degeneracy macros."""

    tex: list[str] = []
    txt: list[str] = []

    def emit(name: str, value: str, comment: str = "") -> None:
        cmt = f"  % {comment}" if comment else ""
        tex.append(f"\\newcommand{{\\{name}}}{{{value}}}{cmt}")
        txt.append(f"{name} = {value}  # {comment}")

    summary = _parse_summary(DEGENERACY / "consolidated_summary.txt")
    A1 = _load(DEGENERACY / "A1_results.pkl")
    A2 = _load(DEGENERACY / "A2_results.pkl")
    A3 = _load(DEGENERACY / "A3_results.pkl")
    A5 = _load(DEGENERACY / "A5_results.pkl")
    A6 = _load(DEGENERACY / "A6_results.pkl")

    # A3: primary = mouse-identity MI/NMI (update_A3_mouse_identity.py writes these)
    #     secondary = fitness-quintile MI/NMI (original nb15; kept for supplementary)
    A3_mi_topo  = float(A3["Topology"]["MI_mouse"])
    A3_nmi_topo = float(A3["Topology"]["NMI_mouse"])
    A3_mi_mag   = float(A3["Magnitude"]["MI_mouse"])
    A3_nmi_mag  = float(A3["Magnitude"]["NMI_mouse"])
    A3_mi_sign  = float(A3["Sign (E/I)"]["MI_mouse"])
    A3_nmi_sign = float(A3["Sign (E/I)"]["NMI_mouse"])
    A3_nmi_max  = max(A3_nmi_topo, A3_nmi_mag, A3_nmi_sign)

    # A3 fitness-quintile NMI (companion analysis; shown in supp_mi_clustering)
    A3_fit_nmi_topo = float(A3["Topology"]["NMI"])
    A3_fit_nmi_mag  = float(A3["Magnitude"]["NMI"])
    A3_fit_nmi_sign = float(A3["Sign (E/I)"]["NMI"])
    A3_fit_nmi_max  = max(A3_fit_nmi_topo, A3_fit_nmi_mag, A3_fit_nmi_sign)

    # A3 AMI sensitivity across k=3-9 (mouse-identity target)
    A3_ami_topo_min  = float(A3["Topology"]["ami_mouse_min"])
    A3_ami_topo_max  = float(A3["Topology"]["ami_mouse_max"])
    A3_ami_mag_min   = float(A3["Magnitude"]["ami_mouse_min"])
    A3_ami_mag_max   = float(A3["Magnitude"]["ami_mouse_max"])
    A3_ami_sign_min  = float(A3["Sign (E/I)"]["ami_mouse_min"])
    A3_ami_sign_max  = float(A3["Sign (E/I)"]["ami_mouse_max"])

    # A4: read from summary (pkl truncated)
    A4_mw_p           = float(summary["A4_mw_p"])
    A4_mantel_rho     = float(summary["A4_mantel_rho"])
    A4_mantel_p_perm  = float(summary["A4_mantel_p_permutation"])

    # derived A1
    A1_rho_all    = float(A1["rho_all"]);    A1_p_all    = float(A1["p_all"])
    A1_rho_within = float(A1["rho_within"]); A1_p_within = float(A1["p_within"])
    A1_rho_between= float(A1["rho_between"]);A1_p_between= float(A1["p_between"])
    A1_mw_beh_p   = float(A1["mw_beh_p"])
    A1_n_pairs    = int(np.array(A1["topo_dists"]).shape[0])

    _topo_dists  = np.array(A1["topo_dists"])
    _pair_types  = np.array(A1["pair_types"])
    _topo_within  = _topo_dists[_pair_types == "within"]
    _topo_between = _topo_dists[_pair_types == "between"]
    A1_topo_within_mean  = float(np.mean(_topo_within))
    A1_topo_between_mean = float(np.mean(_topo_between))
    from scipy.stats import mannwhitneyu as _mw_a1
    _, A1_mw_topo_p = _mw_a1(_topo_within, _topo_between, alternative="two-sided")

    # derived A2
    A2_rho_topo         = float(A2["rho_topo"]);         A2_p_topo         = float(A2["p_topo"])
    A2_rho_mag          = float(A2["rho_mag"]);           A2_p_mag          = float(A2["p_mag"])
    A2_partial_rho_topo = float(A2["partial_rho_topo_given_mag"])
    A2_partial_p_topo   = float(A2["partial_p_topo_given_mag"])
    A2_partial_rho_mag  = float(A2["partial_rho_mag_given_topo"])
    A2_partial_p_mag    = float(A2["partial_p_mag_given_topo"])
    A2_n_pairs          = len(A2["df"])

    # derived A5
    A5_rho_rand    = float(A5["rho_rand"]);           A5_p_rand    = float(A5["p_rand"])
    A5_mw_within_p = float(A5["mw_within_between_p"])

    # derived A6
    A6_mw_p_topo     = float(A6["mw_p_gen_spec_topo"])
    A6_gen_cost_pct  = np.array(A6["gen_cost_pct"])
    A6_mean_cost_pct = float(np.mean(A6_gen_cost_pct))
    A6_min_cost_pct  = float(np.min(A6_gen_cost_pct))
    A6_max_cost_pct  = float(np.max(A6_gen_cost_pct))
    A6_hardest_mouse = MICE[int(np.argmax(A6_gen_cost_pct))]
    A6_easiest_mouse = MICE[int(np.argmin(A6_gen_cost_pct))]

    A6_spec_sens_var = np.array(A6["spec_sens_var"])
    A6_gen_sens_var  = np.array(A6["gen_sens_var_norm"])  # normalized — apples-to-apples
    A6_spec_var_mean = float(np.mean(A6_spec_sens_var))
    A6_gen_var_mean  = float(np.mean(A6_gen_sens_var))
    A6_var_ratio     = A6_spec_var_mean / A6_gen_var_mean if A6_gen_var_mean > 1e-9 else float("nan")
    # Inferential test for the variance ratio (per-neuron, n=14 each; one-sided spec > gen).
    # The ratio is a ratio of means; this MW supplies the significance the figure visually implies.
    from scipy.stats import mannwhitneyu as _mw_a6
    _, A6_var_mw_p = _mw_a6(A6_spec_sens_var, A6_gen_sens_var, alternative="greater")
    A6_var_mw_p = float(A6_var_mw_p)

    A6_gen_topo_sims  = np.array(A6["gen_topo_sims"])
    A6_spec_topo_sims = np.array(A6["spec_topo_sims"])
    A6_gen_topo_mean  = float(np.mean(A6_gen_topo_sims))
    A6_spec_topo_mean = float(np.mean(A6_spec_topo_sims))

    # ------------------------------------------------------------------
    # Referee-audit robustness statistics for the sensitivity-commitment
    # (Claim 24) and generalist-cost (Claim 23) results. Formulas ported
    # verbatim from review/scripts/r1_*.py and r3_*.py so the macros are
    # regenerable and verifiable (seed=1, deterministic bootstrap).
    # ------------------------------------------------------------------
    SS6 = _load(ANALYSIS / "source_sensitivity_results.pkl")
    sens_norm6 = np.asarray(SS6["sensitivity_matrix"], float)      # (54,14)
    labels6    = np.asarray(SS6["labels"])
    gen_norm6  = np.asarray(A6["gen_sensitivity_norm"], float)     # (6,14)

    # (a) Principled within-mouse-pooled variance ratio (text-matches-number fix).
    A6_var_within_per = np.stack(
        [sens_norm6[labels6 == m].var(axis=0, ddof=1) for m in MICE])  # (9,14)
    A6_sv_within = A6_var_within_per.mean(axis=0)
    A6_gv_ddof1  = gen_norm6.var(axis=0, ddof=1)
    A6_var_ratio_within = float(A6_sv_within.mean() / A6_gv_ddof1.mean())

    # (b) Mouse-level primary test (mouse = independent unit; fixes pseudoreplication).
    A6_mouse_scalars = np.array(
        [sens_norm6[labels6 == m].var(axis=0, ddof=1).mean() for m in MICE])
    A6_gen_scalar    = float(A6_gv_ddof1.mean())
    A6_mouse_n_pos   = int((A6_mouse_scalars > A6_gen_scalar).sum())
    A6_mouse_wilcox_p = float(
        _scipy_stats.wilcoxon(A6_mouse_scalars - A6_gen_scalar, alternative="greater").pvalue)

    # (c) Hierarchical bootstrap CI on the ratio (n=6 generalists; magnitude uncertainty).
    _rng6 = np.random.default_rng(1)
    _by_mouse6 = {m: sens_norm6[labels6 == m] for m in MICE}
    _boot = np.empty(10000)
    for _b in range(10000):
        _drawn = _rng6.choice(MICE, size=9, replace=True)
        _svs = []
        for _m in _drawn:
            _sub = _by_mouse6[_m]
            _svs.append(_sub[_rng6.integers(0, _sub.shape[0], _sub.shape[0])].var(0, ddof=1))
        _gmean = gen_norm6[_rng6.integers(0, 6, 6)].var(0, ddof=1).mean()
        _boot[_b] = np.mean(np.stack(_svs), 0).mean() / _gmean if _gmean > 1e-12 else np.nan
    _boot = _boot[np.isfinite(_boot)]
    A6_var_boot_median = float(np.median(_boot))
    A6_var_boot_ci_lo, A6_var_boot_ci_hi = (float(x) for x in np.percentile(_boot, [2.5, 97.5]))

    # (d) Generalist cost: ratio-of-means (stable) vs the published mean-of-ratios.
    A6_gen_mean_per_mouse = np.asarray(A6["gen_mean_per_mouse"], float)
    A6_spec_ref_a6 = A6_gen_mean_per_mouse / (1.0 + A6_gen_cost_pct / 100.0)
    A6_cost_ratio_of_means = float(
        (A6_gen_mean_per_mouse.mean() - A6_spec_ref_a6.mean()) / A6_spec_ref_a6.mean() * 100.0)

    # ---- emit ----
    tex.append("% -- A1: Topology-behavior flatness --")
    emit("statDegAoneRhoAll",     _fmt(A1_rho_all, 3),     "Spearman rho (topo vs beh dist), all pairs")
    emit("statDegAonePAll",       _fmt_p(A1_p_all),         "p-value for rho_all")
    emit("statDegAoneRhoWithin",  _fmt(A1_rho_within, 3),  "rho within-mouse pairs")
    emit("statDegAonePWithin",    _fmt_p(A1_p_within),      "p within-mouse")
    emit("statDegAoneRhoBetween", _fmt(A1_rho_between, 3), "rho between-mouse pairs")
    emit("statDegAonePBetween",   _fmt_p(A1_p_between),     "p between-mouse")
    emit("statDegAoneMwBehP",           _fmt_p(A1_mw_beh_p),           "MW p: beh dist within < between")
    emit("statDegAoneNPairs",           str(A1_n_pairs),               "number of agent pairs in A1 correlation")
    emit("statDegAoneTopoDistWithinMean",  _fmt(A1_topo_within_mean, 3),  "mean within-mouse Jaccard topo distance")
    emit("statDegAoneTopoDistBetweenMean", _fmt(A1_topo_between_mean, 3), "mean between-mouse Jaccard topo distance")
    emit("statDegAoneMwTopoP",          _fmt_p(A1_mw_topo_p),          "MW p: within vs between topo dist (two-sided)")

    tex.append("")
    tex.append("% -- A2: Within-mouse structural variation vs fitness --")
    emit("statDegAtwoRhoTopo",        _fmt(A2_rho_topo, 3),        "Spearman rho: topo dist vs delta_fit (within-mouse)")
    emit("statDegAtwoPTopo",          _fmt_p(A2_p_topo),            "p for rho_topo")
    emit("statDegAtwoRhoMag",         _fmt(A2_rho_mag, 3),          "Spearman rho: mag dist vs delta_fit")
    emit("statDegAtwoPMag",           _fmt_p(A2_p_mag),              "p for rho_mag")
    emit("statDegAtwoPartialRhoTopo", _fmt(A2_partial_rho_topo, 3), "partial rho: topo given mag")
    emit("statDegAtwoPartialPTopo",   _fmt_p(A2_partial_p_topo),    "partial p: topo given mag")
    emit("statDegAtwoPartialRhoMag",  _fmt(A2_partial_rho_mag, 3),  "partial rho: mag given topo")
    emit("statDegAtwoPartialPMag",    _fmt_p(A2_partial_p_mag),     "partial p: mag given topo")
    emit("statDegAtwoNPairs",         str(A2_n_pairs),               "number of within-mouse replicate pairs")

    tex.append("")
    tex.append("% -- A3: Axis-specific MI vs mouse identity (NMI at k=5; AMI k=3-9) --")
    emit("statDegAthreeMiTopo",  _fmt(A3_mi_topo, 4),  "MI: topology axis vs mouse identity")
    emit("statDegAthreeNmiTopo", _fmt(A3_nmi_topo, 4), "NMI: topology axis vs mouse identity (k=5)")
    emit("statDegAthreeMiMag",   _fmt(A3_mi_mag, 4),   "MI: magnitude axis vs mouse identity")
    emit("statDegAthreeNmiMag",  _fmt(A3_nmi_mag, 4),  "NMI: magnitude axis vs mouse identity (k=5)")
    emit("statDegAthreeMiSign",  _fmt(A3_mi_sign, 4),  "MI: sign (E/I) axis vs mouse identity")
    emit("statDegAthreeNmiSign", _fmt(A3_nmi_sign, 4), "NMI: sign axis vs mouse identity (k=5)")
    emit("statDegAthreeNmiMax",  _fmt(A3_nmi_max, 4),  "maximum NMI (mouse identity) across all axes")
    emit("statDegAthreeAmiTopoMin",  _fmt(A3_ami_topo_min, 2),  "AMI topology vs mouse identity: min across k=3-9")
    emit("statDegAthreeAmiTopoMax",  _fmt(A3_ami_topo_max, 2),  "AMI topology vs mouse identity: max across k=3-9")
    emit("statDegAthreeAmiMagMin",   _fmt(A3_ami_mag_min, 2),   "AMI magnitude vs mouse identity: min across k=3-9")
    emit("statDegAthreeAmiMagMax",   _fmt(A3_ami_mag_max, 2),   "AMI magnitude vs mouse identity: max across k=3-9")
    emit("statDegAthreeAmiSignMin",  _fmt(A3_ami_sign_min, 2),  "AMI sign (E/I) vs mouse identity: min across k=3-9")
    emit("statDegAthreeAmiSignMax",  _fmt(A3_ami_sign_max, 2),  "AMI sign (E/I) vs mouse identity: max across k=3-9")
    # Fitness-quintile companion (Supp S13 figure only)
    emit("statDegAthreeFitNmiTopo", _fmt(A3_fit_nmi_topo, 4), "NMI: topology vs fitness quintile (companion)")
    emit("statDegAthreeFitNmiMag",  _fmt(A3_fit_nmi_mag, 4),  "NMI: magnitude vs fitness quintile (companion)")
    emit("statDegAthreeFitNmiSign", _fmt(A3_fit_nmi_sign, 4), "NMI: sign vs fitness quintile (companion)")
    emit("statDegAthreeFitNmiMax",  _fmt(A3_fit_nmi_max, 4),  "max NMI vs fitness quintile (companion)")

    tex.append("")
    tex.append("% -- A4: Sensitivity RSA (Supplementary) --")
    emit("statDegAfourMwP",         _fmt_p(A4_mw_p),          "MW p: sensitivity within vs between")
    emit("statDegAfourMantelRho",   _fmt(A4_mantel_rho, 3),   "Mantel rho: sensitivity RSA")
    emit("statDegAfourMantelPPerm", _fmt_p(A4_mantel_p_perm), "Mantel p permutation")

    tex.append("")
    tex.append("% -- A5: Random agent null --")
    emit("statDegAfiveRhoRand",   _fmt(A5_rho_rand, 3),   "Spearman rho: topo vs beh dist, random agents")
    emit("statDegAfivePRand",     _fmt_p(A5_p_rand),       "p for rho_rand")
    emit("statDegAfiveMwWithinP", _fmt_p(A5_mw_within_p), "MW p: within vs between beh dist, random agents")

    tex.append("")
    tex.append("% -- A6: Generalist vs specialist --")
    emit("statDegAsixMwTopoP",         _fmt_p(A6_mw_p_topo),       "MW p: gen vs spec topology similarity")
    emit("statDegAsixGenTopoMean",     _fmt(A6_gen_topo_mean, 3),  "mean topology cosine similarity, generalists")
    emit("statDegAsixSpecTopoMean",    _fmt(A6_spec_topo_mean, 3), "mean topology cosine similarity, specialists")
    emit("statDegAsixMeanCostPct",     _fmt(A6_mean_cost_pct, 1),  "mean-of-ratios generalist fitness cost pct (per-mouse mean)")
    emit("statDegAsixCostRatioOfMeans", _fmt(A6_cost_ratio_of_means, 1),
         "ratio-of-means generalist fitness cost pct (stable summary; primary)")
    emit("statDegAsixMinCostPct",      _fmt(A6_min_cost_pct, 1),   "min per-mouse generalist fitness cost pct")
    emit("statDegAsixMaxCostPct",      _fmt(A6_max_cost_pct, 1),   "max per-mouse generalist fitness cost pct")
    emit("statDegAsixHardestMouse",    A6_hardest_mouse,             "mouse with highest generalist fitness cost")
    emit("statDegAsixEasiestMouse",    A6_easiest_mouse,             "mouse with lowest generalist fitness cost")
    emit("statDegAsixSpecSensVarMean", _fmt(A6_spec_var_mean, 3),  "mean sensitivity variance per neuron, specialists")
    emit("statDegAsixGenSensVarMean",  _fmt(A6_gen_var_mean, 3),   "mean sensitivity variance per neuron, generalists (normalized)")
    if np.isfinite(A6_var_ratio):
        emit("statDegAsixVarRatio", _fmt(A6_var_ratio, 1), "sensitivity variance ratio: specialist / generalist (normalized)")
    else:
        emit("statDegAsixVarRatio", ">9999", "sensitivity variance ratio (gen var near zero)")
    emit("statDegAsixVarMwP", _fmt_p(A6_var_mw_p),
         "MW p (one-sided spec>gen) for per-neuron sensitivity variance, n=14 each")
    # Referee-audit robustness macros (ported from review r1)
    emit("statDegAsixVarRatioWithin", _fmt(A6_var_ratio_within, 1),
         "sensitivity variance ratio, within-mouse-pooled ddof=1 (principled; matches Methods)")
    emit("statDegAsixVarBootMedian", _fmt(A6_var_boot_median, 1),
         "hierarchical-bootstrap median sensitivity variance ratio (seed=1)")
    emit("statDegAsixVarBootCiLo", _fmt(A6_var_boot_ci_lo, 2),
         "hierarchical-bootstrap 2.5pct sensitivity variance ratio")
    emit("statDegAsixVarBootCiHi", _fmt(A6_var_boot_ci_hi, 2),
         "hierarchical-bootstrap 97.5pct sensitivity variance ratio")
    emit("statDegAsixMouseNPos", str(A6_mouse_n_pos),
         "n mice (of 9) with sensitivity variance > generalist (mouse-level unit)")
    emit("statDegAsixMouseWilcoxP", _fmt_p(A6_mouse_wilcox_p),
         "one-sample Wilcoxon p, mouse-level sensitivity commitment (primary statistic)")

    return tex, txt


# ---------------------------------------------------------------------------
# Block 3 — claim 28: within-mouse sensitivity convergence (nb18)
# ---------------------------------------------------------------------------

def _build_claim28() -> tuple[list[str], list[str]]:
    """Return (tex_lines, txt_lines) for Block 3: sensitivity convergence macros.

    Claim 28: within-mouse sensitivity profiles are no more similar than
    between-mouse profiles — selection builds commitment strength, not direction.

    Source: sens_commitment_evolution.pkl (nb18), spec_sens_all[-1] (gen 150).
    Pairwise cosine similarities: 135 within-mouse pairs vs 1296 between-mouse.
    """
    tex: list[str] = []
    txt: list[str] = []

    def emit(name: str, value: str, comment: str = "") -> None:
        cmt = f"  % {comment}" if comment else ""
        tex.append(f"\\newcommand{{\\{name}}}{{{value}}}{cmt}")
        txt.append(f"{name} = {value}  # {comment}")

    pkl_path = DEGENERACY / "sens_commitment_evolution.pkl"
    if not pkl_path.exists():
        print(f"  [WARN] {pkl_path} not found — skipping claim 28 macros", file=sys.stderr)
        return tex, txt

    data = _load(pkl_path)

    # Sensitivity vectors at gen 150 (last checkpoint): shape (54, 14)
    sens_g150 = np.array(data["spec_sens_all"][-1])  # (54, 14)

    # Agent-to-mouse labels: 9 mice × 6 reps in MICE order
    N_REP = 6
    labels = np.array([mi for mi in range(len(MICE)) for _ in range(N_REP)])

    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    within_sims: list[float] = []
    between_sims: list[float] = []
    n = len(sens_g150)
    for i, j in combinations(range(n), 2):
        sim = cosine_sim(sens_g150[i], sens_g150[j])
        if labels[i] == labels[j]:
            within_sims.append(sim)
        else:
            between_sims.append(sim)

    within_arr  = np.array(within_sims)   # 135 pairs
    between_arr = np.array(between_sims)  # 1296 pairs

    within_mean  = float(within_arr.mean())
    between_mean = float(between_arr.mean())

    _, p_mw = _scipy_stats.mannwhitneyu(within_arr, between_arr, alternative="two-sided")

    # Pseudoreplication-aware label-permutation test (ported from review r2): the
    # 135/1296 agent-pairs share agents and are not independent, so the MW p is
    # anticonservative. Permute mouse labels and recompute the within-between mean gap.
    iu, ju = np.triu_indices(n, 1)
    cvals = np.array([cosine_sim(sens_g150[i], sens_g150[j]) for i, j in zip(iu, ju)])
    same_obs = labels[iu] == labels[ju]
    obs_gap = cvals[same_obs].mean() - cvals[~same_obs].mean()
    _perm_rng = np.random.default_rng(1)
    _null = np.empty(10000)
    for _k in range(10000):
        _perm = _perm_rng.permutation(labels)
        _s = _perm[iu] == _perm[ju]
        _null[_k] = cvals[_s].mean() - cvals[~_s].mean()
    p_perm = float(np.mean(np.abs(_null) >= abs(obs_gap)))

    tex.append("")
    tex.append("% -- Claim 28: sensitivity convergence (nb18, gen 150 pairwise) --")
    emit("statClaimTwentyEightWithinSim",  _fmt(within_mean, 2),
         "within-mouse pairwise cosine sim at gen 150 (135 pairs)")
    emit("statClaimTwentyEightBetweenSim", _fmt(between_mean, 2),
         "between-mouse pairwise cosine sim at gen 150 (1296 pairs)")
    emit("statClaimTwentyEightMwP",        _fmt_p(p_mw),
         "MW p: within < between cosine sim (two-sided; anticonservative, pseudoreplicated)")
    emit("statClaimTwentyEightPermP",      _fmt_p(p_perm),
         "label-permutation p (pseudoreplication-aware); Claim-28 gap n.s. under proper test")

    return tex, txt


# ---------------------------------------------------------------------------
# Block 4 — Activity embedding macros (§2.7, nb20)
# ---------------------------------------------------------------------------

ACT_EMB_DIR = ANALYSIS / "activity_embeddings"


def _build_act_emb() -> tuple[list[str], list[str]]:
    """Return (tex_lines, txt_lines) for Block 4: activity embedding macros.

    Sources:
      ACT_EMB_DIR/B_results.pkl  — B2 motor ctrl, B3 dimensionality,
                                    B4 activity RSA, B5 generalist RSA
      ACT_EMB_DIR/D_results.pkl  — D1–D3 Mantel tests, D4 attractor landscape
    """
    tex: list[str] = []
    txt: list[str] = []

    def emit(name: str, value: str, comment: str = "") -> None:
        cmt = f"  % {comment}" if comment else ""
        tex.append(f"\\newcommand{{\\{name}}}{{{value}}}{cmt}")
        txt.append(f"{name} = {value}  # {comment}")

    b_path = ACT_EMB_DIR / "B_results.pkl"
    d_path = ACT_EMB_DIR / "D_results.pkl"

    if not b_path.exists():
        print(f"  [WARN] {b_path} not found — skipping act-emb macros", file=sys.stderr)
        return tex, txt
    if not d_path.exists():
        print(f"  [WARN] {d_path} not found — skipping D-result macros", file=sys.stderr)

    B = _load(b_path)
    D = _load(d_path) if d_path.exists() else {}

    tex.append("")
    tex.append("% -- Block 4: Activity embedding macros (§2.7, nb20) --")

    # ── B2: motor positive control ───────────────────────────────────────────
    B2 = B["B2"]
    motor_dists      = np.asarray(B2["motor_dists"])
    inter_dists_mean = np.asarray(B2["inter_dists_mean"])

    tex.append("% B2 motor separation positive control")
    emit("statActEmbMotorDist",    _fmt(motor_dists.mean(), 3),
         "mean speed-vs-turn motor distance in PC1-2 (54 agents)")
    emit("statActEmbMotorDistStd", _fmt(motor_dists.std(), 3),
         "SD motor distance")
    emit("statActEmbInterDist",    _fmt(inter_dists_mean.mean(), 3),
         "mean inter-interneuron distance in PC1-2")
    emit("statActEmbInterDistStd", _fmt(inter_dists_mean.std(), 3),
         "SD inter-interneuron distance")

    # ── B3: manifold dimensionality ──────────────────────────────────────────
    B3      = B["B3"]
    eff_dim = np.asarray(B3["eff_dim_90"])
    pr_all  = np.asarray(B3["participation_ratio"])

    tex.append("% B3 manifold dimensionality")
    emit("statActEmbEffDim",    _fmt(eff_dim.mean(), 2),
         "mean effective dimensionality at 90pct var threshold")
    emit("statActEmbEffDimStd", _fmt(eff_dim.std(), 2),
         "SD effective dimensionality")
    emit("statActEmbPR",        _fmt(pr_all.mean(), 2),
         "mean participation ratio")
    emit("statActEmbPRStd",     _fmt(pr_all.std(), 2),
         "SD participation ratio")

    # ── B4: activity RSA (Procrustes) ────────────────────────────────────────
    # De-patched 2026-06-16: B4["within_proc"]/["between_proc"] already store
    # Procrustes DISTANCES (verified range 1.09–2.44, matching D_procrustes).
    # The prior sqrt(2N−2·scale) rescale double-converted and is removed.
    _N_PC = 6
    B4          = B["B4"]
    within_proc  = np.asarray(B4["within_proc"])
    between_proc = np.asarray(B4["between_proc"])
    p_proc       = float(B4["p_proc"])
    robustness   = B4.get("robustness", {})

    tex.append("% B4 activity RSA — Procrustes within vs between")
    emit("statActEmbWithinProc",    _fmt(within_proc.mean(), 3),
         "mean within-mouse Procrustes distance (135 pairs)")
    emit("statActEmbWithinProcStd", _fmt(within_proc.std(), 3),
         "SD within-mouse Procrustes")
    emit("statActEmbBetweenProc",    _fmt(between_proc.mean(), 3),
         "mean between-mouse Procrustes distance (1296 pairs)")
    emit("statActEmbBetweenProcStd", _fmt(between_proc.std(), 3),
         "SD between-mouse Procrustes")
    emit("statActEmbRsaMwP",         _fmt_p(p_proc),
         "MW p: within vs between Procrustes")

    # robustness: min/max p_proc across full/steady/mean modes
    if robustness:
        robust_ps = [float(v["p_proc"]) for v in robustness.values()
                     if "p_proc" in v]
        if robust_ps:
            emit("statActEmbRobustPLo", _fmt(min(robust_ps), 2),
                 "min MW p across activity matrix modes (robustness)")
            emit("statActEmbRobustPHi", _fmt(max(robust_ps), 2),
                 "max MW p across activity matrix modes (robustness)")

    # ── 6-metric embedding robustness (RSA / CKA / Cov-Frob / Proc-raw / LogEuc) ──
    # Recomputed from B_results.pkl by scripts/embedding_robustness.py.
    # (Run `python scripts/embedding_robustness.py` to (re)generate the pkl.)
    _rob_path = ACT_EMB_DIR / "embedding_robustness.pkl"
    _rob = _load(_rob_path) if _rob_path.exists() else None
    if _rob is None:
        print("  [warn] embedding_robustness.pkl missing — run "
              "scripts/embedding_robustness.py; skipping 5 robustness macros.")
    else:
        emit("statActEmbRsaMwPVal",     _fmt(_rob["RSA"]["mw_p"], 3),
             "RSA on RDMs MW p within vs between")
        emit("statActEmbCkaMwPVal",     _fmt(_rob["CKA"]["mw_p"], 3),
             "Linear CKA MW p within vs between")
        emit("statActEmbCovFrobMwPVal", _fmt(_rob["Cov-Frob"]["mw_p"], 3),
             "Cov-Frobenius MW p within vs between")
        emit("statActEmbProcRawMwPVal", _fmt(_rob["Proc-raw"]["mw_p"], 3),
             "Proc-raw no-PCA MW p within vs between")
        emit("statActEmbLogEucMwPVal",  _fmt(_rob["LogEuc-SPD"]["mw_p"], 3),
             "Log-Euclidean SPD MW p within vs between")

    # ── B5: generalist activity RSA ──────────────────────────────────────────
    B5              = B["B5"]
    gen_dists_proc  = np.asarray(B5["gen_dists_proc"])   # de-patched: already distances
    p_gen_vs_within = float(B5["p_gen_vs_within"])

    tex.append("% B5 generalist activity RSA")
    emit("statActEmbGenProc",        _fmt(gen_dists_proc.mean(), 3),
         "mean generalist Procrustes distance (15 pairs)")
    emit("statActEmbGenProcStd",     _fmt(gen_dists_proc.std(), 3),
         "SD generalist Procrustes")
    emit("statActEmbGenVsWithinP",   _fmt_p(p_gen_vs_within),
         "MW p: generalist vs specialist within-mouse Procrustes")

    # ── D1–D3: Mantel dissociation tests ────────────────────────────────────
    # NOTE: macro names use spelled-out labels (Beh/Topo/Sens) to avoid digits
    # immediately after letters — LaTeX command names are letters-only, so
    # \statActEmbD1Rho would parse as \statActEmbD + 1Rho and break.
    if D:
        tex.append("% D1-D3 Mantel dissociation tests")
        for key, macro_rho, macro_p, label in [
            ("D1", "statActEmbBehMantelRho", "statActEmbBehMantelP",
             "Mantel rho: representational vs behavioral distance"),
            ("D2", "statActEmbTopoMantelRho", "statActEmbTopoMantelP",
             "Mantel rho: representational vs topological distance"),
            ("D3", "statActEmbSensMantelRho", "statActEmbSensMantelP",
             "Mantel rho: representational vs sensitivity distance"),
        ]:
            if key not in D:
                continue
            # De-patched 2026-06-16: D_results.pkl rho already computed on
            # distances; the prior `-rho` flip inverted a correct value.
            rho   = float(D[key]["rho"])
            p_val = float(D[key]["p_perm"])
            emit(macro_rho, _fmt(rho, 3), label)
            emit(macro_p,   _fmt_p(p_val), label.replace("rho", "p"))

        # ── D4: attractor landscape ──────────────────────────────────────────
        if "D4" in D:
            D4           = D["D4"]
            within_attr  = np.asarray(D4["within_attr"])
            between_attr = np.asarray(D4["between_attr"])
            p_mw_d4      = float(D4["p_mw"])

            tex.append("% D4 attractor landscape (used in §2.8)")
            emit("statActEmbAttrWithinWass",    _fmt(within_attr.mean(), 3),
                 "mean within-mouse sliced Wasserstein (attractor)")
            emit("statActEmbAttrWithinWassStd", _fmt(within_attr.std(), 3),
                 "SD within-mouse attractor distance")
            emit("statActEmbAttrBetweenWass",    _fmt(between_attr.mean(), 3),
                 "mean between-mouse sliced Wasserstein (attractor)")
            emit("statActEmbAttrBetweenWassStd", _fmt(between_attr.std(), 3),
                 "SD between-mouse attractor distance")
            emit("statActEmbAttrMwP",            _fmt_p(p_mw_d4),
                 "MW p: within vs between attractor distance")

    return tex, txt


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# Block 5 — Circuit comparison macros (§2.8, compute_circuits_results.py)
# ---------------------------------------------------------------------------


def _build_circuits() -> tuple[list[str], list[str]]:
    """Return (tex_lines, txt_lines) for Block 5: §2.8 circuit comparison macros.

    Source: analysis/activity_embeddings/circuits_results.pkl
    Generated by: scripts/compute_circuits_results.py
    """
    tex: list[str] = []
    txt: list[str] = []

    circ_path = ACT_EMB_DIR / "circuits_results.pkl"
    if not circ_path.exists():
        print(f"  [WARN] {circ_path} not found — skipping circuit macros", file=sys.stderr)
        print(f"         Run: python scripts/compute_circuits_results.py", file=sys.stderr)
        return tex, txt

    c = _load(circ_path)

    def emit(name: str, value, comment: str = "") -> None:
        val_str = str(value)
        cmt = f"  % {comment}" if comment else ""
        tex.append(f"\\newcommand{{\\{name}}}{{{val_str}}}{cmt}")
        txt.append(f"{name} = {val_str}  # {comment}")

    tex.append("")
    tex.append("% -- Block 5: Circuit comparison macros (§2.8) --")

    # Most similar within-mouse pair
    tex.append("% Similar pair (min Jaccard within-mouse)")
    emit("statCircSimMouse",        c["sim_mouse"],
         "mouse of most-similar within-mouse pair")
    emit("statCircSimJaccard",      _fmt(c["sim_jaccard"], 3),
         "Jaccard distance of most-similar pair")
    emit("statCircSimBehDist",      _fmt(c["sim_beh_dist"], 4),
         "behavioural distance of most-similar pair")
    emit("statCircSimSharedEdges",  c["sim_shared_edges"],
         "shared edges in most-similar pair")
    emit("statCircSimUnionEdges",   c["sim_union_edges"],
         "union edges in most-similar pair")
    emit("statCircSimSharedFracPct", _fmt(c["sim_shared_frac"] * 100, 1),
         "shared fraction % in most-similar pair")

    # Most dissimilar within-mouse pair
    tex.append("% Dissimilar pair (max Jaccard within-mouse)")
    emit("statCircDisMouse",        c["dis_mouse"],
         "mouse of most-dissimilar within-mouse pair")
    emit("statCircDisJaccard",      _fmt(c["dis_jaccard"], 3),
         "Jaccard distance of most-dissimilar pair")
    emit("statCircDisBehDist",      _fmt(c["dis_beh_dist"], 4),
         "behavioural distance of most-dissimilar pair")
    emit("statCircDisSharedEdges",  c["dis_shared_edges"],
         "shared edges in most-dissimilar pair")
    emit("statCircDisUnionEdges",   c["dis_union_edges"],
         "union edges in most-dissimilar pair")
    emit("statCircDisSharedFracPct", _fmt(c["dis_shared_frac"] * 100, 1),
         "shared fraction % in most-dissimilar pair")

    # Layer breakdown for dissimilar pair (rep 1 = rep_a, rep 5 = rep_b)
    tex.append("% Dissimilar pair layer breakdown (rep_a=r1, rep_b=r5)")
    emit("statCircDisRepOneSensInter",  c["dis_rep_a_SensInter"],
         "dissimilar pair rep1 S->I edge count")
    emit("statCircDisRepOneInterInter", c["dis_rep_a_InterInter"],
         "dissimilar pair rep1 I->I edge count")
    emit("statCircDisRepOneInterMotor", c["dis_rep_a_InterMotor"],
         "dissimilar pair rep1 I->M edge count")
    emit("statCircDisRepFiveSensInter",  c["dis_rep_b_SensInter"],
         "dissimilar pair rep5 S->I edge count")
    emit("statCircDisRepFiveInterInter", c["dis_rep_b_InterInter"],
         "dissimilar pair rep5 I->I edge count")
    emit("statCircDisRepFiveInterMotor", c["dis_rep_b_InterMotor"],
         "dissimilar pair rep5 I->M edge count")

    return tex, txt


# ---------------------------------------------------------------------------

def _build_spec_evol() -> tuple[list[str], list[str]]:
    """Block 6: §2.2 / §2.6 specialisation evolution macros.

    Sources:
      analysis/spec_evolution.pkl        — nb16 (per-gen per-mouse spec index)
      analysis/spec_evol_generalist.pkl  — nb17 (per-gen generalist mean |bias|)
    """
    se_path = ANALYSIS / "spec_evolution.pkl"
    sg_path = ANALYSIS / "spec_evol_generalist.pkl"
    se = _load(se_path)
    sg = _load(sg_path)

    import numpy as np
    sp_mean = np.array(se["spec_mean"])   # (16, 9)
    pop_mean_by_gen = sp_mean.mean(axis=1)

    spec_init  = float(pop_mean_by_gen[0])    # gen 1
    spec_final = float(pop_mean_by_gen[-1])   # gen 150
    gen_final  = float(sg["mean_abs_bias"][-1])  # generalist mean |bias| at gen 150

    tex: list[str] = []
    txt: list[str] = []

    def emit(name: str, val: float, comment: str, fmt: str = "{:.3f}") -> None:
        rendered = fmt.format(val)
        tex.append(
            f"\\newcommand{{\\{name}}}{{{rendered}}}  % {comment}"
        )
        txt.append(f"  \\{name} = {rendered}  ({comment})")

    tex.append("% Specialist specialisation index by generation")
    emit("statSpecEvolSpecInit",  spec_init,  "specialist pop mean spec index at gen 1")
    emit("statSpecEvolSpecFinal", spec_final, "specialist pop mean spec index at gen 150")
    emit("statSpecEvolGenFinal",  gen_final,  "generalist mean |bias| at gen 150")

    return tex, txt


# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # Block 1
    bag = StatBag()
    _build_core(bag)
    core_tex = bag.tex_lines()
    core_txt = bag.txt_lines()

    # Block 2
    deg_tex, deg_txt = _build_degeneracy()

    # Block 3
    c28_tex, c28_txt = _build_claim28()

    # Block 4
    act_tex, act_txt = _build_act_emb()

    # Block 5
    circ_tex, circ_txt = _build_circuits()

    # Block 6
    se_tex, se_txt = _build_spec_evol()

    # Assemble .tex
    tex_file_lines = [
        "% paper_stats.tex — auto-generated by scripts/build_paper_stats.py",
        "% DO NOT edit by hand. Re-run the script after any analysis update.",
        "%",
        "% Block 1  (\\stat...)            core per-mouse, ANOVA, permutation, dynamics",
        "% Block 2  (\\statDeg...)         degeneracy analyses A1-A6",
        "% Block 3  (\\statClaimTwentyEight*) sensitivity convergence (nb18)",
        "% Block 4  (\\statActEmb*)        activity embedding macros (§2.7, nb20)",
        "% Block 5  (\\statCirc*)          circuit comparison macros (§2.8)",
        "% Block 6  (\\statSpecEvol*)      specialisation evolution macros (§2.2/§2.6)",
        "%",
        "% Both paper/latex/ and paper_v2/latex/ \\input this file.",
        "% Each paper uses the subset of macros it needs.",
        "",
        "% ============================================================",
        "% Block 1 — core statistics",
        "% ============================================================",
        "",
    ] + core_tex + [
        "",
        "% ============================================================",
        "% Block 2 — degeneracy analyses (A1-A6)",
        "% ============================================================",
        "",
    ] + deg_tex + [
        "",
        "% ============================================================",
        "% Block 3 — claim 28: sensitivity convergence (nb18)",
        "% ============================================================",
        "",
    ] + c28_tex + [
        "",
        "% ============================================================",
        "% Block 4 — activity embedding macros (§2.7, nb20)",
        "% ============================================================",
        "",
    ] + act_tex + [
        "",
        "% ============================================================",
        "% Block 5 — circuit comparison macros (§2.8)",
        "% ============================================================",
        "",
    ] + circ_tex + [
        "",
        "% ============================================================",
        "% Block 6 — specialisation evolution macros (§2.2/§2.6)",
        "% ============================================================",
        "",
    ] + se_tex + [""]

    OUT_TEX.write_text("\n".join(tex_file_lines), encoding="utf-8")

    # Assemble .txt
    txt_file_lines = [
        "paper_stats.txt — auto-generated by scripts/build_paper_stats.py",
        "Human-readable companion to stats/paper_stats.tex.",
        "",
        "## Block 1 — core statistics",
    ] + core_txt + [
        "",
        "## Block 2 — degeneracy analyses (A1-A6)",
    ] + deg_txt + [
        "",
        "## Block 3 — claim 28: sensitivity convergence (nb18)",
    ] + c28_txt + [
        "",
        "## Block 4 — activity embedding macros (§2.7, nb20)",
    ] + act_txt + [
        "",
        "## Block 5 — circuit comparison macros (§2.8)",
    ] + circ_txt + [
        "",
        "## Block 6 — specialisation evolution macros (§2.2/§2.6)",
    ] + se_txt + [""]

    OUT_TXT.write_text("\n".join(txt_file_lines), encoding="utf-8")

    n_macros = sum(1 for l in tex_file_lines if l.startswith("\\newcommand"))
    print(f"Written {n_macros} macros to:")
    print(f"  {OUT_TEX}")
    print(f"  {OUT_TXT}")


if __name__ == "__main__":
    main()
