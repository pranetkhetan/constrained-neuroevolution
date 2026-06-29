# Selection Navigates a Degenerate Circuit Space

Behavioral individuation without structural differentiation in constrained neuroevolution.

This repository contains the code, the distilled best-evolved agents, the analysis
intermediates, and the manuscript for the paper. It is designed so that a reviewer can
**clone it and regenerate every figure, statistic, and the PDF** without downloading
the large evolutionary archive.

> **Paper:** *Selection navigates a degenerate circuit space: behavioral individuation
> without structural differentiation in constrained neuroevolution.*
> Khetan & Asopa. Preprint DOI: `[BIORXIV-DOI]`.
> **Claim → evidence registry:** [`paper_v2/claims.md`](paper_v2/claims.md).

### Reproduction tiers at a glance

| Tier | You want to… | You need | Cost |
|------|--------------|----------|------|
| **1** | Regenerate all figures, stats, and the PDF | **this repo only** (`pip install` + `rebuild_all.sh`) | minutes, CPU |
| **2** | Re-derive the `analysis/*.pkl` intermediates / audit full activations | Tier-1 **+ Zenodo archive** (full agents, full `B_results.pkl`) | ~1 hr, CPU |
| **3** | Re-evolve the agents from the mouse data | Tier-1 **+ Rosenberg trajectories** (+ GPU) | ~days, GPU |

Most readers want **Tier 1** — it requires nothing beyond a clone.

---

## What this is

We evolve minimal 14-neuron recurrent circuits (Dale's Law, sparse connectivity,
quantized weights) to replicate the natural navigation behavior of 9 individual mice
(Rosenberg et al. 2021), 6 replicate runs per mouse (54 runs total). Architectural
constraints impose a structural floor (0/18 aggregate features differ across mice),
yet behavioral individuation is robust — and what selection shapes is *functional
sensitivity commitment*, which is itself degenerate. Degeneracy is shown to span every
level examined: aggregate statistics, topology, causal sensitivity, representational
geometry, and dynamical geometry.

## Repository layout

```
core/          Agent, evolution, fitness, simulation; core/paths.py (data-path resolution)
utils/         Backend (CPU/GPU), Markov, maze, metrics helpers
scripts/       Pipeline (run/preprocess/analyze/stats), figure generation, reductions
  figure_modules/   One module per paper figure (+ _style.py, _loaders.py)
analysis/      Intermediate results the figures/stats read (Tier-1 inputs)
figures/       Pre-compiled paper figures (regenerable)
stats/         paper_stats.tex — single source of truth for every paper number
paper_v2/      Manuscript (LaTeX, PDF) + claims.md
data/          Small derived data: per-mouse metrics + best_agents.pkl (the 54+6 best agents)
tests/         Unit tests
rebuild_all.sh One-command pipeline: stats -> figures -> LaTeX
```

## How the data and code are partitioned (repo vs Zenodo)

The full project produces ~9 GB of evolutionary data. To keep this repository
clone-able while remaining fully reproducible, the artifacts are split by what the
*analysis/figure pipeline actually reads*:

| Artifact | Size | Lives in | Why |
|----------|------|----------|-----|
| All code (`core/ utils/ scripts/`) | small | **repo** | the pipeline itself |
| Manuscript (`paper_v2/`, `stats/`, `figures/`) | ~30 MB | **repo** | the paper + regenerable figures |
| Per-mouse metrics (`data/mouse_*_metrics.pkl`) | ~40 KB | **repo** | inputs to evolution; tiny, derived |
| **`data/best_agents.pkl`** (54 specialists + 6 generalists, gen 150) | ~130 KB | **repo** | the *only* part of the 8.7 GB agent archive the pipeline reads |
| `analysis/*.pkl` intermediates | ~35 MB | **repo** | precomputed Tier-1 inputs for figures/stats |
| **Slim** `B_results.pkl` (derived results + 4 highlighted agents' activations) | ~9 MB | **repo** | enough for every embedding figure |
| Full `data/agents/` + `data/generalist/` (all generations, all agents) | ~9.7 GB | **Zenodo** | only needed to re-derive `analysis/*.pkl` or audit full runs |
| Full `B_results.pkl` (per-frame activations, all 54 agents) | ~116 MB | **Zenodo** | only needed to recompute embedding intermediates from scratch |
| Raw Rosenberg trajectories | — | **not redistributed** | license; fetch from the original source |

The distillation that makes this split possible is done by two scripts (run once with
the full archive): `scripts/extract_best_agents.py` (agents → `best_agents.pkl`) and
`scripts/slim_b_results.py` (full → slim `B_results.pkl`). Both are in the repo so the
reduction is itself reproducible.

## Quick start — reproduce the paper (Tier 1)

No large download or GPU needed; everything required is in the repo.

```bash
pip install -r requirements.txt

# Regenerate stats macros + all figures + compile the PDF:
PYTHON=python bash rebuild_all.sh --extract
# -> stats/paper_stats.tex, figures/*.pdf, paper_v2/latex/main_v2.pdf

# Or just the figures (skip LaTeX):
PYTHON=python bash rebuild_all.sh --extract --figs-only
```

Verify the shipped numbers match the data:

```bash
python scripts/build_paper_stats.py          # regenerates stats/paper_stats.tex
python scripts/verify_stats_against_data.py   # asserts macros == data
python scripts/detect_monkeypatches.py        # guards against value-mutating patches
pytest tests/
```

### The `best_agents.pkl` keystone

`data/best_agents.pkl` (~130 KB) holds the 54 specialist + 6 generalist gen-150 best
agents — everything the figure/analysis pipeline ever reads from the full 8.7 GB
evolutionary archive. This is why Tier 1 needs no large download. It was produced from
the full archive by `scripts/extract_best_agents.py`.

## Full reproduction from scratch (Tier 2)

Re-deriving the `analysis/*.pkl` intermediates, or auditing the full evolutionary
trajectories, requires the large archive deposited on Zenodo:

> **Zenodo:** `[ZENODO-DOI]` — full per-generation agent runs + full per-frame
> activations. See [`docs/ZENODO.md`](docs/ZENODO.md) for the archive layout and how it
> maps onto the environment variables below.

1. Download and unzip the archive anywhere on disk.
2. Point the code at it (no source edits needed):

   ```bash
   # bash
   export NEUROEVO_DATA_DIR=/path/to/zenodo/data          # contains agents/, generalist/
   export NEUROEVO_ANALYSIS_DIR=/path/to/zenodo/analysis  # contains the full B_results.pkl
   ```
   ```powershell
   # PowerShell
   $env:NEUROEVO_DATA_DIR     = "D:\zenodo\data"
   $env:NEUROEVO_ANALYSIS_DIR = "D:\zenodo\analysis"
   ```

   Path resolution lives in [`core/paths.py`](core/paths.py); unset variables fall back
   to the in-repo defaults.

3. Re-derive the analysis intermediates from the full archive, then re-distil:

   ```bash
   # (a) regenerate analysis/*.pkl from the full agents
   python scripts/analyze_circuits.py            # circuit_features, cross-mouse matrix
   python scripts/supp_analyses.py --run all     # A1-A6 degeneracy suite
   python scripts/phase3_colab_analyses.py       # random-permutation, generalist, power
   python scripts/holdout_eval.py                # held-out specialisation
   python scripts/ei_analysis.py                 # E/I temporal caches

   # (b) recompute embedding robustness from the FULL B_results
   python scripts/embedding_robustness.py        # -> analysis/.../embedding_robustness.pkl

   # (c) re-distil the small artifacts the repo ships
   python scripts/extract_best_agents.py         # full agents   -> data/best_agents.pkl
   python scripts/slim_b_results.py              # full B_results -> slim B_results.pkl

   # (d) rebuild stats + figures + PDF
   PYTHON=python bash rebuild_all.sh --extract
   ```

### Re-running the evolution

To re-evolve agents from the mouse behavioral metrics (GPU recommended; ~days for all
54 runs):

```bash
python scripts/preprocess_mouse.py <rosenberg-trajectories>   # -> data/mouse_*_metrics.pkl (already shipped)
python scripts/run.py loop --start 1 --end 150 --mouse B5 --seed 1
# ... repeat per mouse/replicate; then the Tier-2 analysis steps above.
```

## Data sources

- **Mouse behavioral data:** Rosenberg, Zhang, Perona & Meister (2021), *eLife*. Raw
  trajectories are **not redistributed here**; obtain them from the original source.
  `scripts/preprocess_mouse.py` converts them into the per-mouse metric pickles that
  ship in `data/`.
- **Evolved agents (full archive) & full activations:** Zenodo `[ZENODO-DOI]`.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite both the paper and the Zenodo archive.

## License

See [`LICENSE`](LICENSE).
