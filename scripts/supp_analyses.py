#!/usr/bin/env python
"""
Supplementary analyses for the constrained neuroevolution mouse paper.

Six analyses:
  A1 — Bootstrap test: generalist vs specialist cosine similarity (0.186 vs 0.224)
  A2 — "Between own and other" Δfit decomposition figure
  A3 — Per-mouse fitness correlation: specialist vs generalist
  A4 — Strain confound check from 9×9 cross-mouse matrix
  A5 — Random circuit specialization null distribution  [GPU required]
  A6 — Per-metric specialization text + LaTeX table (Q10 from REVISION_SESSION)

Usage (single analysis):
    python supp_analyses.py --run A1
    python supp_analyses.py --run A4
    python supp_analyses.py --run all

Outputs go to analysis/ (data) and figures/ (plots).
"""

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Ensure the repo root is importable so unpickling Agent objects (core.agent)
# works when this script is run from anywhere (e.g. `python scripts/supp_analyses.py`).
for _p in (BASE_DIR, os.path.join(BASE_DIR, 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
ANALYSIS   = os.path.join(BASE_DIR, 'analysis')
FIGURES    = os.path.join(BASE_DIR, 'figures')
DATA       = os.path.join(BASE_DIR, 'data')

MICE = ['B5', 'B6', 'B7', 'D3', 'D4', 'D5', 'D7', 'D8', 'D9']
GEN  = 150
N_REPS = 6

B_STRAIN = ['B5', 'B6', 'B7']
D_STRAIN = ['D3', 'D4', 'D5', 'D7', 'D8', 'D9']

FIGURE_DPI = 150
FIGURE_STYLE = {
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
}

os.makedirs(FIGURES, exist_ok=True)
os.makedirs(ANALYSIS, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('cupy'):
            module = module.replace('cupy._core.core', 'numpy').replace('cupy', 'numpy')
        return super().find_class(module, name)


def _load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        with open(path, 'rb') as f:
            return _CpuUnpickler(f).load()


def _load_specificity():
    path = os.path.join(ANALYSIS, 'specificity_results.pkl')
    return _load_pickle(path)


def _load_generalist():
    path = os.path.join(ANALYSIS, 'generalist_results.pkl')
    return _load_pickle(path)


def _load_gen_matrix():
    path = os.path.join(ANALYSIS, 'generalization_matrix.npy')
    return np.load(path)


def _load_per_metric():
    path = os.path.join(ANALYSIS, 'cross_mouse_per_metric.pkl')
    return _load_pickle(path)


def _load_all_weight_vectors():
    """Load weight matrices for all 54 evolved agents (54 × 14 × 14 flat)."""
    weights = []
    agents_dir = os.path.join(BASE_DIR, 'data', 'agents')  # Specialist agents are in agents/, not data/
    for mouse in MICE:
        for rep in range(1, N_REPS + 1):
            results_dir = os.path.join(agents_dir, f'results_{mouse}_r{rep}')
            summary_path = os.path.join(results_dir, f'gen_{GEN}', 'summary.pkl')
            if not os.path.exists(summary_path):
                print(f'  WARNING: {summary_path} not found — skipping')
                continue
            pop = _load_pickle(summary_path)
            best = min(pop, key=lambda r: r['fitness'])
            w = best['agent'].weights
            if hasattr(w, 'get'):
                w = w.get()  # CuPy → NumPy
            weights.append(w.flatten().astype(np.float64))
    return np.array(weights)  # (54, 196)


def _load_generalist_weight_vectors(gen_data):
    """Load weight vectors for the N_REPS_G generalist agents from disk.

    The pkl doesn't store the agent objects themselves — colab_11 Cell C8
    only persists results_C / delta_profiles / fitness arrays. Re-load
    each generalist's best agent the same way colab_11 Cell C1 does:
    data/generalist/results_r{rep}/gen_{GEN}/summary.pkl, take the agent
    with the lowest mean-across-9-mice fitness.
    """
    n_reps = int(gen_data.get('N_REPS_G', 6))
    gen_dir = os.path.join(DATA, 'generalist')
    weights = []
    for rep in range(n_reps):
        summary_path = os.path.join(gen_dir, f'results_r{rep}', f'gen_{GEN}', 'summary.pkl')
        if not os.path.exists(summary_path):
            print(f'  WARNING: {summary_path} not found — skipping')
            continue
        pop = _load_pickle(summary_path)
        best = min(pop, key=lambda r: r['fitness'])
        w = best['agent'].weights
        if hasattr(w, 'get'):
            w = w.get()
        weights.append(w.flatten().astype(np.float64))
    return np.array(weights)  # (n_reps, 196)


def _cosine_sim(a, b):
    """Cosine similarity between two flat vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _all_pairwise_cosine(W):
    """All unique pairwise cosine similarities for an (n, d) weight matrix."""
    n = len(W)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(_cosine_sim(W[i], W[j]))
    return np.array(sims)


# ---------------------------------------------------------------------------
# A1 — Bootstrap test: generalist vs specialist cosine similarity
# ---------------------------------------------------------------------------

def run_A1(n_bootstrap=10_000, seed=42):
    """
    Test whether the lower generalist cosine similarity (0.186) vs specialist (0.224)
    is statistically significant, controlling for the massive sample size disparity
    (15 generalist pairs vs 1431 specialist pairs).

    Method: Bootstrap from the specialist pairwise distribution.
    Draw 10,000 samples of n=15 (same as generalist), compute mean each time.
    P-value = fraction of bootstrap means ≤ generalist mean.
    """
    print('\n=== A1: Bootstrap test — cosine similarity ===')

    gen_data  = _load_generalist()
    spec_W    = _load_all_weight_vectors()
    gen_W     = _load_generalist_weight_vectors(gen_data)

    spec_sims = _all_pairwise_cosine(spec_W)
    gen_sims  = _all_pairwise_cosine(gen_W)

    gen_mean  = float(np.mean(gen_sims))
    spec_mean = float(np.mean(spec_sims))
    gen_std   = float(np.std(gen_sims))

    print(f'  Specialist pairwise: n={len(spec_sims)} pairs, mean={spec_mean:.4f}')
    print(f'  Generalist pairwise: n={len(gen_sims)} pairs, mean={gen_mean:.4f} ± {gen_std:.4f}')

    rng = np.random.default_rng(seed)
    boot_means = np.array([
        np.mean(rng.choice(spec_sims, size=len(gen_sims), replace=True))
        for _ in range(n_bootstrap)
    ])

    p_val = float(np.mean(boot_means <= gen_mean))
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    print(f'  Bootstrap (n={n_bootstrap}): 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]')
    print(f'  P(boot_mean ≤ gen_mean) = {p_val:.4f}')
    if p_val < 0.05:
        print('  → SIGNIFICANT: generalist similarity is below specialist range')
    else:
        print('  → NOT SIGNIFICANT: difference consistent with sampling noise')

    # Random baseline (from circuit_features — use generalist data if random not precomputed)
    random_baseline = 0.070  # from GENERALIST_DISCUSSION.md

    # Figure
    plt.rcParams.update(FIGURE_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Left: bar chart with error bars
    ax = axes[0]
    labels  = ['Random\n(null)', 'Generalist\n(n=6)', 'Specialist\n(n=54)']
    means   = [random_baseline, gen_mean, spec_mean]
    errors  = [0, gen_std, float(np.std(spec_sims))]
    colors  = ['#999999', '#e07b39', '#4e8cd4']
    bars = ax.bar(labels, means, yerr=errors, capsize=5, color=colors, alpha=0.85, width=0.55)
    ax.axhline(random_baseline, color='#999', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('Mean pairwise cosine similarity')
    ax.set_title('Structural convergence by training regime')
    ax.set_ylim(0, max(means) * 1.4)

    sig_str = f'p = {p_val:.3f}' if p_val >= 0.001 else 'p < 0.001'
    if p_val < 0.05:
        x0, x1 = 1, 2
        y_line = max(means) * 1.25
        ax.plot([x0, x1], [y_line, y_line], 'k-', lw=1)
        ax.text((x0 + x1) / 2, y_line * 1.02, f'*\n({sig_str})', ha='center', va='bottom', fontsize=8)
    else:
        ax.text(1.5, max(means) * 1.28, f'n.s. ({sig_str})', ha='center', va='bottom', fontsize=8)

    # Right: bootstrap distribution
    ax2 = axes[1]
    ax2.hist(boot_means, bins=60, color='#4e8cd4', alpha=0.7, edgecolor='white', lw=0.3)
    ax2.axvline(gen_mean, color='#e07b39', lw=2, label=f'Generalist mean ({gen_mean:.3f})')
    ax2.axvline(ci_lo, color='#4e8cd4', lw=1.5, ls='--', alpha=0.7)
    ax2.axvline(ci_hi, color='#4e8cd4', lw=1.5, ls='--', alpha=0.7, label='Spec. 95% CI (n=15 bootstrap)')
    ax2.set_xlabel('Bootstrap sample mean (n=15 from specialist pairs)')
    ax2.set_ylabel('Count')
    ax2.set_title('Bootstrap null distribution')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIGURES, 'supp_A1_cosine_bootstrap.pdf')
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out}')

    # Save result dict
    result = {
        'spec_mean': spec_mean, 'gen_mean': gen_mean, 'gen_std': gen_std,
        'boot_ci': (ci_lo, ci_hi), 'p_val': p_val,
        'n_spec_pairs': len(spec_sims), 'n_gen_pairs': len(gen_sims),
    }
    with open(os.path.join(ANALYSIS, 'A1_cosine_bootstrap.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return result


# ---------------------------------------------------------------------------
# A2 — "Between own and other" Δfit decomposition figure
# ---------------------------------------------------------------------------

def run_A2():
    """
    Visualise the three-level Δfit hierarchy:
      Specialist Δfit_own (2.661) > Generalist Δfit (~2.47, flat) > Specialist Δfit_other (2.274)

    Decomposition: 2.274 = generic navigation disruption; +0.39 = individual calibration.
    """
    print('\n=== A2: Δfit decomposition — own / generalist / other ===')

    spec_data = _load_specificity()
    gen_data  = _load_generalist()

    # Option A results
    oa = spec_data['option_A']
    mice_order = spec_data.get('metadata', {}).get('mice', MICE)

    delta_own   = np.array(oa['delta_own'])    # shape (9,)
    delta_other = np.array(oa['delta_other'])  # shape (9,)

    # Generalist Δfit per mouse (mean over permutations)
    gen_mean_by_mouse = dict(zip(gen_data['MICE'], gen_data['gen_mean_per_mouse']))
    gen_delta = np.array([gen_mean_by_mouse[m] for m in mice_order])

    mean_own   = float(np.mean(delta_own))
    mean_other = float(np.mean(delta_other))
    mean_gen   = float(np.mean(gen_delta))

    print(f'  Specialist Δfit_own   = {mean_own:.3f} ± {np.std(delta_own):.3f}')
    print(f'  Generalist Δfit       = {mean_gen:.3f} ± {np.std(gen_delta):.3f}')
    print(f'  Specialist Δfit_other = {mean_other:.3f} ± {np.std(delta_other):.3f}')
    print(f'  Generic floor         = {mean_other:.3f}')
    print(f'  Individual calibration = {mean_own - mean_other:.3f}')

    plt.rcParams.update(FIGURE_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ---- Left panel: grouped bars per mouse ----
    ax = axes[0]
    x = np.arange(len(mice_order))
    w = 0.27
    ax.bar(x - w, delta_own,   width=w, label='Specialist Δfit_own',   color='#c0392b', alpha=0.85)
    ax.bar(x,     gen_delta,   width=w, label='Generalist Δfit',        color='#e07b39', alpha=0.85)
    ax.bar(x + w, delta_other, width=w, label='Specialist Δfit_other',  color='#4e8cd4', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(mice_order, rotation=45)
    ax.set_ylabel('Permutation Δfit')
    ax.set_title('Per-mouse Δfit: own vs generalist vs other')
    ax.legend(fontsize=8)
    ax.set_ylim(1.5, 3.5)
    ax.axhline(mean_gen,   color='#e07b39', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(mean_own,   color='#c0392b', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(mean_other, color='#4e8cd4', lw=0.8, ls='--', alpha=0.5)

    # ---- Right panel: mean summary with decomposition annotation ----
    ax2 = axes[1]
    labels  = ['Spec. Δfit_other\n(generic floor)', 'Generalist Δfit\n(flat across mice)', 'Spec. Δfit_own\n(individual target)']
    means   = [mean_other, mean_gen, mean_own]
    sems    = [np.std(delta_other) / np.sqrt(len(delta_other)),
               np.std(gen_delta)   / np.sqrt(len(gen_delta)),
               np.std(delta_own)   / np.sqrt(len(delta_own))]
    colors  = ['#4e8cd4', '#e07b39', '#c0392b']
    bars = ax2.bar(range(3), means, yerr=sems, capsize=6, color=colors, alpha=0.85, width=0.55)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('Mean permutation Δfit')
    ax2.set_title('Δfit decomposition: generic + individual')
    ax2.set_ylim(1.5, 3.3)

    # Annotation: decomposition arrow
    y_lo = mean_other + 0.02
    y_hi = mean_own   - 0.02
    ax2.annotate('', xy=(2, y_hi), xytext=(2, y_lo),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax2.text(2.32, (y_lo + y_hi) / 2,
             f'+{mean_own - mean_other:.2f}\nindividual\ncalibration',
             ha='left', va='center', fontsize=8, color='black')

    y_lo2 = 1.6
    y_hi2 = mean_other - 0.04
    ax2.annotate('', xy=(0, y_hi2), xytext=(0, y_lo2),
                 arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))
    ax2.text(-0.45, (y_lo2 + y_hi2) / 2,
             f'{mean_other:.2f}\ngeneric\nfloor',
             ha='center', va='center', fontsize=7.5, color='#555')

    plt.tight_layout()
    out = os.path.join(FIGURES, 'supp_A2_deltafit_decomposition.pdf')
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out}')

    result = {
        'mean_own': mean_own, 'mean_gen': mean_gen, 'mean_other': mean_other,
        'individual_calibration': mean_own - mean_other,
        'generic_floor': mean_other,
        'per_mouse': {m: {'own': float(o), 'gen': float(g), 'other': float(t)}
                      for m, o, g, t in zip(mice_order, delta_own, gen_delta, delta_other)},
    }
    with open(os.path.join(ANALYSIS, 'A2_deltafit_decomposition.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return result


# ---------------------------------------------------------------------------
# A3 — Specialist vs generalist per-mouse fitness correlation
# ---------------------------------------------------------------------------

def run_A3():
    """
    Scatter plot: x = diagonal of 9×9 cross-mouse matrix (specialist own-mouse fitness),
    y = generalist per-mouse fitness.
    Expected: strong positive correlation (same mice are hard/easy for both).
    """
    print('\n=== A3: Specialist vs generalist per-mouse fitness ===')

    gen_data = _load_generalist()
    mat      = _load_gen_matrix()   # (9, 9)

    # Diagonal = each specialist on its own mouse (best rep used in colab_4)
    spec_own = np.diag(mat)         # shape (9,)
    # Generalist per-mouse fitness: mean orig_fitness across the 6 generalist reps,
    # computed from results_C (the pkl doesn't store this as a flat per-mouse array).
    pkl_mice = gen_data['MICE']
    results_C = gen_data['results_C']
    gen_fit_by_mouse = {
        m: float(np.mean([results_C[r][m]['orig_fitness'] for r in range(len(results_C))]))
        for m in pkl_mice
    }
    gen_fit = np.array([gen_fit_by_mouse[m] for m in MICE])

    r, p = stats.pearsonr(spec_own, gen_fit)
    print(f'  Pearson r = {r:.3f}, p = {p:.4f}')

    # Regression line
    m_slope, b_int = np.polyfit(spec_own, gen_fit, 1)

    plt.rcParams.update(FIGURE_STYLE)
    fig, ax = plt.subplots(figsize=(5, 4.5))

    colors_map = {'B5': '#2196F3', 'B6': '#1565C0', 'B7': '#0D47A1',
                  'D3': '#E53935', 'D4': '#C62828', 'D5': '#B71C1C',
                  'D7': '#FF8F00', 'D8': '#E65100', 'D9': '#BF360C'}

    for i, mouse in enumerate(MICE):
        c = colors_map.get(mouse, '#555')
        ax.scatter(spec_own[i], gen_fit[i], color=c, s=70, zorder=3)
        ax.annotate(mouse, (spec_own[i], gen_fit[i]),
                    textcoords='offset points', xytext=(5, 3), fontsize=8, color=c)

    x_range = np.linspace(spec_own.min() * 0.97, spec_own.max() * 1.03, 100)
    ax.plot(x_range, m_slope * x_range + b_int, 'k--', lw=1.2, alpha=0.7)

    sig_str = f'p = {p:.3f}' if p >= 0.001 else 'p < 0.001'
    ax.set_xlabel('Specialist own-mouse fitness (9×9 diagonal)')
    ax.set_ylabel('Generalist per-mouse fitness')
    ax.set_title(f'Mouse-intrinsic difficulty\n(r = {r:.3f}, {sig_str})')

    # Strain legend
    patch_b = mpatches.Patch(color='#1565C0', label='B-strain (B5–B7)')
    patch_d = mpatches.Patch(color='#C62828', label='D-strain (D3–D9)')
    ax.legend(handles=[patch_b, patch_d], fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIGURES, 'supp_A3_difficulty_correlation.pdf')
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out}')

    result = {'pearson_r': r, 'pearson_p': p, 'spec_own': dict(zip(MICE, spec_own.tolist())),
              'gen_fit': dict(zip(MICE, gen_fit.tolist()))}
    with open(os.path.join(ANALYSIS, 'A3_difficulty_correlation.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return result


# ---------------------------------------------------------------------------
# A4 — Strain confound check
# ---------------------------------------------------------------------------

def run_A4():
    """
    From 9×9 cross-mouse matrix, compare:
      - Own-mouse fitness (diagonal)
      - Within-strain cross-mouse fitness (B↔B or D↔D, off-diagonal)
      - Cross-strain fitness (B↔D, all off-diagonal entries)

    If within-strain ≈ cross-strain → specialization is INDIVIDUAL, not strain-level.
    """
    print('\n=== A4: Strain confound check ===')

    mat = _load_gen_matrix()   # (9, 9), rows = evaluating agent's mouse, cols = baseline mouse

    n = len(MICE)
    idx = {m: i for i, m in enumerate(MICE)}

    own_vals         = []
    within_strain    = []
    cross_strain     = []

    for i, m_agent in enumerate(MICE):
        for j, m_baseline in enumerate(MICE):
            val = float(mat[i, j])
            if i == j:
                own_vals.append(val)
            else:
                same_strain = ((m_agent in B_STRAIN and m_baseline in B_STRAIN) or
                               (m_agent in D_STRAIN and m_baseline in D_STRAIN))
                if same_strain:
                    within_strain.append(val)
                else:
                    cross_strain.append(val)

    own_arr    = np.array(own_vals)
    within_arr = np.array(within_strain)
    cross_arr  = np.array(cross_strain)

    print(f'  Own-mouse:        n={len(own_arr)}, mean={np.mean(own_arr):.4f} ± {np.std(own_arr):.4f}')
    print(f'  Within-strain:    n={len(within_arr)}, mean={np.mean(within_arr):.4f} ± {np.std(within_arr):.4f}')
    print(f'  Cross-strain:     n={len(cross_arr)}, mean={np.mean(cross_arr):.4f} ± {np.std(cross_arr):.4f}')

    t_stat, p_within_vs_cross = stats.ttest_ind(within_arr, cross_arr)
    mw_stat, p_mw = stats.mannwhitneyu(within_arr, cross_arr, alternative='two-sided')
    print(f'\n  Within vs cross-strain: t={t_stat:.3f}, p={p_within_vs_cross:.4f}')
    print(f'  Mann-Whitney U: stat={mw_stat:.1f}, p={p_mw:.4f}')

    if p_within_vs_cross > 0.05:
        print('  → NOT SIGNIFICANT: within-strain ≈ cross-strain → strain is NOT a confound')
    else:
        print('  → SIGNIFICANT: strain-level clustering present — possible confound')

    # Specialization ratio within each strain
    b_own   = [mat[idx[m], idx[m]] for m in B_STRAIN]
    b_cross = [mat[idx[m], idx[n]] for m in B_STRAIN for n in B_STRAIN if m != n]
    d_own   = [mat[idx[m], idx[m]] for m in D_STRAIN]
    d_cross = [mat[idx[m], idx[n]] for m in D_STRAIN for n in D_STRAIN if m != n]

    b_ratio = np.mean(b_own) / np.mean(b_cross) if np.mean(b_cross) > 0 else np.nan
    d_ratio = np.mean(d_own) / np.mean(d_cross) if np.mean(d_cross) > 0 else np.nan
    print(f'\n  Within-B specialization ratio: {b_ratio:.4f}')
    print(f'  Within-D specialization ratio: {d_ratio:.4f}')
    print(f'  Overall specialization ratio:  {np.mean(own_arr) / np.mean(np.concatenate([within_arr, cross_arr])):.4f}')

    # Figure
    plt.rcParams.update(FIGURE_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: bar chart
    ax = axes[0]
    cat_labels = ['Own-mouse\n(diagonal)', 'Within-strain\n(off-diagonal)', 'Cross-strain\n(off-diagonal)']
    cat_means  = [np.mean(own_arr), np.mean(within_arr), np.mean(cross_arr)]
    cat_sems   = [np.std(own_arr) / np.sqrt(len(own_arr)),
                  np.std(within_arr) / np.sqrt(len(within_arr)),
                  np.std(cross_arr)  / np.sqrt(len(cross_arr))]
    cat_colors = ['#2e7d32', '#1565C0', '#c62828']
    ax.bar(range(3), cat_means, yerr=cat_sems, capsize=5, color=cat_colors, alpha=0.85, width=0.55)
    ax.set_xticks(range(3))
    ax.set_xticklabels(cat_labels)
    ax.set_ylabel('Cross-evaluation fitness (mean)')
    ax.set_title('Fitness by target relationship')

    sig_label = f't-test: p = {p_within_vs_cross:.3f}' if p_within_vs_cross >= 0.001 else 't-test: p < 0.001'
    ax.text(1.5, max(cat_means) * 1.1, sig_label, ha='center', fontsize=8)

    # Right: heatmap of 9×9 matrix with strain annotations
    ax2 = axes[1]
    im = ax2.imshow(mat, cmap='RdYlGn_r', aspect='auto',
                    vmin=np.min(mat), vmax=np.max(mat))
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(MICE, rotation=45, fontsize=7)
    ax2.set_yticklabels(MICE, fontsize=7)
    ax2.set_xlabel('Target mouse (baseline)')
    ax2.set_ylabel('Agent mouse (trained for)')
    ax2.set_title('9×9 cross-evaluation fitness')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # Draw strain boundary lines (B=0:3, D=3:9)
    for pos in [2.5]:
        ax2.axhline(pos, color='white', lw=1.5, ls='--')
        ax2.axvline(pos, color='white', lw=1.5, ls='--')
    ax2.text(1, -0.7, 'B', ha='center', fontsize=9, color='#1565C0', weight='bold')
    ax2.text(5.5, -0.7, 'D', ha='center', fontsize=9, color='#c62828', weight='bold')
    ax2.text(-0.7, 1, 'B', va='center', fontsize=9, color='#1565C0', weight='bold')
    ax2.text(-0.7, 5.5, 'D', va='center', fontsize=9, color='#c62828', weight='bold')

    plt.tight_layout()
    out = os.path.join(FIGURES, 'supp_A4_strain_confound.pdf')
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out}')

    result = {
        'mean_own': float(np.mean(own_arr)),
        'mean_within': float(np.mean(within_arr)),
        'mean_cross': float(np.mean(cross_arr)),
        'p_within_vs_cross': float(p_within_vs_cross),
        'p_mw': float(p_mw),
        'b_ratio': float(b_ratio),
        'd_ratio': float(d_ratio),
    }
    with open(os.path.join(ANALYSIS, 'A4_strain_confound.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return result


# ---------------------------------------------------------------------------
# A5 — Random circuit specialization null distribution  [GPU recommended]
# ---------------------------------------------------------------------------

def run_A5(n_random=54, seed=99):
    """
    Compute specialization ratios for random constrained circuits.
    Evaluates n_random random agents on all 9 mice, builds pseudo-9×9 matrix,
    computes per-agent specialization ratios.

    Expected: all ratios ≈ 1.0 — random circuits do not specialize.
    This provides an explicit null distribution for the paper's central claim
    (evolved specialist ratio = 0.656).

    Note: This requires GPU for reasonable speed (~10–30 min on L4).
          On CPU it will run but slowly.
    """
    print('\n=== A5: Random circuit null distribution [GPU recommended] ===')

    import sys
    sys.path.insert(0, BASE_DIR)
    from config import load_config
    from core.agent import Agent
    from core.fitness import evaluate_batch
    from core.simulation import Simulation

    config_path = os.path.join(BASE_DIR, 'config.yaml')
    config = load_config(config_path)

    # Generate noise matrix (same as training)
    rng = np.random.default_rng(seed)
    from utils.backend import xp
    noise = np.zeros((config.simulation.n_bouts, config.simulation.max_frames), dtype=np.float64)
    for k in range(config.simulation.n_bouts):
        local_rng = np.random.RandomState(k)
        noise[k] = local_rng.uniform(-1, 1, size=config.simulation.max_frames)
    noise_xp = xp.array(noise)

    simulation = Simulation(config.simulation)

    # Load baselines for all 9 mice
    from run_generalist import load_all_mouse_baselines
    all_baselines = load_all_mouse_baselines()

    # Initialize random agents
    np.random.seed(seed)
    random_agents = [Agent(config.network) for _ in range(n_random)]
    print(f'  Generated {n_random} random constrained circuits')

    # Evaluate each random agent on all 9 mice
    # Result: (n_random, 9) fitness matrix
    print('  Evaluating random agents on all 9 mice...')
    fitness_matrix = np.zeros((n_random, len(MICE)))

    batch_size = 18  # Process in batches to avoid OOM
    for mouse_idx, mouse_id in enumerate(MICE):
        baseline = all_baselines[mouse_id]
        for batch_start in range(0, n_random, batch_size):
            batch = random_agents[batch_start: batch_start + batch_size]
            results = evaluate_batch(batch, simulation, config, baseline, noise_xp)
            for k, r in enumerate(results):
                fitness_matrix[batch_start + k, mouse_idx] = r.total
        print(f'    {mouse_id}: done (mean={np.mean(fitness_matrix[:, mouse_idx]):.4f})')

    # Compute specialization ratios
    # For random circuits without a "designated" mouse, assign round-robin
    # to get a pseudo-specialization: ratio = own / mean(others)
    # Agents 0..5 → B5, 6..11 → B6, ..., 48..53 → D9
    ratios = []
    mice_assignments = []
    reps_per = n_random // len(MICE)
    for i, agent_idx in enumerate(range(n_random)):
        assigned_mouse_idx = agent_idx // reps_per if reps_per > 0 else 0
        assigned_mouse_idx = min(assigned_mouse_idx, len(MICE) - 1)
        mice_assignments.append(MICE[assigned_mouse_idx])
        own_fit   = fitness_matrix[agent_idx, assigned_mouse_idx]
        other_fit = np.mean([fitness_matrix[agent_idx, j]
                             for j in range(len(MICE)) if j != assigned_mouse_idx])
        ratio = own_fit / other_fit if other_fit > 0 else 1.0
        ratios.append(ratio)

    ratios = np.array(ratios)
    print(f'\n  Random circuit specialization ratios:')
    print(f'    mean = {np.mean(ratios):.4f} ± {np.std(ratios):.4f}')
    print(f'    min  = {np.min(ratios):.4f}, max = {np.max(ratios):.4f}')
    print(f'    Evolved specialists: ratio = 0.656')

    # Figure
    plt.rcParams.update(FIGURE_STYLE)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ratios, bins=20, color='#999999', alpha=0.8, edgecolor='white', lw=0.5,
            label=f'Random circuits (n={n_random})')
    ax.axvline(0.656, color='#c0392b', lw=2, label='Evolved specialists (0.656)', zorder=5)
    ax.axvline(np.mean(ratios), color='#555', lw=1.5, ls='--',
               label=f'Random mean ({np.mean(ratios):.3f})', alpha=0.8)
    ax.set_xlabel('Behavioral specialization ratio (own / other fitness)')
    ax.set_ylabel('Count')
    ax.set_title('Specialization null distribution: random vs evolved circuits')
    ax.legend(fontsize=8)

    # Z-score of evolved value
    z = (0.656 - np.mean(ratios)) / np.std(ratios)
    p_one = float(stats.norm.cdf(z))
    ax.text(0.05, 0.92, f'Evolved z = {z:.1f}\n(p < {p_one:.0e})',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    out = os.path.join(FIGURES, 'supp_A5_random_null.pdf')
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out}')

    result = {
        'ratios': ratios.tolist(),
        'mean_ratio': float(np.mean(ratios)),
        'std_ratio': float(np.std(ratios)),
        'fitness_matrix': fitness_matrix.tolist(),
        'mice_assignments': mice_assignments,
    }
    with open(os.path.join(ANALYSIS, 'A5_random_null.pkl'), 'wb') as f:
        pickle.dump(result, f)
    np.save(os.path.join(ANALYSIS, 'A5_random_fitness_matrix.npy'), fitness_matrix)
    return result


# ---------------------------------------------------------------------------
# A6 — Per-metric supplementary table (Q10 from REVISION_SESSION)
# ---------------------------------------------------------------------------

def run_A6():
    """
    Read per_metric_specialization_summary.csv and:
    1. Print the LaTeX snippet for a supplementary table
    2. Print the §2.7 sentence to add to the paper
    3. Save the formatted table to analysis/A6_per_metric_table.tex
    """
    print('\n=== A6: Per-metric supplementary table (Q10) ===')

    import csv

    csv_path = os.path.join(ANALYSIS, 'per_metric_specialization_summary.csv')
    if not os.path.exists(csv_path):
        print(f'  ERROR: {csv_path} not found')
        return None

    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Map internal names to display names
    name_map = {
        'markov':     'Markov transition score',
        'occupancy':  'Spatial occupancy',
        'tortuosity': 'Path tortuosity',
        'turn_bias':  'Turn bias',
        'total':      '\\textbf{Total (composite)}',
    }

    print('\n  ---- LaTeX table snippet ----\n')

    table_lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Per-metric behavioral specialization ratios. '
        r'Each ratio = mean own-mouse fitness / mean cross-mouse fitness; '
        r'values below 1 indicate the agent models its own mouse better than others. '
        r'Turn bias shows the strongest individual signature.}',
        r'\label{tab:per_metric_specialization}',
        r'\begin{tabular}{lccc}',
        r'\hline',
        r'\textbf{Metric} & \textbf{Own-mouse} & \textbf{Cross-mouse} & \textbf{Ratio} \\',
        r'\hline',
    ]

    for row in rows:
        metric = row['metric']
        display = name_map.get(metric, metric)
        own    = float(row['own_mean'])
        other  = float(row['other_mean'])
        ratio  = float(row['ratio'])
        table_lines.append(f'{display} & {own:.4f} & {other:.4f} & {ratio:.4f} \\\\')
        if metric == 'turn_bias':
            table_lines.append(r'\hline')

    table_lines += [
        r'\hline',
        r'\end{tabular}',
        r'\end{table}',
    ]

    table_tex = '\n'.join(table_lines)
    print(table_tex)

    # §2.7 sentence
    # Find turn_bias row
    turn_row  = next(r for r in rows if r['metric'] == 'turn_bias')
    total_row = next(r for r in rows if r['metric'] == 'total')

    sentence = (
        f"All four behavioral metrics contribute to specialization "
        f"(Supplementary Table~\\ref{{tab:per_metric_specialization}}), "
        f"with turn bias showing the strongest individual signature "
        f"(ratio = {float(turn_row['ratio']):.2f}) and the composite fitness "
        f"yielding ratio = {float(total_row['ratio']):.2f}."
    )
    print('\n  ---- §2.7 sentence ----\n')
    print(sentence)

    # Save
    tex_out = os.path.join(ANALYSIS, 'A6_per_metric_table.tex')
    with open(tex_out, 'w') as f:
        f.write(table_tex + '\n\n')
        f.write('% §2.7 sentence:\n')
        f.write('% ' + sentence + '\n')
    print(f'\n  Saved → {tex_out}')

    # Also make a figure (optional bar chart)
    plt.rcParams.update(FIGURE_STYLE)
    display_names = ['Markov', 'Occupancy', 'Tortuosity', 'Turn bias']
    metric_names  = ['markov', 'occupancy', 'tortuosity', 'turn_bias']
    data_rows     = [next(r for r in rows if r['metric'] == m) for m in metric_names]

    own_vals   = [float(r['own_mean']) for r in data_rows]
    other_vals = [float(r['other_mean']) for r in data_rows]

    x    = np.arange(len(display_names))
    w    = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w/2, own_vals,   width=w, label='Own-mouse',   color='#2e7d32', alpha=0.85)
    ax.bar(x + w/2, other_vals, width=w, label='Cross-mouse', color='#c62828', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylabel('Mean fitness component error')
    ax.legend()

    for i, (o, t) in enumerate(zip(own_vals, other_vals)):
        ratio_val = o / t if t > 0 else 1.0
        ax.text(i, max(o, t) + max(other_vals) * 0.04, f'{ratio_val:.2f}', ha='center', fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIGURES, 'supp_A6_per_metric.pdf')
    plt.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f'  Saved figure → {out}')

    return {'table_tex': table_tex, 'sentence': sentence}


# ---------------------------------------------------------------------------
# A6d - Generalist-vs-specialist degeneracy boundary (produces A6_results.pkl)
# ---------------------------------------------------------------------------
#
# This is the producer for ``analysis/degeneracy_analyses/A6_results.pkl`` -- the
# generalist boundary-condition analysis behind Figure 5 (fig:sensitivity_commitment):
# topology diversity, per-neuron ablation-sensitivity variance, and fitness cost of
# generalism. It was historically produced only in a Colab notebook; folded here so the
# documented ``supp_analyses.py`` Tier-2 command regenerates it.
#
# Provenance: generalist agents are read from ``data/best_agents.pkl`` (the keystone),
# NOT from ``data/generalist/`` directly, so A6, generalist_results, and the keystone all
# trace to a single dataset. The replicate count is whatever the keystone holds
# (6 at first submission; 15 after the referee-follow-up extension). The specialist
# side is reused from the shipped ``source_sensitivity_results.pkl`` (54 agents unchanged).
#
# The sensitivity/topology computation is CPU-only (pure NumPy, seed=42, N_PERM=20);
# the fitness-cost step needs ``evaluate_batch`` (GPU). On a CPU-only run the fitness/cost
# arrays are carried forward from the existing pkl with a warning, so the sensitivity
# result can still be refreshed offline.

_A6D_N_STEPS = 1000
_A6D_N_PERM = 20
_A6D_SEED = 42
_SOURCE_NAMES = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5',
                 'I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'M0', 'M1']


def _a6d_to_numpy(w):
    return w.get() if hasattr(w, 'get') else np.asarray(w)


def _a6d_load_generalists_from_keystone():
    """Load the generalist best agents (all replicates) from data/best_agents.pkl."""
    import copy  # noqa: F401  (used by callers)
    ks_path = os.path.join(DATA, 'best_agents.pkl')
    blob = _load_pickle(ks_path)
    gens = blob['generalists']
    reps = sorted(gens.keys())
    agents = [gens[r]['agent'] for r in reps]
    return agents


def _a6d_fixed_sequence():
    rng = np.random.default_rng(_A6D_SEED)
    t = np.linspace(0, 10 * np.pi, _A6D_N_STEPS)
    seq = np.column_stack([
        0.5 + 0.4 * np.sin(t * 0.7),
        0.3 + 0.3 * np.sin(t * 1.1 + 1),
        0.3 + 0.3 * np.cos(t * 1.1),
        0.2 + 0.2 * np.sin(t * 0.5),
        0.1 * np.sin(t * 0.9),
        0.05 * rng.standard_normal(_A6D_N_STEPS),
    ])
    return seq, rng


def _a6d_run_sequence(agent, seq):
    import numpy as _np
    w = _a6d_to_numpy(agent.weights)
    state = _np.zeros(14)
    out = []
    for s in seq:
        state[:6] = s
        state[6:] = _np.tanh(state @ w)[6:]
        out.append(state[12:14].copy())
    return _np.array(out)


def _a6d_permute_source(agent, src_idx, rng):
    import copy
    a = copy.deepcopy(agent)
    a.weights = _a6d_to_numpy(a.weights)
    row = a.weights[src_idx].copy()
    nz = np.where(row != 0)[0]
    if len(nz) > 1:
        perm = rng.permutation(nz.copy())
        new_row = np.zeros_like(row)
        for old_t, new_t in zip(nz, perm):
            new_row[new_t] = row[old_t]
        a.weights[src_idx] = new_row
    return a


def _a6d_permute_all(agent, rng):
    import copy
    a = copy.deepcopy(agent)
    a.weights = _a6d_to_numpy(a.weights)
    for src in range(14):
        row = a.weights[src].copy()
        nz = np.where(row != 0)[0]
        if len(nz) > 1:
            perm = rng.permutation(nz.copy())
            new_row = np.zeros_like(row)
            for old_t, new_t in zip(nz, perm):
                new_row[new_t] = row[old_t]
            a.weights[src] = new_row
    return a


def run_A6_degeneracy(config_path='config.yaml'):
    """Produce analysis/degeneracy_analyses/A6_results.pkl from the keystone agents.

    References:
        Fig 5 (fig:sensitivity_commitment); Methods A6. Sensitivity protocol is the
        per-source ablation MSE normalized by the full-circuit permutation baseline.
    """
    print('\n=== A6d: Generalist degeneracy boundary (A6_results.pkl) ===')
    out_dir = os.path.join(ANALYSIS, 'degeneracy_analyses')
    os.makedirs(out_dir, exist_ok=True)
    a6_path = os.path.join(out_dir, 'A6_results.pkl')

    # ---- specialist side: reuse shipped intermediates (54 agents unchanged) ----
    SS = _load_pickle(os.path.join(ANALYSIS, 'source_sensitivity_results.pkl'))
    spec_sens_matrix = np.asarray(SS['sensitivity_matrix'], float)   # (54,14)
    spec_sens_var = spec_sens_matrix.var(axis=0)                     # (14,)

    prev = _load_pickle(a6_path) if os.path.exists(a6_path) else {}
    spec_topo_sims = np.asarray(prev.get('spec_topo_sims', []), float)  # unchanged

    # ---- generalist side: recompute over the keystone replicate set ----
    gen_agents = _a6d_load_generalists_from_keystone()
    N_GEN = len(gen_agents)
    print(f'  Loaded {N_GEN} generalist agents from best_agents.pkl')

    W_gen = np.array([_a6d_to_numpy(a.weights) for a in gen_agents])   # (N,14,14)
    gen_topo = (W_gen != 0).astype(float).reshape(N_GEN, -1)
    gen_topo_sims = np.array([_cosine_sim(gen_topo[i], gen_topo[j])
                              for i in range(N_GEN) for j in range(i + 1, N_GEN)])
    if spec_topo_sims.size:
        _, mw_p_gen_spec_topo = stats.mannwhitneyu(
            spec_topo_sims, gen_topo_sims, alternative='two-sided')
    else:
        mw_p_gen_spec_topo = float('nan')

    # sensitivity (per-source ablation) + full-circuit normalisation -- CPU
    print('  Computing generalist sensitivity profiles (CPU)...')
    seq, rng = _a6d_fixed_sequence()
    gen_sensitivity = np.zeros((N_GEN, 14))
    for ai, agent in enumerate(gen_agents):
        base = _a6d_run_sequence(agent, seq)
        for src in range(14):
            mses = [np.mean((_a6d_run_sequence(_a6d_permute_source(agent, src, rng), seq) - base) ** 2)
                    for _ in range(_A6D_N_PERM)]
            gen_sensitivity[ai, src] = np.mean(mses)
    gen_sens_var = gen_sensitivity.var(axis=0)

    gen_full_circuit_baselines = np.zeros(N_GEN)
    for ai, agent in enumerate(gen_agents):
        base = _a6d_run_sequence(agent, seq)
        mses = [np.mean((_a6d_run_sequence(_a6d_permute_all(agent, rng), seq) - base) ** 2)
                for _ in range(_A6D_N_PERM)]
        gen_full_circuit_baselines[ai] = np.mean(mses)

    gen_sensitivity_norm = gen_sensitivity.copy().astype(float)
    for ai in range(N_GEN):
        b = gen_full_circuit_baselines[ai]
        gen_sensitivity_norm[ai] = gen_sensitivity_norm[ai] / b if b > 1e-12 else 0.0
    gen_sens_var_norm = gen_sensitivity_norm.var(axis=0)

    ratio = spec_sens_var.mean() / gen_sens_var_norm.mean() if gen_sens_var_norm.mean() > 0 else float('inf')
    print(f'  Specialist norm sens var: {spec_sens_var.mean():.6f}  '
          f'Generalist: {gen_sens_var_norm.mean():.6f}  ratio: {ratio:.2f}x')

    # fitness / cost -- needs evaluate_batch (GPU). Carry forward on CPU-only.
    try:
        from utils.backend import HAS_GPU
    except Exception:
        HAS_GPU = False
    A6_gen_fitness = np.asarray(prev.get('gen_fitness_matrix', np.full((N_GEN, len(MICE)), np.nan)))
    gen_mean_per_mouse = np.asarray(prev.get('gen_mean_per_mouse', np.full(len(MICE), np.nan)))
    gen_cost_pct = np.asarray(prev.get('gen_cost_pct', np.full(len(MICE), np.nan)))
    if HAS_GPU:
        print('  Computing generalist fitness/cost (GPU)...')
        from config import load_config
        from core.simulation import Simulation
        from core.fitness import evaluate_batch
        from utils.backend import xp
        cfg = load_config(config_path)
        sim = Simulation(cfg.physics)
        n_bouts, max_frames = cfg.simulation.n_bouts, cfg.simulation.max_frames
        noise = np.zeros((n_bouts, max_frames))
        for s in range(n_bouts):
            noise[s] = np.random.RandomState(s).uniform(-1, 1, size=max_frames)
        noise = xp.array(noise)
        # per-mouse baselines
        bl = {}
        for m in MICE:
            d = _load_pickle(os.path.join(DATA, f'mouse_{m}_metrics.pkl'))
            bl[m] = {'node_pdf': d.get('node_pdf'), 'revisit_rate': d.get('reversal_rate'),
                     'straightness': d.get('straightness'), 'turn_bias': d.get('turn_bias'),
                     'markov_profile': d.get('markov_profile'), 'physics': d.get('physics', {})}
        A6_gen_fitness = np.full((N_GEN, len(MICE)), np.nan)
        for j, m in enumerate(MICE):
            res = evaluate_batch(gen_agents, sim, cfg, bl[m], noise)
            A6_gen_fitness[:, j] = [r.total for r in res]
        gen_mean_per_mouse = A6_gen_fitness.mean(axis=0)
        # specialist own-mouse reference: recover from prior pkl (unchanged specialists)
        prev_gmp = np.asarray(prev.get('gen_mean_per_mouse', gen_mean_per_mouse), float)
        prev_cost = np.asarray(prev.get('gen_cost_pct', np.zeros(len(MICE))), float)
        spec_own = prev_gmp / (1.0 + prev_cost / 100.0)
        gen_cost_pct = (gen_mean_per_mouse - spec_own) / spec_own * 100.0
        print(f'  Generalist cost (mean-of-ratios): {gen_cost_pct.mean():.1f}%')
    else:
        print('  [WARN] No GPU: fitness/cost carried forward from existing pkl '
              '(sensitivity/topology updated). Re-run on GPU to refresh cost.')

    A6_results = {
        'N_REPS_G': N_GEN,
        'W_gen': W_gen,
        'gen_topo_sims': gen_topo_sims,
        'spec_topo_sims': spec_topo_sims,
        'mw_p_gen_spec_topo': float(mw_p_gen_spec_topo),
        'gen_fitness_matrix': A6_gen_fitness,
        'gen_mean_per_mouse': gen_mean_per_mouse,
        'gen_cost_pct': gen_cost_pct,
        'gen_sensitivity': gen_sensitivity,
        'gen_sensitivity_norm': gen_sensitivity_norm,
        'gen_full_circuit_baselines': gen_full_circuit_baselines,
        'spec_sens_var': spec_sens_var,
        'gen_sens_var': gen_sens_var,
        'gen_sens_var_norm': gen_sens_var_norm,
    }
    with open(a6_path, 'wb') as f:
        pickle.dump(A6_results, f)
    print(f'  Saved -> {a6_path}  (N_REPS_G={N_GEN})')
    return A6_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

ANALYSES = {
    'A1': run_A1,
    'A2': run_A2,
    'A3': run_A3,
    'A4': run_A4,
    'A5': run_A5,
    'A6': run_A6,
    'A6d': run_A6_degeneracy,
}


def main():
    parser = argparse.ArgumentParser(description='Run supplementary analyses')
    parser.add_argument('--run', default='all', choices=list(ANALYSES.keys()) + ['all'],
                        help='Which analysis to run (default: all)')
    args = parser.parse_args()

    to_run = list(ANALYSES.keys()) if args.run == 'all' else [args.run]
    print(f'Running analyses: {to_run}')

    for key in to_run:
        try:
            ANALYSES[key]()
        except Exception as e:
            print(f'  [ERROR] {key} failed: {e}')
            import traceback; traceback.print_exc()

    print('\nDone.')


if __name__ == '__main__':
    main()
