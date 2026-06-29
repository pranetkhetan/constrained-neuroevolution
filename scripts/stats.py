#!/usr/bin/env python
"""
Statistical tests for circuit analysis.

Provides:
- One-way ANOVA with Benjamini-Hochberg FDR correction
- Permutation tests (non-parametric alternative to ANOVA)
- Bootstrap confidence intervals
- Evolved vs. random comparison (one-sample t-test + Cohen's d)

Usage:
    import stats
    results = stats.anova_with_fdr(df, group_col='mouse', features=FEATURE_NAMES)
    perm_results = stats.permutation_test(df, group_col='mouse', features=FEATURE_NAMES)
"""
import numpy as np
from scipy import stats as sp_stats


def anova_with_fdr(data, group_col, features, alpha=0.05):
    """
    One-way ANOVA per feature with Benjamini-Hochberg FDR correction.

    Args:
        data: list of dicts (each dict has group_col and feature keys)
        group_col: key for group labels (e.g. 'mouse')
        features: list of feature names to test
        alpha: significance level for FDR

    Returns:
        dict of feature -> {F, p_raw, p_fdr, significant, eta_squared}
    """
    # Group data
    groups = {}
    for row in data:
        g = row[group_col]
        if g not in groups:
            groups[g] = []
        groups[g].append(row)

    group_names = sorted(groups.keys())

    results = {}
    p_values = []

    for feat in features:
        group_arrays = []
        for g in group_names:
            vals = [row[feat] for row in groups[g]]
            group_arrays.append(vals)

        # Skip constant features (zero variance → undefined F)
        all_check = np.concatenate(group_arrays)
        if np.ptp(all_check) == 0:
            results[feat] = {'F': np.nan, 'p_raw': np.nan, 'eta_squared': 0.0}
            p_values.append(np.nan)
            continue

        # One-way ANOVA
        F, p = sp_stats.f_oneway(*group_arrays)

        # Eta-squared effect size
        all_vals = np.concatenate(group_arrays)
        grand_mean = np.mean(all_vals)
        ss_between = sum(len(a) * (np.mean(a) - grand_mean)**2 for a in group_arrays)
        ss_total = np.sum((all_vals - grand_mean)**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

        results[feat] = {
            'F': float(F),
            'p_raw': float(p),
            'eta_squared': float(eta_sq),
        }
        p_values.append(float(p))

    # Benjamini-Hochberg FDR correction
    p_fdr = _benjamini_hochberg(p_values)
    for i, feat in enumerate(features):
        results[feat]['p_fdr'] = p_fdr[i]
        results[feat]['significant'] = p_fdr[i] < alpha

    return results


def _benjamini_hochberg(p_values):
    """Apply Benjamini-Hochberg FDR correction to a list of p-values."""
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values and track original indices (NaNs go last)
    indexed = sorted(enumerate(p_values), key=lambda x: (np.isnan(x[1]), x[1]))
    corrected = [0.0] * n

    # BH: p_adj[i] = p[i] * n / rank  (NaNs stay NaN)
    prev = 1.0
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        if np.isnan(p):
            corrected[orig_idx] = np.nan
            continue
        rank = rank_idx + 1
        adjusted = min(p * n / rank, prev)
        adjusted = min(adjusted, 1.0)
        corrected[orig_idx] = adjusted
        prev = adjusted

    return corrected


def permutation_test(data, group_col, features, n_perms=10000, seed=42):
    """
    Permutation test for each feature (non-parametric alternative to ANOVA).

    Shuffles group labels and builds a null F-statistic distribution.

    Args:
        data: list of dicts
        group_col: key for group labels
        features: list of feature names
        n_perms: number of permutations
        seed: random seed

    Returns:
        dict of feature -> {F_obs, p_perm}
    """
    rng = np.random.RandomState(seed)

    groups = {}
    for row in data:
        g = row[group_col]
        if g not in groups:
            groups[g] = []
        groups[g].append(row)

    group_names = sorted(groups.keys())
    group_sizes = [len(groups[g]) for g in group_names]

    results = {}
    for feat in features:
        # Observed F
        group_arrays = [np.array([row[feat] for row in groups[g]]) for g in group_names]

        # Skip constant features
        all_vals = np.concatenate(group_arrays)
        if np.ptp(all_vals) == 0:
            results[feat] = {'F_obs': np.nan, 'p_perm': np.nan}
            continue

        F_obs, _ = sp_stats.f_oneway(*group_arrays)
        n_total = len(all_vals)

        # Permutation null distribution
        n_exceed = 0
        for _ in range(n_perms):
            perm = rng.permutation(n_total)
            perm_arrays = []
            start = 0
            for size in group_sizes:
                perm_arrays.append(all_vals[perm[start:start + size]])
                start += size
            F_perm, _ = sp_stats.f_oneway(*perm_arrays)
            if F_perm >= F_obs:
                n_exceed += 1

        p_perm = (n_exceed + 1) / (n_perms + 1)  # +1 for observed
        results[feat] = {
            'F_obs': float(F_obs),
            'p_perm': float(p_perm),
        }

    return results


def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=42):
    """
    Bootstrap confidence interval for the mean.

    Args:
        values: array-like of observations
        n_boot: number of bootstrap resamples
        ci: confidence level (default 0.95)
        seed: random seed

    Returns:
        (mean, lower, upper)
    """
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.randint(0, n, size=n)]
        boot_means[i] = np.mean(sample)

    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (float(np.mean(values)), float(lower), float(upper))


def evolved_vs_random(evolved_data, random_baseline_data, features):
    """
    Test whether evolved features differ from random baseline.

    Uses one-sample t-test (evolved values vs random mean) and Cohen's d.

    Args:
        evolved_data: list of dicts (evolved agents' features)
        random_baseline_data: dict with 'mean' (dict) and 'all' (list of dicts)
        features: list of feature names

    Returns:
        dict of feature -> {evolved_mean, evolved_ci, random_mean, t, p, cohens_d}
    """
    random_mean = random_baseline_data['mean']
    random_all = random_baseline_data['all']

    results = {}
    for feat in features:
        evolved_vals = np.array([r[feat] for r in evolved_data])
        random_vals = np.array([r[feat] for r in random_all])

        # Skip constant features
        if np.ptp(evolved_vals) == 0 and np.ptp(random_vals) == 0:
            results[feat] = {
                'evolved_mean': float(evolved_vals[0]),
                'evolved_ci': (float(evolved_vals[0]), float(evolved_vals[0])),
                'random_mean': float(random_mean[feat]),
                't': np.nan, 'p': np.nan, 'cohens_d': 0.0,
            }
            continue

        e_mean, e_lo, e_hi = bootstrap_ci(evolved_vals)

        # One-sample t-test: evolved vs random mean
        t_stat, p_val = sp_stats.ttest_1samp(evolved_vals, random_mean[feat])

        # Cohen's d
        pooled_std = np.sqrt(
            (np.var(evolved_vals, ddof=1) + np.var(random_vals, ddof=1)) / 2
        )
        cohens_d = (np.mean(evolved_vals) - random_mean[feat]) / pooled_std if pooled_std > 0 else 0.0

        results[feat] = {
            'evolved_mean': float(e_mean),
            'evolved_ci': (float(e_lo), float(e_hi)),
            'random_mean': float(random_mean[feat]),
            't': float(t_stat),
            'p': float(p_val),
            'cohens_d': float(cohens_d),
        }

    return results


def print_anova_results(results, title="ANOVA Results"):
    """Pretty-print ANOVA results table."""
    print(f"\n{title}")
    print(f"{'Feature':<18} {'F':>8} {'p_raw':>10} {'p_fdr':>10} {'eta2':>8} {'Sig':>5}")
    print("-" * 62)
    for feat, r in sorted(results.items(), key=lambda x: x[1]['p_fdr']):
        sig = '*' if r['significant'] else ''
        print(f"{feat:<18} {r['F']:>8.2f} {r['p_raw']:>10.4f} {r['p_fdr']:>10.4f} "
              f"{r['eta_squared']:>8.3f} {sig:>5}")


def print_evolved_vs_random(results, title="Evolved vs Random"):
    """Pretty-print evolved vs random comparison."""
    print(f"\n{title}")
    print(f"{'Feature':<18} {'Evolved':>10} {'Random':>10} {'Cohen d':>10} {'p':>10}")
    print("-" * 62)
    for feat, r in results.items():
        print(f"{feat:<18} {r['evolved_mean']:>10.3f} {r['random_mean']:>10.3f} "
              f"{r['cohens_d']:>10.2f} {r['p']:>10.4f}")


# ═══════════════════════════════════════════════════════════════════════
# POST-HOC POWER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def post_hoc_power(anova_results, n_total, k_groups, alpha=0.05):
    """
    Post-hoc power analysis for one-way ANOVA results.

    Uses observed eta-squared to compute the non-centrality parameter
    and achieved power via the non-central F distribution.

    Args:
        anova_results: dict from anova_with_fdr()
        n_total: total sample size (e.g. 54)
        k_groups: number of groups (e.g. 9)
        alpha: significance threshold

    Returns:
        dict of feature -> {eta_squared, f_squared, ncp, power,
                            min_detectable_eta2}
    """
    from scipy.stats import f as f_dist

    df1 = k_groups - 1
    df2 = n_total - k_groups
    f_crit = f_dist.ppf(1 - alpha, df1, df2)

    results = {}
    for feat, r in anova_results.items():
        eta2 = r.get('eta_squared', 0.0)
        if np.isnan(eta2) or eta2 <= 0:
            results[feat] = {
                'eta_squared': eta2, 'f_squared': 0.0,
                'ncp': 0.0, 'power': alpha,
                'min_detectable_eta2': None,
            }
            continue

        # Convert eta² → Cohen's f² → non-centrality parameter
        f_sq = eta2 / (1 - eta2)
        ncp = f_sq * n_total

        # Power = P(F > f_crit | H1) under non-central F
        power = 1 - f_dist.cdf(f_crit, df1, df2, loc=0, scale=1)
        # Non-central F: use ncf
        from scipy.stats import ncf
        power = 1 - ncf.cdf(f_crit, df1, df2, ncp)

        results[feat] = {
            'eta_squared': float(eta2),
            'f_squared': float(f_sq),
            'ncp': float(ncp),
            'power': float(power),
        }

    # Minimum detectable eta² at 80% power
    # Binary search for eta2 where power = 0.8
    from scipy.stats import ncf
    lo, hi = 0.001, 0.99
    for _ in range(50):
        mid = (lo + hi) / 2
        f_sq_mid = mid / (1 - mid)
        ncp_mid = f_sq_mid * n_total
        pw = 1 - ncf.cdf(f_crit, df1, df2, ncp_mid)
        if pw < 0.8:
            lo = mid
        else:
            hi = mid
    min_eta2 = (lo + hi) / 2

    for feat in results:
        results[feat]['min_detectable_eta2'] = float(min_eta2)

    return results


def print_power_results(results, title="Post-Hoc Power Analysis"):
    """Pretty-print power analysis results."""
    print(f"\n{title}")
    print(f"{'Feature':<18} {'eta2':>8} {'f2':>8} {'Power':>8} {'Min eta2':>10}")
    print("-" * 56)
    for feat, r in sorted(results.items(), key=lambda x: -x[1].get('power', 0)):
        print(f"{feat:<18} {r['eta_squared']:>8.3f} {r['f_squared']:>8.3f} "
              f"{r['power']:>8.3f} {r['min_detectable_eta2']:>10.3f}")


# ═══════════════════════════════════════════════════════════════════════
# STRAIN CLUSTERING (B vs D)
# ═══════════════════════════════════════════════════════════════════════

def strain_clustering_test(generalization_matrix, mice):
    """
    Test whether B-strain mice cluster separately from D-strain in the
    generalization matrix.

    Compares within-strain off-diagonal fitness to between-strain fitness
    using a Mann-Whitney U test.

    Args:
        generalization_matrix: (N, N) fitness array (lower = better)
        mice: list of mouse IDs in matrix order

    Returns:
        dict with within_B, within_D, between, U_stat, p_value, etc.
    """
    N = len(mice)
    strain = np.array(['B' if m.startswith('B') else 'D' for m in mice])

    within_same = []
    between = []
    within_B = []
    within_D = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            val = generalization_matrix[i, j]
            if strain[i] == strain[j]:
                within_same.append(val)
                if strain[i] == 'B':
                    within_B.append(val)
                else:
                    within_D.append(val)
            else:
                between.append(val)

    within_same = np.array(within_same)
    between = np.array(between)
    within_B = np.array(within_B)
    within_D = np.array(within_D)

    # Mann-Whitney: within-strain vs between-strain
    # Lower fitness = better, so within-strain should be lower if clustering
    U, p = sp_stats.mannwhitneyu(within_same, between, alternative='less')

    return {
        'within_same_mean': float(within_same.mean()),
        'within_same_std': float(within_same.std()),
        'within_B_mean': float(within_B.mean()) if len(within_B) > 0 else None,
        'within_D_mean': float(within_D.mean()) if len(within_D) > 0 else None,
        'between_mean': float(between.mean()),
        'between_std': float(between.std()),
        'U_stat': float(U),
        'p_value': float(p),
        'n_within': len(within_same),
        'n_between': len(between),
    }


# ═══════════════════════════════════════════════════════════════════════
# WEIGHT-LEVEL FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_weight_features(agent):
    """
    Extract per-pathway weight magnitude features from an agent.

    These go beyond the 18 structural features (which are counts/fractions)
    to capture the actual synaptic strengths, which may differ across mice
    even when topology is identical.

    Returns:
        dict with ~6 features: per-pathway mean absolute weight
    """
    W = np.asarray(agent.weights) if not hasattr(agent.weights, 'get') \
        else agent.weights.get()

    idx_s = agent.idx_sensory
    idx_i = agent.idx_inter
    idx_m = agent.idx_motor

    def pathway_w_mean(src, dst):
        sub = W[np.ix_(src, dst)]
        active = sub[sub != 0]
        return float(np.mean(np.abs(active))) if len(active) > 0 else 0.0

    return {
        'w_si': pathway_w_mean(idx_s, idx_i),
        'w_sm': pathway_w_mean(idx_s, idx_m),
        'w_ii': pathway_w_mean(idx_i, idx_i),
        'w_im': pathway_w_mean(idx_i, idx_m),
        'w_mi': pathway_w_mean(idx_m, idx_i),
        'w_mm': pathway_w_mean(idx_m, idx_m),
    }

WEIGHT_FEATURE_NAMES = ['w_si', 'w_sm', 'w_ii', 'w_im', 'w_mi', 'w_mm']


def weight_vector_similarity(agents_data):
    """
    Compute pairwise cosine similarity of full weight vectors.

    Args:
        agents_data: list of dicts, each with 'agent' key containing Agent object

    Returns:
        (N, N) cosine similarity matrix, list of labels
    """
    vectors = []
    labels = []
    for row in agents_data:
        W = np.asarray(row['agent'].weights) if not hasattr(row['agent'].weights, 'get') \
            else row['agent'].weights.get()
        vectors.append(W.ravel())
        labels.append(f"{row['mouse']}/r{row['rep']}")

    vectors = np.array(vectors)
    # Cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = vectors / norms
    sim_matrix = normed @ normed.T
    return sim_matrix, labels


# ═══════════════════════════════════════════════════════════════════════
# COHEN'S d BOOTSTRAP CI
# ═══════════════════════════════════════════════════════════════════════

def cohens_d_bootstrap_ci(evolved_vals, random_vals, n_boot=10000, ci=0.95, seed=42):
    """
    Bootstrap CI for Cohen's d (evolved vs random).

    Args:
        evolved_vals: array of evolved feature values
        random_vals: array of random baseline feature values
        n_boot: bootstrap iterations
        ci: confidence level
        seed: random seed

    Returns:
        (d, d_lo, d_hi)
    """
    rng = np.random.RandomState(seed)
    evolved_vals = np.asarray(evolved_vals, dtype=float)
    random_vals = np.asarray(random_vals, dtype=float)
    n_e, n_r = len(evolved_vals), len(random_vals)

    def _cohens_d(e, r):
        pooled_std = np.sqrt((np.var(e, ddof=1) + np.var(r, ddof=1)) / 2)
        return (np.mean(e) - np.mean(r)) / pooled_std if pooled_std > 0 else 0.0

    d_obs = _cohens_d(evolved_vals, random_vals)

    boot_ds = np.empty(n_boot)
    for i in range(n_boot):
        e_sample = evolved_vals[rng.randint(0, n_e, size=n_e)]
        r_sample = random_vals[rng.randint(0, n_r, size=n_r)]
        boot_ds[i] = _cohens_d(e_sample, r_sample)

    alpha = 1 - ci
    d_lo = np.percentile(boot_ds, 100 * alpha / 2)
    d_hi = np.percentile(boot_ds, 100 * (1 - alpha / 2))
    return (float(d_obs), float(d_lo), float(d_hi))


# ═══════════════════════════════════════════════════════════════════════
# CONNECTION-LEVEL ANOVA
# ═══════════════════════════════════════════════════════════════════════

def connection_level_anova(agents_data, alpha=0.05):
    """
    Per-connection (i→j) ANOVA across mice.

    Tests whether specific connections are present more/less often in
    agents evolved for different mice, even when pathway-level counts
    are identical.

    Args:
        agents_data: list of dicts with 'agent' and 'mouse' keys
        alpha: significance threshold for FDR

    Returns:
        dict with:
          'p_matrix': (14,14) FDR-corrected p-values
          'f_matrix': (14,14) F-statistics
          'presence': dict[mouse -> (14,14) fraction of agents with connection]
          'n_significant': count of significant connections
          'significant_connections': list of (i, j, F, p_fdr)
    """
    # Group agents by mouse
    mice_agents = {}
    for row in agents_data:
        m = row['mouse']
        if m not in mice_agents:
            mice_agents[m] = []
        W = np.asarray(row['agent'].weights) if not hasattr(row['agent'].weights, 'get') \
            else row['agent'].weights.get()
        mice_agents[m].append((W != 0).astype(float))

    mice = sorted(mice_agents.keys())
    n_neurons = mice_agents[mice[0]][0].shape[0]

    # Presence fraction per mouse
    presence = {}
    for m in mice:
        mats = np.array(mice_agents[m])
        presence[m] = mats.mean(axis=0)

    # Per-connection ANOVA
    f_matrix = np.full((n_neurons, n_neurons), np.nan)
    p_raw_list = []
    conn_indices = []

    for i in range(n_neurons):
        for j in range(n_neurons):
            groups = []
            for m in mice:
                vals = [mat[i, j] for mat in mice_agents[m]]
                groups.append(vals)

            all_vals = np.concatenate(groups)
            if np.ptp(all_vals) == 0:
                p_raw_list.append(np.nan)
                conn_indices.append((i, j))
                continue

            F, p = sp_stats.f_oneway(*groups)
            f_matrix[i, j] = F
            p_raw_list.append(p)
            conn_indices.append((i, j))

    # FDR correction
    p_fdr = _benjamini_hochberg(p_raw_list)
    p_matrix = np.full((n_neurons, n_neurons), np.nan)
    for idx, (i, j) in enumerate(conn_indices):
        p_matrix[i, j] = p_fdr[idx]

    # Significant connections
    sig_conns = []
    for idx, (i, j) in enumerate(conn_indices):
        if not np.isnan(p_fdr[idx]) and p_fdr[idx] < alpha:
            sig_conns.append((i, j, float(f_matrix[i, j]), float(p_fdr[idx])))
    sig_conns.sort(key=lambda x: x[3])

    return {
        'p_matrix': p_matrix,
        'f_matrix': f_matrix,
        'presence': presence,
        'n_significant': len(sig_conns),
        'significant_connections': sig_conns,
        'mice': mice,
    }


# ═══════════════════════════════════════════════════════════════════════
# REQUIRED REPLICATES ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def required_replicates(anova_results, k_groups=9, alpha=0.05, target_power=0.8):
    """
    Compute how many replicates per group are needed for target power
    at each observed effect size.

    Args:
        anova_results: dict from anova_with_fdr()
        k_groups: number of groups
        alpha: significance level
        target_power: desired power (default 0.8)

    Returns:
        dict of feature -> {eta_squared, required_n_per_group, required_n_total}
    """
    from scipy.stats import ncf, f as f_dist

    results = {}
    for feat, r in anova_results.items():
        eta2 = r.get('eta_squared', 0.0)
        if np.isnan(eta2) or eta2 <= 0:
            results[feat] = {
                'eta_squared': eta2,
                'required_n_per_group': np.inf,
                'required_n_total': np.inf,
            }
            continue

        f_sq = eta2 / (1 - eta2)

        # Binary search for n_per_group
        lo, hi = 2, 500
        for _ in range(50):
            mid = (lo + hi) // 2
            n_total = mid * k_groups
            df1 = k_groups - 1
            df2 = n_total - k_groups
            ncp = f_sq * n_total
            f_crit = f_dist.ppf(1 - alpha, df1, df2)
            pw = 1 - ncf.cdf(f_crit, df1, df2, ncp)
            if pw < target_power:
                lo = mid + 1
            else:
                hi = mid

        required_n = hi
        results[feat] = {
            'eta_squared': float(eta2),
            'required_n_per_group': int(required_n),
            'required_n_total': int(required_n * k_groups),
        }

    return results
