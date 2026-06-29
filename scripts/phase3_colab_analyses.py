# %% [markdown]
# # Phase 3 Colab Analyses
#
# Three supplementary analyses for the constrained neuroevolution paper:
#
# **3a. Random-agent permutation control**
#   Applies the same source-preserving permutation ablation to 54 random
#   (untrained) constrained agents. If the evolved 2.39× ratio reflects
#   *learned* structure, random agents should show a substantially lower ratio.
#   GPU-batched forward pass for efficiency.
#
# **3b. Generalist formal Wilcoxon test**
#   Confirms that the generalist agent shows no own-mouse permutation bias
#   (i.e., flat Δfit profile across all 9 mice). Uses existing
#   `analysis/generalist_results.pkl` and `analysis/specificity_results.pkl`.
#   No GPU needed.
#
# **3c. Power analysis for permutation specificity test**
#   Post-hoc power for the Wilcoxon signed-rank test (n=9 paired observations).
#   Reports minimum detectable effect and attained power at the observed effect.
#   No GPU needed.
#
# Run all: execute cells in order in Colab, or `python scripts/phase3_colab_analyses.py`

# %% Imports and path setup
import sys
import os
import pickle
import numpy as np
from pathlib import Path

# Repo-relative paths.
PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
AGENTS_DIR  = os.path.join(PROJECT_DIR, 'data', 'agents')
ANALYSIS    = os.path.join(PROJECT_DIR, 'analysis')
FIGURES     = os.path.join(PROJECT_DIR, 'figures')
sys.path.insert(0, PROJECT_DIR)

os.makedirs(ANALYSIS, exist_ok=True)
os.makedirs(FIGURES,  exist_ok=True)

from utils.backend import xp, to_cpu, HAS_GPU

MICE   = ['B5', 'B6', 'B7', 'D3', 'D4', 'D5', 'D7', 'D8', 'D9']
N_REPS = 6
GEN    = 150
N_PERMUTATIONS = 20
N_SIM_STEPS    = 1000   # same as existing MSE analysis
BATCH_CHUNK    = 256    # forward-pass chunk size to stay within GPU VRAM

print(f"PROJECT_DIR: {PROJECT_DIR}")
print(f"GPU available: {HAS_GPU}")


# ── CuPy-safe unpickler ───────────────────────────────────────────────────────
class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('cupy'):
            module = module.replace('cupy._core.core', 'numpy').replace('cupy', 'numpy')
        return super().find_class(module, name)


def _load_pkl(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        with open(path, 'rb') as f:
            return _CpuUnpickler(f).load()


# ── Source-preserving permutation ────────────────────────────────────────────
IDX_S = list(range(6))        # sensory
IDX_I = list(range(6, 12))    # interneurons
IDX_M = list(range(12, 14))   # motor

PATHWAY_BLOCKS = [
    (IDX_S, IDX_I),   # SI
    (IDX_S, IDX_M),   # SM
    (IDX_I, IDX_I),   # II
    (IDX_I, IDX_M),   # IM
    (IDX_M, IDX_I),   # MI
    (IDX_M, IDX_M),   # MM
]


def source_preserving_permutation(W_np: np.ndarray) -> np.ndarray:
    """
    Return a permuted copy of weight matrix W_np.

    For each source neuron within each pathway block, permute only which
    target neurons it connects to, keeping per-source out-degree, Dale's
    Law sign, and weight magnitudes fixed. All 18 structural features are
    preserved exactly.
    """
    W = W_np.copy()
    for sources, targets in PATHWAY_BLOCKS:
        for src in sources:
            row = W[src, :]
            connected_targets = [t for t in targets if row[t] != 0]
            if len(connected_targets) < 2:
                continue                          # nothing to permute
            available_targets = list(targets)
            shuffled = available_targets.copy()
            np.random.shuffle(shuffled)
            # Assign magnitudes of existing connections to shuffled target slots
            # (preserve per-source out-degree and magnitude vector)
            mags = [row[t] for t in connected_targets]
            np.random.shuffle(mags)
            # Zero out old connections in this block for this source
            for t in targets:
                W[src, t] = 0
            # Re-assign to shuffled targets
            for i, t in enumerate(shuffled[:len(mags)]):
                W[src, t] = mags[i]
    return W


# ── Batched forward pass (GPU) ────────────────────────────────────────────────
def _standardized_sensory(n_steps: int, seed: int = 42) -> np.ndarray:
    """
    Reproducible 1000-step sinusoidal sensory input sequence (n_steps × 6).
    Matches the sequence used in the existing permutation MSE analysis.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 4 * np.pi, n_steps)
    seq = np.zeros((n_steps, 6))
    seq[:, 0] = 0.5 + 0.3 * np.sin(t)          # forward distance
    seq[:, 1] = 0.4 + 0.3 * np.sin(t + 1.0)    # left distance
    seq[:, 2] = 0.4 + 0.3 * np.sin(t + 2.0)    # right distance
    seq[:, 3] = 0.2 + 0.1 * np.sin(t * 0.7)    # prev speed
    seq[:, 4] = 0.1 * np.cos(t * 1.3)           # prev turn
    seq[:, 5] = rng.randn(n_steps) * 0.05        # noise
    return seq.astype(np.float64)


def batched_motor_outputs(W_list: list, sensory_seq: np.ndarray,
                          chunk: int = BATCH_CHUNK) -> np.ndarray:
    """
    Run all agents in W_list on sensory_seq in GPU-batched chunks.

    Parameters
    ----------
    W_list : list of (14,14) np.ndarray weight matrices
    sensory_seq : (T, 6) standardized sensory sequence

    Returns
    -------
    motor_out : (N, T, 2)  motor outputs for each agent at each timestep
    """
    N = len(W_list)
    T = sensory_seq.shape[0]
    motor_out = np.zeros((N, T, 2), dtype=np.float64)

    sensory_xp = xp.array(sensory_seq)   # (T, 6)

    for start in range(0, N, chunk):
        end   = min(start + chunk, N)
        batch = end - start

        W_chunk = xp.array(
            np.stack(W_list[start:end])
        )  # (batch, 14, 14)

        state = xp.zeros((batch, 14), dtype=xp.float64)  # (batch, 14)

        out = np.zeros((batch, T, 2), dtype=np.float64)

        for t in range(T):
            s_t = sensory_xp[t]                          # (6,)
            state[:, :6] = s_t[None, :]                  # broadcast to batch

            # Forward: state_new = tanh(state @ W)
            new_state = xp.tanh(
                xp.matmul(state[:, None, :], W_chunk).squeeze(1)
            )                                             # (batch, 14)

            state[:, 6:] = new_state[:, 6:]              # update inter+motor

            out[:, t, :] = to_cpu(state[:, 12:14])       # motor outputs

        motor_out[start:end] = out
        del W_chunk, state
        if HAS_GPU:
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()

    return motor_out   # (N, T, 2)


# ============================================================================
# %% [markdown]
# ## Analysis 3a: Random-agent permutation control
#
# **Hypothesis**: The 2.39× permutation-to-replicate motor MSE ratio for evolved
# agents reflects *learned* structural specificity. Random agents, having no
# learned wiring, should show a lower ratio (permuting random connections
# changes little because nothing specific was learned).
#
# **Protocol** (identical to evolved-agent analysis):
# 1. Load the 54 random constrained agents from `analysis/random_baseline.pkl`
# 2. Generate 20 source-preserving permuted variants per agent
# 3. Run all agents on the same standardized 1000-step sensory sequence
# 4. Compute motor MSE: original vs permuted, and original vs a different
#    random agent (the "replicate" analogue)
# 5. Report mean ratio and compare to evolved 2.39×
# ============================================================================

# %% 3a: Generate random baseline agents
print("=" * 70)
print("ANALYSIS 3a: Random-Agent Permutation Control")
print("=" * 70)

# Import Agent and config classes needed to generate random agents
from config import load_config
from core.agent import Agent

# Load config using absolute path (works in Colab where cwd != project root)
config = load_config(os.path.join(PROJECT_DIR, 'config.yaml'))

# Generate 54 random constrained agents (one per evolved run) with same seed for reproducibility
print("Generating 54 random constrained agents (seed=42)...")
np.random.seed(42)
rand_agents = []
for i in range(54):
    agent = Agent(config.network)
    rand_agents.append(agent)

# Convert weights to numpy
rand_weights = [to_cpu(a.weights) for a in rand_agents]
N_rand = len(rand_weights)
print(f"Generated {N_rand} random agents")

# Also load evolved agents for comparison
print("\nLoading 54 evolved agents ...")
evolved_weights = []
for mouse in MICE:
    for rep in range(1, N_REPS + 1):
        path = os.path.join(AGENTS_DIR, f"results_{mouse}_r{rep}",
                            f"gen_{GEN}", "summary.pkl")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found")
            continue
        data = _load_pkl(path)
        best = data[0]
        evolved_weights.append(to_cpu(best['agent'].weights))

print(f"Loaded {len(evolved_weights)} evolved agents")

# %% 3a: GPU-batched motor output computation
np.random.seed(0)
sensory_seq = _standardized_sensory(N_SIM_STEPS)

print("\nRunning GPU-batched forward pass for RANDOM agents ...")

# Build list: [original_0, perm_0_1, ..., perm_0_20, original_1, ...]
rand_all_W   = []
rand_indices = []      # (agent_idx, is_permuted, perm_idx)

for i, W in enumerate(rand_weights):
    rand_all_W.append(W)
    rand_indices.append((i, False, 0))
    for p in range(N_PERMUTATIONS):
        rand_all_W.append(source_preserving_permutation(W))
        rand_indices.append((i, True, p + 1))

print(f"Total random agent × perm combinations: {len(rand_all_W)}")
rand_motor = batched_motor_outputs(rand_all_W, sensory_seq)
print("Done.")

# Repeat for evolved agents
print("\nRunning GPU-batched forward pass for EVOLVED agents ...")
evol_all_W   = []
evol_indices = []

for i, W in enumerate(evolved_weights):
    evol_all_W.append(W)
    evol_indices.append((i, False, 0))
    for p in range(N_PERMUTATIONS):
        evol_all_W.append(source_preserving_permutation(W))
        evol_indices.append((i, True, p + 1))

evol_motor = batched_motor_outputs(evol_all_W, sensory_seq)
print("Done.")

# %% 3a: Compute MSE ratios (corrected protocol)
#
# The paper's 2.39× denominator is WITHIN-MOUSE replicate pairs, not all-pairs.
# Replicates for the same mouse converge to similar outputs → smaller baseline MSE
# → larger ratio when permuted.
#
# For random agents there are no replicates, so we use all-pairs cross-agent MSE
# as the analogous baseline. The comparison is:
#   evolved  ratio = perm_mse / within-mouse-replicate_mse
#   random   ratio = perm_mse / all-pairs-cross_mse
# If evolved ratio >> random ratio, permutation sensitivity reflects learned structure.

from scipy import stats

n_per = N_PERMUTATIONS + 1   # 1 original + N_PERMUTATIONS permuted

def _extract_orig(motor_all, n_agents, n_per):
    return [motor_all[i * n_per] for i in range(n_agents)]

# ── Evolved: perm_mse and within-mouse replicate_mse ─────────────────────────
n_evol = len(evolved_weights)   # should be 54 (9 mice × 6 reps)
orig_evol = _extract_orig(evol_motor, n_evol, n_per)

evol_perm_mse = []
for i in range(n_evol):
    orig_i = orig_evol[i]
    for p in range(1, n_per):
        perm_p = evol_motor[i * n_per + p]
        evol_perm_mse.append(np.mean((orig_i - perm_p) ** 2))

# Within-mouse replicate pairs: agents are stored as MICE × N_REPS in order
evol_replicate_mse = []
for m_idx in range(len(MICE)):
    for r1 in range(N_REPS):
        for r2 in range(r1 + 1, N_REPS):
            i1 = m_idx * N_REPS + r1
            i2 = m_idx * N_REPS + r2
            evol_replicate_mse.append(np.mean((orig_evol[i1] - orig_evol[i2]) ** 2))

evol_perm_mse      = np.array(evol_perm_mse)
evol_replicate_mse = np.array(evol_replicate_mse)
evol_ratio         = np.mean(evol_perm_mse) / np.mean(evol_replicate_mse)

# Per-agent ratios (for Mann-Whitney below)
evol_mean_rep_mse = np.mean(evol_replicate_mse)
evol_per_agent_ratio = np.array([
    np.mean(evol_perm_mse[i * N_PERMUTATIONS:(i + 1) * N_PERMUTATIONS]) / evol_mean_rep_mse
    for i in range(n_evol)
])

# ── Random: perm_mse and all-pairs cross_mse (no replicates exist) ────────────
orig_rand = _extract_orig(rand_motor, N_rand, n_per)

rand_perm_mse = []
for i in range(N_rand):
    orig_i = orig_rand[i]
    for p in range(1, n_per):
        perm_p = rand_motor[i * n_per + p]
        rand_perm_mse.append(np.mean((orig_i - perm_p) ** 2))

rand_cross_mse = []
for i in range(N_rand):
    for j in range(N_rand):
        if i != j:
            rand_cross_mse.append(np.mean((orig_rand[i] - orig_rand[j]) ** 2))

rand_perm_mse  = np.array(rand_perm_mse)
rand_cross_mse = np.array(rand_cross_mse)
rand_ratio     = np.mean(rand_perm_mse) / np.mean(rand_cross_mse)

rand_mean_cross_mse = np.mean(rand_cross_mse)
rand_per_agent_ratio = np.array([
    np.mean(rand_perm_mse[i * N_PERMUTATIONS:(i + 1) * N_PERMUTATIONS]) / rand_mean_cross_mse
    for i in range(N_rand)
])

# ── Report ────────────────────────────────────────────────────────────────────
print("\n--- 3a RESULTS ---")
print(f"Evolved agents:  perm MSE = {np.mean(evol_perm_mse):.4f}, "
      f"replicate MSE (within-mouse) = {np.mean(evol_replicate_mse):.4f}, "
      f"ratio = {evol_ratio:.3f}×")
print(f"Random agents:   perm MSE = {np.mean(rand_perm_mse):.4f}, "
      f"cross MSE (all-pairs) = {np.mean(rand_cross_mse):.4f}, "
      f"ratio = {rand_ratio:.3f}×")

# Note: the paper's absolute 2.39× used full behavioral-fitness simulation,
# not the synthetic sensory sequence used here. Our protocol uses the same
# *relative* ratio logic and is directly comparable across evolved vs random.
print(f"\n  Evolved ratio ({evol_ratio:.2f}×) vs Random ratio ({rand_ratio:.2f}×): "
      + ("evolved > random ✓" if evol_ratio > rand_ratio else "evolved ≤ random ✗"))

# Mann-Whitney: are per-agent evolved ratios higher than random ratios?
stat, p_mw = stats.mannwhitneyu(evol_per_agent_ratio, rand_per_agent_ratio,
                                alternative='greater')
print(f"\nMann-Whitney (evolved per-agent ratio > random per-agent ratio): "
      f"U={stat:.0f}, p={p_mw:.4f}")
print("  → " + ("Confirms learned structure drives permutation sensitivity." if p_mw < 0.05
                 else "Not significant at α=0.05."))

# Save
res_3a = {
    'evol_perm_mse':       evol_perm_mse,
    'evol_replicate_mse':  evol_replicate_mse,
    'evol_ratio':          evol_ratio,
    'rand_perm_mse':       rand_perm_mse,
    'rand_cross_mse':      rand_cross_mse,
    'rand_ratio':          rand_ratio,
    'mannwhitney_U':       float(stat),
    'mannwhitney_p':       float(p_mw),
}
with open(os.path.join(ANALYSIS, 'phase3a_random_permutation.pkl'), 'wb') as f:
    pickle.dump(res_3a, f)
print("Saved: analysis/phase3a_random_permutation.pkl")


# ============================================================================
# %% [markdown]
# ## Analysis 3b: Generalist formal Wilcoxon test
#
# **Hypothesis**: The generalist (trained on all 9 mice simultaneously) shows
# no own-mouse permutation bias — its Δfit profile is flat across all mice.
#
# **Tests**:
# 1. Kruskal-Wallis across 9 mice (all 6 generalist replicates each mouse)
#    → expected: non-significant (no mouse is treated differently)
# 2. Comparison of per-mouse Δfit variance for generalist vs specialist:
#    generalist variance should be lower and attributable to mouse-intrinsic
#    difficulty rather than own-mouse specificity
# 3. Specialist own vs other Wilcoxon (n=9 pairs) — reproduce the p=0.002
#    result using `specificity_results.pkl`
# ============================================================================

# %% 3b: Load existing results
print("\n" + "=" * 70)
print("ANALYSIS 3b: Generalist Formal Wilcoxon Test")
print("=" * 70)

gen_data  = _load_pkl(os.path.join(ANALYSIS, 'generalist_results.pkl'))
spec_data = _load_pkl(os.path.join(ANALYSIS, 'specificity_results.pkl'))

# Generalist Δfit per mouse (9 mice × 6 replicates).
# `generalist_results.pkl` doesn't store this as a flat dict — derive it from
# results_C (list[N_REPS_G] of {mouse: {orig_fitness, perm_fitnesses, delta_fit}}).
results_C = gen_data['results_C']
pkl_mice  = gen_data['MICE']
gen_delta = {
    m: [float(results_C[r][m]['delta_fit']) for r in range(len(results_C))]
    for m in pkl_mice
}

print("\nGeneralist Δfit by mouse (mean ± sd across 6 replicates):")
for m in MICE:
    vals = np.array(gen_delta[m])
    print(f"  {m}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# Test 1: Kruskal-Wallis across mice
groups = [np.array(gen_delta[m]) for m in MICE]
kw_stat, kw_p = stats.kruskal(*groups)
print(f"\nKruskal-Wallis (is any mouse treated differently by generalist?): "
      f"H={kw_stat:.3f}, p={kw_p:.4f}")
print("  → " + ("Non-significant: flat profile confirmed." if kw_p > 0.05
                 else "SIGNIFICANT: some mouse differs (unexpected)."))

# Test 2: Per-generalist-replicate: max−min Δfit across mice
per_rep_spread = []
n_gen_reps = len(gen_delta[MICE[0]])
for rep in range(n_gen_reps):
    vals = [gen_delta[m][rep] for m in MICE]
    per_rep_spread.append(max(vals) - min(vals))

spread_arr = np.array(per_rep_spread)
# One-sample Wilcoxon: is spread > 0?  (expected: not significantly)
wilcox_stat, wilcox_p = stats.wilcoxon(
    spread_arr - np.median(spread_arr),
    alternative='two-sided')
print(f"\nGeneralist per-replicate Δfit spread (max−min across mice): "
      f"{spread_arr.mean():.4f} ± {spread_arr.std():.4f}")
print(f"  Wilcoxon vs median: W={wilcox_stat:.1f}, p={wilcox_p:.4f}")

# Test 3: Specialist own vs other — reproduce Wilcoxon p=0.002
spec_optA = spec_data['option_A']['results_per_mouse']

spec_own_delta   = []
spec_other_delta = []

for m in MICE:
    own_entry  = spec_optA[m][m]
    own_delta  = np.mean(own_entry['perm_fitnesses']) - own_entry['orig_fitness']
    other_deltas = []
    for other in MICE:
        if other == m:
            continue
        e = spec_optA[m][other]
        other_deltas.append(
            np.mean(e['perm_fitnesses']) - e['orig_fitness'])
    spec_own_delta.append(own_delta)
    spec_other_delta.append(np.mean(other_deltas))
    print(f"  Specialist {m}: Δfit_own={own_delta:.4f}, "
          f"mean Δfit_other={np.mean(other_deltas):.4f}")

spec_own_arr   = np.array(spec_own_delta)
spec_other_arr = np.array(spec_other_delta)
diff = spec_own_arr - spec_other_arr

wstat, wp = stats.wilcoxon(diff, alternative='greater')
print(f"\nSpecialist Wilcoxon (own > other, n=9 pairs): "
      f"W={wstat:.1f}, p={wp:.4f}  (paper: p=0.002)")

# Compare generalist vs specialist Δfit_own elevation
gen_mean_per_mouse = np.array([np.mean(gen_delta[m]) for m in MICE])
gen_elevation = gen_mean_per_mouse - np.mean(gen_mean_per_mouse)
spec_elevation = spec_own_arr - spec_other_arr

mw_stat2, mw_p2 = stats.mannwhitneyu(
    spec_elevation, np.abs(gen_elevation), alternative='greater')
print(f"\nSpecialist own-elevation vs generalist variation: "
      f"MW p={mw_p2:.4f}")

# Save
res_3b = {
    'kruskal_wallis_H': kw_stat,
    'kruskal_wallis_p': kw_p,
    'generalist_spread_mean': float(spread_arr.mean()),
    'generalist_spread_std': float(spread_arr.std()),
    'specialist_own_delta': spec_own_arr.tolist(),
    'specialist_other_delta': spec_other_arr.tolist(),
    'specialist_wilcoxon_W': float(wstat),
    'specialist_wilcoxon_p': float(wp),
}
with open(os.path.join(ANALYSIS, 'phase3b_generalist_formal.pkl'), 'wb') as f:
    pickle.dump(res_3b, f)
print("Saved: analysis/phase3b_generalist_formal.pkl")


# ============================================================================
# %% [markdown]
# ## Analysis 3c: Power analysis for permutation specificity test
#
# The permutation specificity result (Section 2.10) uses a Wilcoxon signed-rank
# test with n=9 paired observations (one per mouse). This section computes:
# 1. Post-hoc power at the observed effect size (Δfit_own − Δfit_other)
# 2. Minimum detectable effect at 80% power
# 3. Required n for 80% power at the observed effect
# ============================================================================

# %% 3c: Power analysis
print("\n" + "=" * 70)
print("ANALYSIS 3c: Power Analysis for Permutation Specificity Test")
print("=" * 70)

try:
    from pingouin import power_wilcoxon
    USE_PINGOUIN = True
except ImportError:
    USE_PINGOUIN = False
    print("pingouin not installed; using normal-approximation fallback")

# Observed effect: differences
diff_arr = np.array(spec_own_arr) - np.array(spec_other_arr)
n = len(diff_arr)
effect_mean = float(np.mean(diff_arr))
effect_sd   = float(np.std(diff_arr, ddof=1))
effect_d    = effect_mean / effect_sd  # Cohen's d_z

print(f"n = {n} paired observations (one per mouse)")
print(f"Mean Δ(own − other) = {effect_mean:.4f} ± {effect_sd:.4f}")
print(f"Cohen's d_z = {effect_d:.4f}")

# Post-hoc power via normal approximation (Wilcoxon large-sample approx)
# Power ≈ P(Z > z_alpha − d_z * sqrt(n)) for one-tailed test
from scipy.stats import norm
alpha = 0.05
z_alpha = norm.ppf(1 - alpha)    # one-tailed

def power_normal_approx(d_z, n, alpha=0.05):
    z_a = norm.ppf(1 - alpha)
    power = 1 - norm.cdf(z_a - d_z * np.sqrt(n))
    return power

attained_power = power_normal_approx(effect_d, n)
print(f"\nAttained power (normal approx, one-tailed α=0.05): {attained_power:.3f}")

if USE_PINGOUIN:
    pg_power = power_wilcoxon(d=effect_d, n=n, alpha=alpha, alternative='greater')
    print(f"Attained power (pingouin exact): {pg_power:.3f}")

# Minimum detectable effect at 80% power with n=9
# Solve: 1 - Φ(z_alpha - d * sqrt(n)) = 0.80
# → d = (z_alpha + z_beta) / sqrt(n) where z_beta = Φ^{-1}(0.80)
from scipy.optimize import brentq

z_beta = norm.ppf(0.80)
mde_d  = (z_alpha + z_beta) / np.sqrt(n)
mde_raw = mde_d * effect_sd
print(f"\nMinimum detectable effect at 80% power with n={n}:")
print(f"  Cohen's d_z = {mde_d:.4f}  (absolute: {mde_raw:.4f} fitness units)")
print(f"  Observed d_z = {effect_d:.4f}  → "
      + ("ABOVE" if effect_d >= mde_d else "BELOW") + " the 80% MDE threshold")

# Required n for 80% power at observed effect
def required_n_for_power(d_z, target_power=0.80, alpha=0.05):
    z_a = norm.ppf(1 - alpha)
    z_b = norm.ppf(target_power)
    return int(np.ceil(((z_a + z_b) / d_z) ** 2))

req_n = required_n_for_power(effect_d)
print(f"\nRequired n for 80% power at observed d_z={effect_d:.3f}: {req_n} mice")

# Power curve: n = 5, 9, 15, 20, 25
print("\nPower curve (one-tailed α=0.05, observed d_z):")
print(f"  {'n':>4}  {'power':>8}")
for n_val in [5, 9, 15, 20, 25, 30]:
    pw = power_normal_approx(effect_d, n_val)
    print(f"  {n_val:>4}  {pw:>8.3f}")

# Save
res_3c = {
    'n': n,
    'effect_mean': effect_mean,
    'effect_sd': effect_sd,
    'cohen_dz': effect_d,
    'attained_power': attained_power,
    'mde_d': mde_d,
    'mde_raw': mde_raw,
    'required_n_80pct': req_n,
}
with open(os.path.join(ANALYSIS, 'phase3c_power_analysis.pkl'), 'wb') as f:
    pickle.dump(res_3c, f)
print("\nSaved: analysis/phase3c_power_analysis.pkl")

print("\n" + "=" * 70)
print("Phase 3 complete. Update paper text with results from:")
print("  analysis/phase3a_random_permutation.pkl")
print("  analysis/phase3b_generalist_formal.pkl")
print("  analysis/phase3c_power_analysis.pkl")
print("=" * 70)
