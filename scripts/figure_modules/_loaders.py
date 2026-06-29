"""
Lazy-loading, cached data store for figure generation.

Each pickle is loaded at most once per pipeline run. Figure modules call
store.mouse_data(), store.circuit_features(), etc. -- if the data was already
loaded by a previous figure, it returns instantly from cache.
"""

import os
import pickle
from typing import Any

import numpy as np


class _CpuUnpickler(pickle.Unpickler):
    """Unpickler that maps CuPy arrays to NumPy (for loading GPU-trained data on CPU)."""
    def find_class(self, module, name):
        if module.startswith('cupy'):
            module = module.replace('cupy._core.core', 'numpy').replace('cupy', 'numpy')
        return super().find_class(module, name)


def _load_pickle_cpu(path: str) -> Any:
    """Load a pickle file, converting CuPy arrays to NumPy if needed."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        with open(path, 'rb') as f:
            return _CpuUnpickler(f).load()


# ---------------------------------------------------------------------------
# Best-agent access (Tier-1 keystone)
# ---------------------------------------------------------------------------
# The compact data/best_agents.pkl (~130 KB) holds the 54 specialist + 6
# generalist gen-150 best agents distilled from the full 8.7 GB data/agents/.
# Figure/analysis code calls load_best_agent(project_dir, mouse, rep); it reads
# best_agents.pkl when present (repo default) and falls back to scanning
# data/agents/ for users working from the full Tier-2 archive.
_BEST_AGENTS_CACHE = {}


def load_best_agent(project_dir: str, mouse: str, rep: int, gen: int = 150):
    """Return the gen-`gen` best Agent for (mouse, rep).

    Prefers data/best_agents.pkl; falls back to data/agents/.../summary.pkl.
    Pass mouse='__generalist__' to load a generalist replicate.
    """
    bp = os.path.join(project_dir, 'data', 'best_agents.pkl')
    if bp not in _BEST_AGENTS_CACHE and os.path.exists(bp):
        _BEST_AGENTS_CACHE[bp] = _load_pickle_cpu(bp)
    blob = _BEST_AGENTS_CACHE.get(bp)
    if blob is not None:
        if mouse == '__generalist__':
            rec = blob.get('generalists', {}).get(rep)
        else:
            rec = blob.get('specialists', {}).get((mouse, rep))
        if rec is not None:
            return rec['agent']
        # fall through to disk scan if the key is unexpectedly absent

    # Fallback: scan the full agent archive (Tier-2). Honour NEUROEVO_DATA_DIR so a
    # user can keep the downloaded Zenodo archive outside the repo.
    try:
        from core.paths import agents_dir, generalist_dir
        base = generalist_dir() if mouse == '__generalist__' else agents_dir()
    except Exception:
        base = os.path.join(project_dir, 'data',
                            'generalist' if mouse == '__generalist__' else 'agents')
    if mouse == '__generalist__':
        run = os.path.join(str(base), f'results_r{rep}')
    else:
        run = os.path.join(str(base), f'results_{mouse}_r{rep}')
    path = os.path.join(run, f'gen_{gen}', 'summary.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No best_agents.pkl entry for ({mouse}, {rep}) and no {path}. "
            f"Ship data/best_agents.pkl or the full data/agents/ archive."
        )
    pop = _load_pickle_cpu(path)
    return min(pop, key=lambda r: r['fitness'])['agent']


class DataStore:
    """
    Lazy-loading, cached data store. Each pickle loaded at most once.

    Usage:
        store = DataStore('analysis/', 'figures/')
        data = store.mouse_data()       # loads on first call
        data = store.mouse_data()       # returns cached on second call
    """

    def __init__(self, analysis_dir: str, figures_dir: str, project_dir: str):
        self._analysis = analysis_dir
        self._figures = figures_dir
        self._project = project_dir
        self._cache: dict[str, Any] = {}

    def _load(self, key: str, path: str) -> Any:
        if key not in self._cache:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required data file missing: {path}")
            self._cache[key] = _load_pickle_cpu(path)
        return self._cache[key]

    # ── analysis/ pickles ───────────────────────────────────────────────
    def mouse_data(self):
        return self._load('mouse_data',
                          os.path.join(self._analysis, 'mouse_data.pkl'))

    def circuit_features(self):
        return self._load('circuit_features',
                          os.path.join(self._analysis, 'circuit_features.pkl'))

    def random_baseline(self):
        return self._load('random_baseline',
                          os.path.join(self._analysis, 'random_baseline.pkl'))

    def stats_results(self):
        return self._load('stats_results',
                          os.path.join(self._analysis, 'stats_results.pkl'))

    def generalization_meta(self):
        return self._load('generalization_meta',
                          os.path.join(self._analysis, 'generalization_meta.pkl'))

    def deep_analysis_results(self):
        return self._load('deep_analysis_results',
                          os.path.join(self._analysis, 'deep_analysis_results.pkl'))

    def dynamics_results(self):
        return self._load('dynamics_results',
                          os.path.join(self._analysis, 'dynamics_results.pkl'))

    def dynamics_results_full(self):
        return self._load('dynamics_results_full',
                          os.path.join(self._analysis, 'dynamics_results_full.pkl'))

    def specificity_results(self):
        return self._load('specificity_results',
                          os.path.join(self._analysis, 'specificity_results.pkl'))

    def cross_mouse_per_metric(self):
        return self._load('cross_mouse_per_metric',
                          os.path.join(self._analysis, 'cross_mouse_per_metric.pkl'))

    def generalist_results(self):
        return self._load('generalist_results',
                          os.path.join(self._analysis, 'generalist_results.pkl'))

    def a3_difficulty(self):
        return self._load('a3_difficulty',
                          os.path.join(self._analysis, 'A3_difficulty_correlation.pkl'))

    def a5_random_null(self):
        return self._load('a5_random_null',
                          os.path.join(self._analysis, 'A5_random_null.pkl'))

    def phase3a(self):
        return self._load('phase3a',
                          os.path.join(self._analysis, 'phase3a_random_permutation.pkl'))

    def phase3b(self):
        return self._load('phase3b',
                          os.path.join(self._analysis, 'phase3b_generalist_formal.pkl'))

    def phase3c(self):
        return self._load('phase3c',
                          os.path.join(self._analysis, 'phase3c_power_analysis.pkl'))

    def weight_data(self):
        return self._load('weight_data',
                          os.path.join(self._analysis, 'weight_data.pkl'))

    def source_sensitivity(self):
        return self._load('source_sensitivity',
                          os.path.join(self._analysis, 'source_sensitivity_results.pkl'))

    def generalization_matrix(self):
        key = 'generalization_matrix'
        if key not in self._cache:
            path = os.path.join(self._analysis, 'generalization_matrix.npy')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required data file missing: {path}")
            self._cache[key] = np.load(path)
        return self._cache[key]

    # ── figures/ caches ─────────────────────────────────────────────────
    def emergent_data_permouse(self):
        return self._load('emergent_data_permouse',
                          os.path.join(self._analysis, 'emergent_data_permouse.pkl'))

    def _ei_cache_path(self, name):
        """E/I caches are shipped under analysis/; ei_analysis.py also writes
        them under figures/. Prefer whichever exists."""
        fig = os.path.join(self._figures, name)
        ana = os.path.join(self._analysis, name)
        return fig if os.path.exists(fig) else ana

    def ei_cache_speed(self):
        return self._load('ei_cache_speed', self._ei_cache_path('ei_cache_speed.pkl'))

    def ei_cache_turn(self):
        return self._load('ei_cache_turn', self._ei_cache_path('ei_cache_turn.pkl'))

    def ei_cache_all(self):
        return self._load('ei_cache_all', self._ei_cache_path('ei_cache_all.pkl'))

    def ei_cache_inter(self):
        path = self._ei_cache_path('ei_cache_inter.pkl')
        if not os.path.exists(path):
            return {}   # placeholder until generated
        return self._load('ei_cache_inter', path)

    def trajectories(self):
        return self._load('trajectories',
                          os.path.join(self._figures, 'trajectories.pkl'))

    def act_emb_b(self):
        """Activity embedding B_results.pkl (B2–B5: motor ctrl, dimensionality, RSA)."""
        return self._load('act_emb_b',
                          os.path.join(self._analysis, 'activity_embeddings',
                                       'B_results.pkl'))

    def act_emb_d(self):
        """Activity embedding D_results.pkl (D1–D4: Mantel tests, attractor landscape)."""
        return self._load('act_emb_d',
                          os.path.join(self._analysis, 'activity_embeddings',
                                       'D_results.pkl'))

    def spec_evol(self):
        """Specialisation index over generations (nb16). Keys: spec_mean (16,9), spec_std, SAMPLE_GENS, MICE."""
        return self._load('spec_evol',
                          os.path.join(self._analysis, 'spec_evolution.pkl'))

    def spec_evol_gen(self):
        """Generalist per-mouse bias over generations (nb17). Keys: bias_mean, mean_abs_bias, SAMPLE_GENS."""
        return self._load('spec_evol_gen',
                          os.path.join(self._analysis, 'spec_evol_generalist.pkl'))

    def sens_commitment_evo(self):
        """Sensitivity commitment evolution (nb18). Keys: spec_var_mean_traj, gen_var_mean_traj, within_sim_mean, between_sim_mean."""
        return self._load('sens_commitment_evo',
                          os.path.join(self._analysis, 'sens_commitment_evolution.pkl'))

    def holdout_results(self):
        """Cross-bout holdout evaluation (holdout_eval.py). Keys: holdout_matrix (9x9), training_ratio, holdout_ratio, per_mouse_holdout, bout_counts, split_fraction."""
        return self._load('holdout_results',
                          os.path.join(self._analysis, 'R9_holdout_results.pkl'))
