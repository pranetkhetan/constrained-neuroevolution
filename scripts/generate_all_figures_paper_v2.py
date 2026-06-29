"""
generate_all_figures_paper_v2.py
=================================
Figure generation pipeline for paper_v2 (degeneracy-as-central-thesis reframe).

Main figures (5):
  fig1  fig2  fig3  figA  figC
  (figB retired 2026-06-08: NMI panel absorbed into figA Panel E)

Supplementary figures (reused from v1 modules, listed here for independence
from generate_all_figures.py):
  s1-s17 as in v1, with the addition of supp_sensitivity_rsa (A4).

Usage
-----
    python scripts/generate_all_figures_paper_v2.py              # all figures
    python scripts/generate_all_figures_paper_v2.py --main-only
    python scripts/generate_all_figures_paper_v2.py --supp-only
    python scripts/generate_all_figures_paper_v2.py --figure fig2
    python scripts/generate_all_figures_paper_v2.py --list
"""

import argparse, os, sys, subprocess, time
from pathlib import Path

PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from scripts.figure_modules._style import apply_pub_style
from scripts.figure_modules._loaders import DataStore

# -- Main figure modules (paper_v2 new) ----------------------------------------
from scripts.figure_modules import (
    main_fig1_system_v2,
    main_fig2_paradox_v2,
    main_fig3_causal_v2,
    main_figA_topo_flat_v2,
    main_figC_sensitivity_v2,
    main_fig_methods_fitness,
    main_fig_embedding_rsa,      # §2.7: representational geometry
    main_circuit_comparison,     # §2.8: circuit wiring diagram comparison
)

# -- Supplementary figure modules (reused from v1) -----------------------------
from scripts.figure_modules import (
    supp_s_mi_clustering,
    supp_s1_permouse,
    supp_s2_clustering,
    supp_s4_trajectories,
    supp_s5_fixedpoints,
    supp_s6_permutation,
    supp_s9_nonzero,
    supp_s10_permetric,
    supp_s11_randomnull,
    supp_s12_difficulty,
    supp_s13_randomcontrol,
    supp_s14_powercurve,
    supp_s15_generalist,
    supp_s16_networks_per_mouse,
    supp_s17_dynamics_null,
    supp_act_emb_ctrl,           # §2.7 supp: B2 motor ctrl + B3 dimensionality
    supp_d4_attractor,           # §2.8 supp: attractor landscape (D4)
    supp_spec_evol_traj,         # §2.2 supp: per-mouse spec index trajectories
    supp_sens_commitment_evo,    # §2.6 supp: sensitivity commitment temporal co-development
    supp_ablation_heatmap,       # §2.5 supp: 54x14 per-source ablation (null result)
    supp_holdout,                # §2.1 supp: holdout specialisation per mouse (S12)
    # paper_v2 figures previously generated outside this driver (v1 driver / ad-hoc) —
    # consolidated here so rebuild_all.sh regenerates 100% of paper_v2 figures.
    main_fig2_convergence,       # §2.1 supp: per-mouse convergence (fig1_permouse)
    main_fig3_emergent,          # §2.1 supp: emergent behaviours (fig2_permouse)
    main_fig4_network,           # §2.1 supp: E/I network dynamics (fig3_network_ei)
    supp_s8_cosine,              # §2.2 supp: evolved-vs-random cosine (S5)
    supp_weighting_sensitivity,  # §2.2 supp: fitness-weighting robustness (S11)
    supp_b3_dim,                 # §2.7 supp: manifold dimensionality (S17)
    supp_emb_robustness,         # §2.7 supp: 6-metric robustness (S18)
)

MICE = ['B5', 'B6', 'B7', 'D3', 'D4', 'D5', 'D7', 'D8', 'D9']

MANIFEST = {
    # ── main figures ──────────────────────────────────────────────────────────
    'fig1': {
        'module': main_fig1_system_v2,
        'outputs': ['fig1_system_v2.pdf'],
        'desc': 'System + convergence (A:setup B:conv C:emergent D:E/I heatmap)',
        'label': 'fig:system', 'claims': 'C1-C3', 'type': 'main',
    },
    'fig2': {
        'module': main_fig2_paradox_v2,
        'outputs': ['fig2_paradox_v2.pdf'],
        'desc': 'Degeneracy paradox (A:ANOVA bars B:9x9 matrix C:spec index)',
        'label': 'fig:paradox', 'claims': 'C4-C5', 'type': 'main',
    },
    'figA': {
        'module': main_figA_topo_flat_v2,
        'outputs': ['figA_topo_flat_v2.pdf'],
        'desc': (
            'Topo-behaviour degeneracy §2.3 + MI §2.4 '
            '(A:schematic placeholder B:54x54 Jaccard heatmap '
            'C:joint KDE+marginals with within/between KDE dual E:NMI bars)'
        ),
        'label': 'fig:topo_flat', 'claims': 'C6-C8', 'type': 'main',
    },
    # figB retired: NMI panel absorbed into figA Panel E (2026-06-08 Session 14)
    'fig3': {
        'module': main_fig3_causal_v2,
        'outputs': ['fig3_causal_v2.pdf'],
        'desc': 'Causal necessity (A:perm MSE B:ablation heatmap C:dose-response)',
        'label': 'fig:causal', 'claims': 'C9-C11', 'type': 'main',
    },
    'figC': {
        'module': main_figC_sensitivity_v2,
        'outputs': ['figC_sensitivity_v2.pdf'],
        'desc': 'Sensitivity commitment (A:topo sim B:fitness cost C:var log-scale)',
        'label': 'fig:sensitivity_commitment', 'claims': 'C12-C14', 'type': 'main',
    },
    'methods_fitness': {
        'module': main_fig_methods_fitness,
        'outputs': ['fig_methods_fitness_metrics.pdf'],
        'desc': 'Methods: behavioral fitness metrics panel',
        'label': 'fig:methods_fitness_metrics', 'claims': '-', 'type': 'main',
    },
    'fig_embedding_rsa': {
        'module': main_fig_embedding_rsa,
        'outputs': ['fig_embedding_rsa_v2.pdf'],
        'desc': '§2.7 rep-geometry (A:B4 KDE B:D2 scatter C:3x3 loading clouds)',
        'label': 'fig:embedding_rsa', 'claims': 'C15-C19', 'type': 'main',
    },
    'fig_circuit_comparison': {
        'module': main_circuit_comparison,
        'outputs': ['fig_circuit_comparison.pdf'],
        'desc': '§2.8 wiring diagrams: similar (D9 r2/r5) vs dissimilar (B5 r1/r5) pairs',
        'label': 'fig:circuit_comparison', 'claims': 'C20-C22', 'type': 'main',
    },
    # ── supplementary figures ─────────────────────────────────────────────────
    'supp_act_emb_ctrl': {
        'module': supp_act_emb_ctrl,
        'outputs': ['fig_supp_act_emb_ctrl.pdf'],
        'desc': '§2.7 supp: B2 motor separation ctrl + B3 manifold dimensionality',
        'label': 'fig:supp_act_emb_ctrl', 'claims': '-', 'type': 'supp',
    },
    'supp_d4_attractor': {
        'module': supp_d4_attractor,
        'outputs': ['fig_supp_d4_attractor.pdf'],
        'desc': '§2.8 supp: attractor landscape — 54×54 heatmap + KDE + PCA scatter',
        'label': 'fig:supp_d4', 'claims': '-', 'type': 'supp',
    },
    'supp_spec_evol_per_mouse': {
        'module': supp_spec_evol_traj,
        'outputs': ['fig_supp_spec_evol_per_mouse.pdf'],
        'desc': '§2.2 supp: per-mouse spec index trajectories (3×3 grid)',
        'label': 'fig:supp_spec_evol_per_mouse', 'claims': '-', 'type': 'supp',
    },
    'supp_sens_commitment_evo': {
        'module': supp_sens_commitment_evo,
        'outputs': ['fig_supp_sens_commitment_evo.pdf'],
        'desc': '§2.6 supp: sensitivity commitment temporal co-development (3-panel)',
        'label': 'fig:supp_sens_commitment_evo', 'claims': '-', 'type': 'supp',
    },
    'supp_dynamics_null': {
        'module': supp_s17_dynamics_null,
        'outputs': ['fig_s17_dynamics_null.pdf'],
        'desc': '§2.8 supp: per-mouse λ₁ strip + trajectory RSM within vs between',
        'label': 'fig:supp_dynamics_null', 'claims': '-', 'type': 'supp',
    },
    'supp_mi_clustering': {
        'module': supp_s_mi_clustering,
        'outputs': ['supp_mi_clustering.pdf'],
        'desc': 'Structural clustering PCA — k-means vs mouse identity (Supp S13)',
        'label': 'fig:supp_mi_clustering', 'claims': 'C8', 'type': 'supp',
    },
    's1': {
        'module': supp_s1_permouse,
        'outputs': [f'fig_s1_{m}.png' for m in MICE],
        'desc': 'Per-mouse convergence (Supp)',
        'label': 'fig:supp_permouse', 'claims': '-', 'type': 'supp',
    },
    's2': {
        'module': supp_s2_clustering,
        'outputs': ['strain_clustering.pdf'],
        'desc': 'Strain clustering (Supp)',
        'label': 'fig:supp_strain', 'claims': '-', 'type': 'supp',
    },
    's4': {
        'module': supp_s4_trajectories,
        'outputs': ['fig_s4_trajectories.pdf'],
        'desc': 'Activation trajectories (Supp)',
        'label': 'fig:supp_trajectories', 'claims': '-', 'type': 'supp',
    },
    's5': {
        'module': supp_s5_fixedpoints,
        'outputs': ['fig_s5_fixedpoints.pdf'],
        'desc': 'Fixed points (Supp)',
        'label': 'fig:supp_fixedpoints', 'claims': '-', 'type': 'supp',
    },
    's6': {
        'module': supp_s6_permutation,
        'outputs': ['fig_s6_permutation.pdf'],
        'desc': 'Weight permutation ablation (Supp)',
        'label': 'fig:supp_permutation', 'claims': '-', 'type': 'supp',
    },
    's9': {
        'module': supp_s9_nonzero,
        'outputs': ['fig_c4_nonzero_positions.png'],
        'desc': 'Non-zero positions (Supp)',
        'label': 'fig:supp_nonzero', 'claims': '-', 'type': 'supp',
    },
    's10': {
        'module': supp_s10_permetric,
        'outputs': ['fig_per_metric_crosseval.pdf'],
        'desc': 'Per-metric 4-panel heatmaps (Supp S6)',
        'label': 'fig:supp_per_metric_heatmaps', 'claims': '-', 'type': 'supp',
    },
    's11': {
        'module': supp_s11_randomnull,
        'outputs': ['supp_A5_random_null.pdf'],
        'desc': 'Random null distribution (Supp)',
        'label': 'fig:supp_random_null', 'claims': '-', 'type': 'supp',
    },
    's12': {
        'module': supp_s12_difficulty,
        'outputs': ['supp_A3_difficulty_correlation.pdf'],
        'desc': 'Difficulty correlation (Supp)',
        'label': 'fig:supp_difficulty', 'claims': '-', 'type': 'supp',
    },
    's13': {
        'module': supp_s13_randomcontrol,
        'outputs': ['fig_supp_random_control.png'],
        'desc': 'Random-agent control (Supp)',
        'label': 'fig:supp_random_control', 'claims': '-', 'type': 'supp',
    },
    's14': {
        'module': supp_s14_powercurve,
        'outputs': ['fig_supp_power_curve.png'],
        'desc': 'Power curve (Supp)',
        'label': 'fig:supp_power_curve', 'claims': '-', 'type': 'supp',
    },
    's15': {
        'module': supp_s15_generalist,
        'outputs': ['fig_supp_generalist_formal.png'],
        'desc': 'Generalist formal (Supp)',
        'label': 'fig:supp_generalist_formal', 'claims': '-', 'type': 'supp',
    },
    's16': {
        'module': supp_s16_networks_per_mouse,
        'outputs': ['fig_supp_networks_per_mouse.pdf'],
        'desc': 'Evolved networks per mouse (Supp)',
        'label': 'fig:supp_networks_per_mouse', 'claims': '-', 'type': 'supp',
    },
    's17': {
        'module': supp_s17_dynamics_null,
        'outputs': ['fig_s17_dynamics_null.pdf'],
        'desc': 'Dynamical null results (Supp)',
        'label': 'fig:supp_dynamics_null', 'claims': '-', 'type': 'supp',
    },
    # ── power analysis (was v1 fig7 panels A+B) ───────────────────────────────
    'supp_power': {
        'module': None,   # generated inline below
        'outputs': ['fig_supp_power_v2.pdf'],
        'desc': 'Power analysis panels A+B -- from fig7_paradox (Supp)',
        'label': 'fig:supp_power', 'claims': '-', 'type': 'supp',
    },
    # ── §2.5 ablation null result (moved from fig3 Panel B) ─────────────────
    'supp_ablation_heatmap': {
        'module': supp_ablation_heatmap,
        'outputs': ['fig_supp_ablation_heatmap.pdf'],
        'desc': '54x14 per-source ablation heatmap -- null result (Supp)',
        'label': 'fig:supp_ablation_heatmap', 'claims': '-', 'type': 'supp',
    },
    # ── §2.1 / S12 holdout specialisation per mouse ──────────────────────────
    'supp_holdout': {
        'module': supp_holdout,
        'outputs': ['fig_supp_holdout.pdf'],
        'desc': 'Holdout specialisation indices per mouse (Supp S12)',
        'label': 'fig:supp_holdout', 'claims': '-', 'type': 'supp',
    },
    # ── consolidated paper_v2 figures (previously v1-driver / ad-hoc) ─────────
    'permouse_conv': {
        'module': main_fig2_convergence,
        'outputs': ['fig1_permouse.pdf'],
        'desc': 'Per-metric convergence per mouse (Supp S3)',
        'label': 'fig:supp_convergence_permouse', 'claims': '-', 'type': 'supp',
    },
    'permouse_emergent': {
        'module': main_fig3_emergent,
        'outputs': ['fig2_permouse.pdf'],
        'desc': 'Emergent behaviours per mouse (Supp S4)',
        'label': 'fig:supp_emergent', 'claims': '-', 'type': 'supp',
    },
    'network_ei': {
        'module': main_fig4_network,
        'outputs': ['fig3_network_ei.pdf'],
        'desc': 'E/I balance dynamics (Supp S5)',
        'label': 'fig:supp_ei_dynamics', 'claims': '-', 'type': 'supp',
    },
    'cosine_evr': {
        'module': supp_s8_cosine,
        'outputs': ['fig_c1_cosine_evolved_vs_random.png'],
        'desc': 'Evolved vs random weight cosine similarity (Supp S6)',
        'label': 'fig:supp_cosine_dist', 'claims': '-', 'type': 'supp',
    },
    'weighting': {
        'module': supp_weighting_sensitivity,
        'outputs': ['supp_s_weighting.pdf'],
        'desc': 'Specialisation index robustness to fitness weighting (Supp S11)',
        'label': 'fig:supp_weighting_schemes', 'claims': '-', 'type': 'supp',
    },
    'b3_dim': {
        'module': supp_b3_dim,
        'outputs': ['fig_supp_b3_dim.pdf'],
        'desc': 'Manifold dimensionality near-maximal (Supp S17)',
        'label': 'fig:supp_b3_dim', 'claims': '-', 'type': 'supp',
    },
    'emb_robustness': {
        'module': supp_emb_robustness,
        'outputs': ['fig_supp_emb_robustness.pdf'],
        'desc': 'Representational degeneracy robust across 6 metrics (Supp S18)',
        'label': 'fig:supp_emb_robustness', 'claims': '-', 'type': 'supp',
    },
}


def _generate_supp_power(store, figures_dir):
    """Power analysis panels A+B from v1 main_fig7_paradox, moved to Supp."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scripts.figure_modules.main_fig7_paradox import _power_panel, _reps_panel
    from scripts.stats import post_hoc_power, required_replicates
    apply_pub_style()
    stats   = store.stats_results()
    anova   = stats['anova']
    power_r = post_hoc_power(anova, n_total=54, k_groups=9)
    reps_d  = required_replicates(anova)
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, wspace=0.35,
                            left=0.07, right=0.97, top=0.93, bottom=0.10)
    _power_panel(fig.add_subplot(gs[0]), power_r)
    _reps_panel(fig.add_subplot(gs[1]), reps_d)
    out = os.path.join(figures_dir, 'fig_supp_power_v2.pdf')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'Saved -> {out}')
    return [out]


def print_manifest():
    print(f"\n{'Key':<12} {'Type':<6} {'Label':<32} {'Claims':<10} Description")
    print('-' * 110)
    for key, info in MANIFEST.items():
        print(f"{key:<12} {info['type']:<6} {info['label']:<32} {info['claims']:<10} {info['desc']}")
    n_main = sum(1 for v in MANIFEST.values() if v['type'] == 'main')
    n_supp = sum(1 for v in MANIFEST.values() if v['type'] == 'supp')
    print(f"\nTotal: {n_main} main, {n_supp} supplementary")


def main():
    parser = argparse.ArgumentParser(description='Generate paper_v2 figures.')
    parser.add_argument('--figure', '-f', type=str, default=None)
    parser.add_argument('--main-only', action='store_true')
    parser.add_argument('--supp-only', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--figures-dir', type=str, default=None)
    parser.add_argument('--analysis-dir', type=str, default=None)
    args = parser.parse_args()

    if args.list:
        print_manifest(); return

    figures_dir  = args.figures_dir  or os.path.join(PROJECT_DIR, 'figures')
    analysis_dir = args.analysis_dir or os.path.join(PROJECT_DIR, 'analysis')
    os.makedirs(figures_dir, exist_ok=True)

    apply_pub_style()
    store = DataStore(analysis_dir, figures_dir, PROJECT_DIR)

    if args.figure:
        if args.figure not in MANIFEST:
            print(f'Unknown figure: {args.figure}. Available: {", ".join(MANIFEST)}')
            sys.exit(1)
        keys = [args.figure]
    elif args.main_only:
        keys = [k for k,v in MANIFEST.items() if v['type'] == 'main']
    elif args.supp_only:
        keys = [k for k,v in MANIFEST.items() if v['type'] == 'supp']
    else:
        keys = list(MANIFEST.keys())

    total_start = time.time()
    results = {}
    for key in keys:
        info = MANIFEST[key]
        print(f'[{key}] {info["desc"]} ...', end=' ', flush=True)
        t0 = time.time()
        try:
            if key == 'supp_power':
                outputs = _generate_supp_power(store, figures_dir)
            else:
                outputs = info['module'].generate(store, figures_dir)
            dt = time.time() - t0
            results[key] = ('OK', outputs, dt)
            print(f'OK ({dt:.1f}s)')
        except Exception as e:
            import traceback
            dt = time.time() - t0
            results[key] = ('FAIL', [], dt)
            print(f'FAIL ({dt:.1f}s): {e}')
            traceback.print_exc()

    total_dt = time.time() - total_start
    ok   = sum(1 for s,_,_ in results.values() if s == 'OK')
    fail = sum(1 for s,_,_ in results.values() if s == 'FAIL')
    print(f'\nDone: {ok} OK, {fail} FAIL in {total_dt:.1f}s')
    if fail:
        for k,(s,_,_) in results.items():
            if s == 'FAIL': print(f'  FAIL: {k}')
        sys.exit(1)

if __name__ == '__main__':
    main()
