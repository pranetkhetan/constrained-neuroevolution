"""
generate_all_figures.py
=======================
Unified figure generation pipeline for the constrained neuroevolution paper.

Generates all 8 main figures + 15 supplementary figures (41 image files total)
from pre-extracted pickle data. No heavy computation -- just plotting.

Data is loaded lazily and cached: each pickle file is read at most once per run.
Re-running after a style change costs only ~2-3s for data load + <1s per figure.

Usage
-----
    python scripts/generate_all_figures.py              # all figures
    python scripts/generate_all_figures.py --figure fig2 # single figure
    python scripts/generate_all_figures.py --main-only   # main figures only
    python scripts/generate_all_figures.py --supp-only   # supplementary only
    python scripts/generate_all_figures.py --list         # print manifest
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
import traceback

# Ensure project root is importable
PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from scripts.figure_modules._style import apply_pub_style
from scripts.figure_modules._loaders import DataStore

# Lazy imports -- modules are imported only when their figure is requested
from scripts.figure_modules import (
    main_fig2_convergence,
    main_fig3_emergent,
    main_fig4_network,
    main_fig5_circuit,
    main_fig7_paradox,
    main_fig8_specificity,
    main_fig9_topology_encoding,
    main_fig_evr,
    supp_s1_permouse,
    supp_s2_clustering,
    supp_s4_trajectories,
    supp_s16_networks_per_mouse,
    supp_s5_fixedpoints,
    supp_s6_permutation,
    supp_s7_topology,
    supp_s9_nonzero,
    supp_s10_permetric,
    supp_s11_randomnull,
    supp_s12_difficulty,
    supp_s13_randomcontrol,
    supp_s14_powercurve,
    supp_s15_generalist,
    supp_s17_dynamics_null,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure manifest: maps CLI name -> (module, output files, description, type)
# ═══════════════════════════════════════════════════════════════════════════════
MANIFEST = {
    'fig1': {
        'module': None,  # special case: subprocess
        'outputs': ['sim_setup_figure.png'],
        'desc': 'Setup diagram',
        'type': 'main',
        'label': 'fig:setup',
    },
    'fig2': {
        'module': main_fig2_convergence,
        'outputs': ['fig1_permouse.pdf'],
        'desc': 'Convergence curves',
        'type': 'main',
        'label': 'fig:convergence',
    },
    'fig3': {
        'module': main_fig3_emergent,
        'outputs': ['fig2_permouse.pdf'],
        'desc': 'Emergent behavior',
        'type': 'main',
        'label': 'fig:emergent',
    },
    'fig4': {
        'module': main_fig4_network,
        'outputs': ['fig3_network_ei.pdf'],
        'desc': 'Network/EI/similarity',
        'type': 'main',
        'label': 'fig:architecture',
    },
    'fig_evr': {
        'module': main_fig_evr,
        'outputs': ['fig_evr_forest.pdf'],
        'desc': 'Evolved vs random forest plot (Cohen d, 18 features)',
        'type': 'main',
        'label': 'fig:evr',
    },
    'fig5': {
        'module': main_fig5_circuit,
        'outputs': ['fig5_circuit_specialization.pdf'],
        'desc': 'Cosine heatmap + cosine dist + fitness + specialization (A/B/C/D)',
        'type': 'main',
        'label': 'fig:circuit',
    },
    'fig7': {
        'module': main_fig7_paradox,
        'outputs': ['fig7_paradox.pdf'],
        'desc': 'Power analysis A+B combined',
        'type': 'main',
        'label': 'fig:paradox',
    },
    'fig8': {
        'module': main_fig8_specificity,
        'outputs': ['specificity_optionA.png'],
        'desc': 'Permutation specificity',
        'type': 'main',
        'label': 'fig:specificity',
    },
    'fig9': {
        'module': main_fig9_topology_encoding,
        'outputs': ['fig9_topology_encoding.pdf'],
        'desc': 'Topology encoding: topology vs magnitude + per-source ablation',
        'type': 'main',
        'label': 'fig:topology_encoding',
    },
    's1': {
        'module': supp_s1_permouse,
        'outputs': [f'fig_s1_{m}.png' for m in
                    ['B5', 'B6', 'B7', 'D3', 'D4', 'D5', 'D7', 'D8', 'D9']],
        'desc': 'Per-mouse convergence',
        'type': 'supp',
        'label': 'fig:supp_permouse',
    },
    's2': {
        'module': supp_s2_clustering,
        'outputs': ['strain_clustering.pdf'],
        'desc': 'Strain clustering',
        'type': 'supp',
        'label': 'fig:supp_strain',
    },
    's4': {
        'module': supp_s4_trajectories,
        'outputs': ['fig_s4_trajectories.pdf'],
        'desc': 'Activation trajectories',
        'type': 'supp',
        'label': 'fig:supp_trajectories',
    },
    's5': {
        'module': supp_s5_fixedpoints,
        'outputs': ['fig_s5_fixedpoints.pdf'],
        'desc': 'Fixed points',
        'type': 'supp',
        'label': 'fig:supp_fixedpoints',
    },
    's6': {
        'module': supp_s6_permutation,
        'outputs': ['fig_s6_permutation.pdf'],
        'desc': 'Weight permutation ablation',
        'type': 'supp',
        'label': 'fig:supp_permutation',
    },
    's7': {
        'module': supp_s7_topology,
        'outputs': ['specificity_optionB.png'],
        'desc': 'Topology vs magnitude',
        'type': 'supp',
        'label': 'fig:supp_magnitude',
    },
    's9': {
        'module': supp_s9_nonzero,
        'outputs': ['fig_c4_nonzero_positions.png'],
        'desc': 'Non-zero positions',
        'type': 'supp',
        'label': 'fig:supp_nonzero',
    },
    's10': {
        'module': supp_s10_permetric,
        'outputs': ['fig_per_metric_crosseval.pdf',
                     'fig_specialization_by_component.pdf'],
        'desc': 'Per-metric crosseval',
        'type': 'supp',
        'label': 'fig:supp_permetric',
    },
    's11': {
        'module': supp_s11_randomnull,
        'outputs': ['supp_A5_random_null.pdf'],
        'desc': 'Random null distribution',
        'type': 'supp',
        'label': 'fig:supp_random_null',
    },
    's12': {
        'module': supp_s12_difficulty,
        'outputs': ['supp_A3_difficulty_correlation.pdf'],
        'desc': 'Difficulty correlation',
        'type': 'supp',
        'label': 'fig:supp_difficulty',
    },
    's13': {
        'module': supp_s13_randomcontrol,
        'outputs': ['fig_supp_random_control.png'],
        'desc': 'Random-agent control',
        'type': 'supp',
        'label': 'fig:supp_random_control',
    },
    's14': {
        'module': supp_s14_powercurve,
        'outputs': ['fig_supp_power_curve.png'],
        'desc': 'Power curve',
        'type': 'supp',
        'label': 'fig:supp_power_curve',
    },
    's15': {
        'module': supp_s15_generalist,
        'outputs': ['fig_supp_generalist_formal.png'],
        'desc': 'Generalist formal',
        'type': 'supp',
        'label': 'fig:supp_generalist_formal',
    },
    's16': {
        'module': supp_s16_networks_per_mouse,
        'outputs': ['fig_supp_networks_per_mouse.pdf'],
        'desc': 'Evolved network diagrams per mouse (rep 1 best agent)',
        'type': 'supp',
        'label': 'fig:supp_networks_per_mouse',
    },
    's17': {
        'module': supp_s17_dynamics_null,
        'outputs': ['fig_s17_dynamics_null.pdf'],
        'desc': 'Dynamical null results: per-mouse lambda1 + RSM within vs between',
        'type': 'supp',
        'label': 'fig:supp_dynamics_null',
    },
}


def generate_fig1_setup(figures_dir: str) -> list[str]:
    """Generate Fig 1 via the standalone sim_setup_figure.py script."""
    script = os.path.join(PROJECT_DIR, 'scripts', 'sim_setup_figure.py')
    out = os.path.join(figures_dir, 'sim_setup_figure.png')
    result = subprocess.run(
        [sys.executable, script, '--output', out],
        capture_output=True, text=True, cwd=PROJECT_DIR,
    )
    if result.returncode != 0:
        print(f"    STDERR: {result.stderr[:500]}")
        return []
    return [out]


def print_manifest():
    """Print the full figure manifest."""
    print(f"\n{'Key':<6} {'Type':<5} {'LaTeX label':<25} {'Outputs':<50} {'Description'}")
    print("-" * 120)
    for key, info in MANIFEST.items():
        outputs = ', '.join(info['outputs'])
        if len(outputs) > 48:
            outputs = outputs[:45] + '...'
        print(f"{key:<6} {info['type']:<5} {info['label']:<25} {outputs:<50} {info['desc']}")
    total_files = sum(len(info['outputs']) for info in MANIFEST.values())
    print(f"\nTotal: {len(MANIFEST)} figure groups, {total_files} image files")


def main():
    parser = argparse.ArgumentParser(
        description='Generate all paper figures from pre-extracted data.')
    parser.add_argument('--figure', '-f', type=str, default=None,
                        help='Generate a single figure (e.g., fig2, s3)')
    parser.add_argument('--main-only', action='store_true',
                        help='Generate main figures only')
    parser.add_argument('--supp-only', action='store_true',
                        help='Generate supplementary figures only')
    parser.add_argument('--list', action='store_true',
                        help='Print figure manifest and exit')
    parser.add_argument('--figures-dir', type=str, default=None,
                        help='Output directory (default: PROJECT/figures)')
    parser.add_argument('--analysis-dir', type=str, default=None,
                        help='Analysis directory (default: PROJECT/analysis)')
    args = parser.parse_args()

    if args.list:
        print_manifest()
        return

    # Resolve directories
    figures_dir = args.figures_dir or os.path.join(PROJECT_DIR, 'figures')
    analysis_dir = args.analysis_dir or os.path.join(PROJECT_DIR, 'analysis')
    os.makedirs(figures_dir, exist_ok=True)

    # Apply publication style
    apply_pub_style()

    # Create shared data store (lazy-loading, cached)
    store = DataStore(analysis_dir, figures_dir, PROJECT_DIR)

    # Determine which figures to generate
    if args.figure:
        keys = [args.figure]
        if args.figure not in MANIFEST:
            print(f"Unknown figure: {args.figure}")
            print(f"Available: {', '.join(MANIFEST.keys())}")
            sys.exit(1)
    elif args.main_only:
        keys = [k for k, v in MANIFEST.items() if v['type'] == 'main']
    elif args.supp_only:
        keys = [k for k, v in MANIFEST.items() if v['type'] == 'supp']
    else:
        keys = list(MANIFEST.keys())

    # Generate
    total_start = time.time()
    results = {}
    for key in keys:
        info = MANIFEST[key]
        print(f"[{key}] {info['desc']} ...", end=' ', flush=True)
        t0 = time.time()
        try:
            if key == 'fig1':
                outputs = generate_fig1_setup(figures_dir)
            else:
                outputs = info['module'].generate(store, figures_dir)
            dt = time.time() - t0
            results[key] = ('OK', outputs, dt)
            print(f"OK ({dt:.1f}s) -> {', '.join(os.path.basename(p) for p in outputs)}")
        except Exception as e:
            dt = time.time() - t0
            results[key] = ('FAIL', [], dt)
            print(f"FAIL ({dt:.1f}s): {e}")

    # Summary
    total_dt = time.time() - total_start
    ok = sum(1 for s, _, _ in results.values() if s == 'OK')
    fail = sum(1 for s, _, _ in results.values() if s == 'FAIL')
    total_files = sum(len(outs) for _, outs, _ in results.values())
    print(f"\nDone: {ok} OK, {fail} FAIL, {total_files} files generated in {total_dt:.1f}s")

    if fail > 0:
        print("\nFailed figures:")
        for key, (status, _, _) in results.items():
            if status == 'FAIL':
                print(f"  {key}: {MANIFEST[key]['desc']}")
        sys.exit(1)


if __name__ == '__main__':
    main()
