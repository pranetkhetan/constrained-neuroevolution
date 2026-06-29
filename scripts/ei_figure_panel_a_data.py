"""
ei_figure_panel_a_data.py
=========================
Generates figures/ei_cache_inter.pkl — the interneuron E/I timeseries
data required for Panel A of Figure 4 (fig:architecture).

This only needs to be re-run when:
  • ei_cache_inter.pkl is missing, or
  • the agent data in data/agents/ has changed.

The cache is skipped automatically if the file already exists
(edit the path below or delete the file to force a recompute).

Usage:
    python scripts/ei_figure_panel_a_data.py
    python scripts/ei_figure_panel_a_data.py --force   # ignore existing cache
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from scripts.ei_analysis import compute_ei_timeseries

BASE_DIR   = os.path.join(PROJECT_DIR, 'data', 'agents')
CACHE_PATH = os.path.join(PROJECT_DIR, 'figures', 'ei_cache_inter.pkl')


def main():
    parser = argparse.ArgumentParser(
        description='Generate ei_cache_inter.pkl for Fig 4 Panel A.')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if cache already exists.')
    parser.add_argument('--base-dir', default=BASE_DIR,
                        help=f'Agent results directory (default: {BASE_DIR})')
    parser.add_argument('--cache-path', default=CACHE_PATH,
                        help=f'Output cache path (default: {CACHE_PATH})')
    parser.add_argument('--elite-frac', type=float, default=0.1,
                        help='Fraction of top agents to average over (default: 0.1)')
    args = parser.parse_args()

    if args.force and os.path.exists(args.cache_path):
        os.remove(args.cache_path)
        print(f'Removed existing cache: {args.cache_path}')

    print(f'Base dir : {args.base_dir}')
    print(f'Cache    : {args.cache_path}')
    print(f'Targets  : interneurons (indices 6-11)')
    print()

    compute_ei_timeseries(
        args.base_dir,
        target_indices=np.arange(6, 12),   # interneurons 6-11
        elite_frac=args.elite_frac,
        cache_path=args.cache_path,
    )

    print(f'\nDone. Cache written to: {args.cache_path}')
    print('Next step: python scripts/generate_all_figures.py --figure fig4')


if __name__ == '__main__':
    main()
