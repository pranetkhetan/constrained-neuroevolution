"""
Centralised data-path resolution.

By default the pipeline reads data from ``<repo>/data`` and ``<repo>/analysis``.
A user who downloads the large Tier-2 archive from Zenodo (the full
``data/agents/``, ``data/generalist/`` and the full ``B_results.pkl``) can save it
anywhere and point the code at it via environment variables — no source edits:

    # PowerShell
    $env:NEUROEVO_DATA_DIR     = "D:\zenodo\constrained-neuroevolution\data"
    $env:NEUROEVO_ANALYSIS_DIR = "D:\zenodo\constrained-neuroevolution\analysis"

    # bash
    export NEUROEVO_DATA_DIR=/mnt/zenodo/.../data
    export NEUROEVO_ANALYSIS_DIR=/mnt/zenodo/.../analysis

If a variable is unset, the in-repo default is used. ``agents_dir()`` additionally
falls back to the repo's ``data/agents`` only when the override has no agents, so a
partial override still works.

Resolution order for the data root:
    1. $NEUROEVO_DATA_DIR        (if set)
    2. <repo>/data               (default)
"""

import os
from pathlib import Path

# Repo root = parent of this file's package directory (core/).
REPO_ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Root directory for data (metrics, agents, generalist, raw)."""
    env = os.environ.get("NEUROEVO_DATA_DIR")
    return Path(env) if env else REPO_ROOT / "data"


def analysis_dir() -> Path:
    """Root directory for analysis intermediates (*.pkl, *.npy)."""
    env = os.environ.get("NEUROEVO_ANALYSIS_DIR")
    return Path(env) if env else REPO_ROOT / "analysis"


def agents_dir() -> Path:
    """Directory holding results_<MOUSE>_r<REP>/ run folders (full Tier-2 archive).

    Prefers $NEUROEVO_DATA_DIR/agents; falls back to the in-repo data/agents only
    if the override directory does not contain it.
    """
    primary = data_dir() / "agents"
    if primary.is_dir():
        return primary
    fallback = REPO_ROOT / "data" / "agents"
    return fallback if fallback.is_dir() else primary


def generalist_dir() -> Path:
    primary = data_dir() / "generalist"
    if primary.is_dir():
        return primary
    fallback = REPO_ROOT / "data" / "generalist"
    return fallback if fallback.is_dir() else primary
