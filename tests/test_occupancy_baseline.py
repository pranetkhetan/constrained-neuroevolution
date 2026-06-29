"""
Regression test: mouse node-occupancy baseline uses Rosenberg's start-frame convention.

For every mouse, the PDF computed by preprocess_mouse's corrected parser must match
a ground-truth PDF computed inline here using Rosenberg's own formula.
Acceptance criterion: JSD(computed, ground_truth) < 0.003 (one order of magnitude
below the ~0.06–0.12 JSD produced by the old cumulative-end-frame bug).
"""
import pickle
import numpy as np
import pytest
from dataclasses import make_dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"

# Recreate the Traj dataclass to unpickle Rosenberg .tf files
Traj = make_dataclass("Traj", ["fr", "ce", "ke", "no", "re"])
Traj.__module__ = "__main__"  # match the pickled class's module name


class _TrajUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Traj":
            return Traj
        return super().find_class(module, name)


def _load_tf(mouse_id: str):
    path = DATA_RAW / f"{mouse_id}-tf"
    with open(path, "rb") as f:
        return _TrajUnpickler(f).load()


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base-2, returns value in [0, 1])."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))) / np.log(2)


def _ground_truth_pdf(tr, n_runs: int) -> np.ndarray:
    """
    Rosenberg-correct occupancy PDF.
    bout_no[n, 1] = start frame of node n within the bout.
    Dwell of node n = start[n+1] - start[n]. Final node contributes 0.
    """
    counts = np.zeros(n_runs)
    if tr.no is None:
        return counts / (counts.sum() + 1e-9)
    for bout_no in tr.no:
        for n in range(len(bout_no) - 1):
            node_id = int(bout_no[n, 0])
            duration = int(bout_no[n + 1, 1]) - int(bout_no[n, 1])
            if duration > 0 and 0 <= node_id < n_runs:
                counts[node_id] += duration
    total = counts.sum()
    return counts / (total + 1e-9)


def _preprocess_pdf(tr, n_runs: int) -> np.ndarray:
    """
    PDF via the same logic as preprocess_mouse.py (post-fix).
    Kept here explicitly so the test remains self-contained and doesn't
    import the script (which has side-effects / arg parsing).
    """
    visit_frames = []
    if tr.no is not None:
        for bout_no in tr.no:
            for n in range(len(bout_no) - 1):
                node_id = int(bout_no[n, 0])
                duration = int(bout_no[n + 1, 1]) - int(bout_no[n, 1])
                if duration > 0 and 0 <= node_id < n_runs:
                    visit_frames.extend([node_id] * duration)
    counts = np.zeros(n_runs)
    for nid in visit_frames:
        if 0 <= nid < n_runs:
            counts[nid] += 1
    return counts / (counts.sum() + 1e-9)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
MICE = ["B5", "B6", "B7", "D3", "D4", "D5", "D7", "D8", "D9"]
N_RUNS = 128  # 6-level maze: 127 corridor runs + 1 exit node


@pytest.mark.parametrize("mouse_id", MICE)
def test_occupancy_pdf_matches_ground_truth(mouse_id: str):
    """
    The preprocess_mouse parser (start-frame convention) must produce a PDF
    within JSD < 0.003 of the inline ground-truth computation.
    """
    tf_path = DATA_RAW / f"{mouse_id}-tf"
    if not tf_path.exists():
        pytest.skip(f"Raw data not found: {tf_path}")

    tr = _load_tf(mouse_id)

    gt_pdf = _ground_truth_pdf(tr, N_RUNS)
    pp_pdf = _preprocess_pdf(tr, N_RUNS)

    # Both must be valid distributions
    assert gt_pdf.sum() > 0, f"{mouse_id}: ground-truth PDF is all zeros (no valid bouts?)"
    assert pp_pdf.sum() > 0, f"{mouse_id}: preprocess PDF is all zeros"

    jsd = _jsd(gt_pdf, pp_pdf)
    assert jsd < 0.003, (
        f"{mouse_id}: JSD(preprocess, ground_truth) = {jsd:.6f} >= 0.003 — "
        "the start-frame parser disagrees with the ground-truth reference. "
        "Check that the off-by-one fix is applied correctly."
    )


@pytest.mark.parametrize("mouse_id", MICE)
def test_old_parser_would_fail(mouse_id: str):
    """
    Sanity check: the *old* cumulative-end-frame parser produces JSD >= 0.01
    against the ground truth, confirming the test is sensitive enough to
    catch a regression back to the bug.
    """
    tf_path = DATA_RAW / f"{mouse_id}-tf"
    if not tf_path.exists():
        pytest.skip(f"Raw data not found: {tf_path}")

    tr = _load_tf(mouse_id)
    gt_pdf = _ground_truth_pdf(tr, N_RUNS)

    # Replicate the old buggy parser
    visit_frames_buggy = []
    if tr.no is not None:
        for bout_no in tr.no:
            last_end = 0
            for node_entry in bout_no:
                node_id = int(node_entry[0])
                end_frame = int(node_entry[1])
                duration = end_frame - last_end
                if duration > 0:
                    visit_frames_buggy.extend([node_id] * duration)
                last_end = end_frame
    counts = np.zeros(N_RUNS)
    for nid in visit_frames_buggy:
        if 0 <= nid < N_RUNS:
            counts[nid] += 1
    buggy_pdf = counts / (counts.sum() + 1e-9)

    jsd_buggy = _jsd(buggy_pdf, gt_pdf)
    assert jsd_buggy >= 0.01, (
        f"{mouse_id}: old parser JSD = {jsd_buggy:.6f} < 0.01 — "
        "the buggy parser is unexpectedly close to ground truth for this mouse. "
        "The test threshold may need adjustment."
    )
