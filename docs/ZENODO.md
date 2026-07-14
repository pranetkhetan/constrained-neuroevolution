# Zenodo deposit — preparation and upload

This document describes how to assemble and upload the large Tier-2 / Tier-3 data
archive that backs the repository. The repository itself (code, manuscript, distilled
`best_agents.pkl`, slim `B_results.pkl`, `analysis/*.pkl`) is fully sufficient for
**Tier-1** reproduction; the Zenodo deposit is only needed to **re-derive the analysis
intermediates** or **audit the full evolutionary runs**.

## What goes on Zenodo (and what does not)

| Include in Zenodo | Size | Source path |
|-------------------|------|-------------|
| Full per-mouse agent runs (54 dirs × 150 generations) | ~8.7 GB | `data/agents/` |
| Full generalist runs (15 dirs × 150 generations) | ~2.5 GB | `data/generalist/` |
| Full per-frame activations (all 54 agents) | ~116 MB | `analysis/activity_embeddings/B_results.pkl` |

**Do NOT upload:**
- Raw Rosenberg trajectories (`data/raw/`, `data/mouse_B5.pkl`, `data/mouse_D3.pkl`) —
  licensing; point users to the original Rosenberg et al. (2021) source instead.
- `data/archive_buggy_*` — internal recovery archives, no scientific content.
- Anything already in the GitHub repo (code, paper, distilled/slim/derived pkls).

> **Optional size reduction:** each `gen_150/` folder contains a `best_agent.mp4`
> (rendered trajectory video, not used by any analysis). Excluding `*.mp4` trims the
> agent archive noticeably with no loss of reproducibility.

## Archive layout (must match `NEUROEVO_DATA_DIR` / `NEUROEVO_ANALYSIS_DIR`)

Package so that, once unzipped, the directory tree mirrors the repo's `data/` and
`analysis/`. A user then sets two environment variables (see the repo README) and the
code reads from there with no source edits.

```
constrained-neuroevolution-data-v1/
├── data/
│   ├── agents/
│   │   ├── results_B5_r1/ ... results_D9_r6/      (54 dirs, gen_1 … gen_150 each)
│   │   └── ...
│   └── generalist/
│       └── results_r0/ ... results_r14/           (15 dirs)
└── analysis/
    └── activity_embeddings/
        └── B_results.pkl                          (full, ~116 MB)
```

A reviewer sets:
```bash
export NEUROEVO_DATA_DIR=/path/to/constrained-neuroevolution-data-v1/data
export NEUROEVO_ANALYSIS_DIR=/path/to/constrained-neuroevolution-data-v1/analysis
```

## Build the archives

From the **working repository** (the one that contains the full `data/agents`):

```bash
# 1. (optional) strip the unused trajectory videos to shrink the archive
find data/agents data/generalist -name 'best_agent.mp4' -delete   # safe: not read by analysis

# 2. assemble a clean staging tree mirroring repo layout
STAGE=constrained-neuroevolution-data-v1
mkdir -p "$STAGE/data" "$STAGE/analysis/activity_embeddings"
cp -r data/agents      "$STAGE/data/"
cp -r data/generalist  "$STAGE/data/"
cp analysis/activity_embeddings/B_results.pkl "$STAGE/analysis/activity_embeddings/"

# 3. zip (split into parts if the host limits single-file size)
#    Zenodo allows up to 50 GB per record; a single zip is fine here (~11 GB).
zip -r -s 0 "$STAGE.zip" "$STAGE"
sha256sum "$STAGE.zip" > "$STAGE.zip.sha256"
```

## Upload to Zenodo

1. Create a new record at <https://zenodo.org/uploads/new>.
2. Upload `constrained-neuroevolution-data-v1.zip` (+ the `.sha256`).
3. Metadata:
   - **Title:** *Constrained Neuroevolution — full evolved-agent runs and activations*
   - **Authors:** Pranet Khetan; Aditya Asopa
   - **Related identifier:** "is supplement to" → the GitHub repo URL and the
     bioRxiv DOI once available.
   - **License:** match the repository `LICENSE`.
   - **Version:** `v1`.
4. **Reserve the DOI** before publishing (Zenodo offers a pre-reserved DOI).
5. Insert the reserved DOI into:
   - `README.md`           (`[ZENODO-DOI]`, 3 places)
   - `CITATION.cff`        (`[ZENODO-DOI]`)
   - `paper_v2/latex/main_v2.tex` (`[ZENODO-DOI]` in the Data Availability statement)
   then recompile the paper and make the follow-up commit.

## Verifying the deposit

After upload, in a clean checkout with the archive downloaded and the two env vars set:

```bash
python scripts/extract_best_agents.py    # full agents -> data/best_agents.pkl (should match the shipped one)
python scripts/slim_b_results.py         # full B_results -> slim copy (should match the shipped one)
python scripts/embedding_robustness.py   # 6-metric robustness (should reproduce the shipped pkl)
```

If these reproduce the repo's shipped artifacts, the deposit + repo are consistent.
