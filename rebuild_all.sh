#!/usr/bin/env bash
# rebuild_all.sh
# Full pipeline: (optional) stats extraction → figures → LaTeX compile.
#
# Usage:
#   bash rebuild_all.sh              # figures + supp_A6 + LaTeX only
#   bash rebuild_all.sh --extract    # also re-extracts stats macros first
#   bash rebuild_all.sh --figs-only  # skip LaTeX, just regenerate figures
#
# Run from repo root via Git Bash.

set -euo pipefail

# Python interpreter: override with `PYTHON=/path/to/python bash rebuild_all.sh`.
PYTHON="${PYTHON:-python}"
PROJECT="$(cd "$(dirname "$0")" && pwd)"
LATEX_DIR="$PROJECT/paper_v2/latex"

EXTRACT=false
FIGS_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --extract)   EXTRACT=true ;;
        --figs-only) FIGS_ONLY=true ;;
    esac
done

if $EXTRACT; then
    echo "========================================================"
    echo " 1/4  Extract stats (embedding robustness + build_paper_stats)"
    echo "========================================================"
    # 6-metric embedding robustness -> embedding_robustness.pkl (feeds 5 macros).
    "$PYTHON" "$PROJECT/scripts/embedding_robustness.py"
    "$PYTHON" "$PROJECT/scripts/build_paper_stats.py"
else
    echo "[skip] Stats extraction (pass --extract to enable)"
fi

echo "========================================================"
echo " 2/4  Generate main + supplementary figures"
echo "========================================================"
"$PYTHON" "$PROJECT/scripts/generate_all_figures_paper_v2.py"

echo "========================================================"
echo " 3/4  Generate supp_A6 per-metric bar figure"
echo "========================================================"
"$PYTHON" "$PROJECT/scripts/supp_analyses.py" --run A6

if $FIGS_ONLY; then
    echo "[skip] LaTeX compile (--figs-only mode)"
    echo "Done."
    exit 0
fi

echo "========================================================"
echo " 4/4  Compile LaTeX (3-pass + bibtex)"
echo "========================================================"
cd "$LATEX_DIR"
pdflatex -interaction=nonstopmode main_v2.tex
bibtex main_v2
pdflatex -interaction=nonstopmode main_v2.tex
pdflatex -interaction=nonstopmode main_v2.tex

echo ""
echo "Done. Output: $LATEX_DIR/main_v2.pdf"
