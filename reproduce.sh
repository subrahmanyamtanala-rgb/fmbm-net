#!/usr/bin/env bash
# reproduce.sh — one-command PDF compilation for FMBM-Net
set -e
cd "$(dirname "$0")/paper"
echo "[1/5] Generating figures..."
python ../src/generate_figures.py
echo "[2/5] First LaTeX pass..."
pdflatex -interaction=nonstopmode FMBM_Net_IEEE.tex > /dev/null
echo "[3/5] BibTeX..."
bibtex FMBM_Net_IEEE > /dev/null
echo "[4/5] Second LaTeX pass..."
pdflatex -interaction=nonstopmode FMBM_Net_IEEE.tex > /dev/null
echo "[5/5] Final LaTeX pass..."
pdflatex -interaction=nonstopmode FMBM_Net_IEEE.tex | grep "Output written"
echo "Done — paper/FMBM_Net_IEEE_Paper.pdf"
