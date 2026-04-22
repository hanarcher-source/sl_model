#!/bin/bash
# Sentence K=3 baseline, win50, ep3: centered anchors + Phase-1 diagnostics.
# No other variations (anchors frozen, shared proj).

set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model/scripts

echo "Submitting sentence_k3 win50 ep3 centered-anchor Phase-1 diagnostics (4 stocks)..."
sbatch "$ROOT/run_train_pool0709_sentence_k3_win50_ep3_center_phase1diag_000617XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_k3_win50_ep3_center_phase1diag_000981XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_k3_win50_ep3_center_phase1diag_002263XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_k3_win50_ep3_center_phase1diag_002366XSHE.sh"
echo "Done."

