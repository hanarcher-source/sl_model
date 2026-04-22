#!/bin/bash
# Sentence K=3 baseline, win50, ep3: router_health + anchor_geometry only (no prefix shuffle).
# Checkpoints: training_runs/pool_0709_0710_train0709_sentence_k3_baseline_win50_ep3_phase1diag/<TAG>/

set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model/scripts

echo "Submitting sentence_k3 win50 ep3 Phase-1 diagnostics (4 stocks, no shuffle probe)..."
sbatch "$ROOT/run_train_pool0709_sentence_k3_win50_ep3_phase1diag_000617XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_k3_win50_ep3_phase1diag_000981XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_k3_win50_ep3_phase1diag_002263XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_k3_win50_ep3_phase1diag_002366XSHE.sh"
echo "Done."
