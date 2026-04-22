#!/bin/bash
# Clean window-length ablation for the baseline sentence preset model:
#   - sentence preset anchors, hard top-K=3 prepend
#   - NO centering, NO separate projections, anchors frozen (baseline)
#   - epochs=3
#
# Two sets of four stocks:
#   - window_len=100 (batch_size=256)
#   - window_len=150 (batch_size=128 to reduce GPU memory pressure)

set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model/scripts

echo "Submitting baseline sentence_k3 win100 (4 stocks)..."
sbatch "$ROOT/run_train_pool0709_sentence_k3_win100_ep3_000617XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_k3_win100_ep3_000981XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_k3_win100_ep3_002263XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_k3_win100_ep3_002366XSHE.sh"

echo "Submitting baseline sentence_k3 win150 (4 stocks)..."
sbatch "$ROOT/run_train_pool0709_sentence_k3_win150_ep3_000617XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_k3_win150_ep3_000981XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_k3_win150_ep3_002263XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_k3_win150_ep3_002366XSHE.sh"

echo "Done (8 sbatch calls)."

