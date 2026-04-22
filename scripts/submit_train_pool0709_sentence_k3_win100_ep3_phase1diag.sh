#!/bin/bash
# Phase-1 diagnostics re-run for the clean sentence K=3 baseline (win100, ep3).
# Adds router health logging + JSONL + prefix shuffle probe.
#
# Note: checkpoints won't overwrite (timestamped), but we use distinct Slurm logs.

set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model/scripts

echo "Submitting sentence_k3 win100 ep3 Phase-1 diagnostics (4 stocks)..."
sbatch "$ROOT/run_train_pool0709_sentence_k3_win100_ep3_phase1diag_000617XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_k3_win100_ep3_phase1diag_000981XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_k3_win100_ep3_phase1diag_002263XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_k3_win100_ep3_phase1diag_002366XSHE.sh"
echo "Done."

