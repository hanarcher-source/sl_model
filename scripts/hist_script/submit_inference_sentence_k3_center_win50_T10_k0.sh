#!/bin/bash
# Sentence K=3, centered anchors, win50: eval under fixed sampling (sample=on, T=1.0, top_k=0).
# Checkpoints: training_runs/pool_0709_0710_train0709_sentence_k3_baseline_win50_ep3_center_phase1diag/

set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model/scripts/hist_script

stocks=("000617_XSHE" "000981_XSHE" "002263_XSHE" "002366_XSHE")
tags=("000617XSHE" "000981XSHE" "002263XSHE" "002366XSHE")

TEMP=1.0
TOPK=0
SWEEP_TAG="sentence_k3_center_win50_T10_k0_ep3"

echo "Submitting sentence_k3 centered win50 evals (T10_k0)..."
for i in "${!stocks[@]}"; do
  STOCK="${stocks[$i]}" TAG="${tags[$i]}" TEMP="$TEMP" TOPK="$TOPK" SWEEP_TAG="$SWEEP_TAG" \
    sbatch "$ROOT/run_inference_sentence_k3_center_win50_sweep_one.sh"
done
echo "Done."
