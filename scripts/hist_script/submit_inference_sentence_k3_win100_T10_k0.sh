#!/bin/bash
# Evaluate sentence_k3 win100 checkpoints under fixed sampling:
#   sample=on, temp=1.0, top-k=0
#
# Writes to:
#   saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/sentence_k3_win100/

set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model/scripts/hist_script

stocks=("000617_XSHE" "000981_XSHE" "002263_XSHE" "002366_XSHE")
tags=("000617XSHE" "000981XSHE" "002263XSHE" "002366XSHE")

TEMP=1.0
TOPK=0
SWEEP_TAG="sentence_k3_win100_T10_k0_ep3"

echo "Submitting sentence_k3 win100 evals (T10_k0)..."
for i in "${!stocks[@]}"; do
  STOCK="${stocks[$i]}" TAG="${tags[$i]}" TEMP="$TEMP" TOPK="$TOPK" SWEEP_TAG="$SWEEP_TAG" \
    sbatch "$ROOT/run_inference_sentence_k3_win100_sweep_one.sh"
done
echo "Done."

