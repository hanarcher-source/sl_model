#!/bin/bash
# Fixed sampling regime for fair comparison:
#   sample=on, temperature=1.0 (T10), top-k=0 (k0)
#
# Submits:
#   - blank (no_anchor) : 4 stocks
#   - sentence_preset_s2ip: 4 stocks × (K_PREPEND=3,5)

set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model/scripts/hist_script

stocks=("000617_XSHE" "000981_XSHE" "002263_XSHE" "002366_XSHE")
tags=("000617XSHE" "000981XSHE" "002263XSHE" "002366XSHE")

TEMP=1.0
TOPK=0

echo "Submitting blank fixed-sampling (T10_k0)..."
SWEEP_TAG_BLANK="blank_T10_k0_win50"
for i in "${!stocks[@]}"; do
  STOCK="${stocks[$i]}" TAG="${tags[$i]}" TEMP="$TEMP" TOPK="$TOPK" SWEEP_TAG="$SWEEP_TAG_BLANK" \
    sbatch "$ROOT/run_inference_blank_fixed_sampling_sweep_one.sh"
done

echo "Submitting sentence_preset_s2ip fixed-sampling (T10_k0) K=3..."
SWEEP_TAG_S3="sentence_s2ip_k3_T10_k0_win50"
for i in "${!stocks[@]}"; do
  STOCK="${stocks[$i]}" TAG="${tags[$i]}" TEMP="$TEMP" TOPK="$TOPK" SWEEP_TAG="$SWEEP_TAG_S3" K_PREPEND=3 \
    sbatch "$ROOT/run_inference_sentence_preset_s2ip_sweep_one.sh"
done

echo "Submitting sentence_preset_s2ip fixed-sampling (T10_k0) K=5..."
SWEEP_TAG_S5="sentence_s2ip_k5_T10_k0_win50"
for i in "${!stocks[@]}"; do
  STOCK="${stocks[$i]}" TAG="${tags[$i]}" TEMP="$TEMP" TOPK="$TOPK" SWEEP_TAG="$SWEEP_TAG_S5" K_PREPEND=5 \
    sbatch "$ROOT/run_inference_sentence_preset_s2ip_sweep_one.sh"
done

echo "Done."

