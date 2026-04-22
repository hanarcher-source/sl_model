#!/bin/bash
# Submit 8 eval jobs: 4 stocks × (Variant A vs Variant B)
# Fixed sampling regime: sample=on, temp=1.0, top-k=0.
#
# Variant A training root:
#   training_runs/pool_0709_0710_train0709_sentence_s2ip_A_center_sep_k3
# Variant B training root:
#   training_runs/pool_0709_0710_train0709_sentence_s2ip_B_trainable_center_sep_k3

set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model
HS=$ROOT/scripts/hist_script

stocks=("000617_XSHE" "000981_XSHE" "002263_XSHE" "002366_XSHE")
tags=("000617XSHE" "000981XSHE" "002263XSHE" "002366XSHE")

TEMP=1.0
TOPK=0

TRAIN_A="$ROOT/training_runs/pool_0709_0710_train0709_sentence_s2ip_A_center_sep_k3"
TRAIN_B="$ROOT/training_runs/pool_0709_0710_train0709_sentence_s2ip_B_trainable_center_sep_k3"

echo "Submitting sentence S2IP Variant A evals (T10_k0)..."
SWEEP_A="sentence_s2ip_varA_T10_k0_win50"
for i in "${!stocks[@]}"; do
  STOCK="${stocks[$i]}" TAG="${tags[$i]}" TEMP="$TEMP" TOPK="$TOPK" SWEEP_TAG="$SWEEP_A" TRAIN_ROOT="$TRAIN_A" \
    sbatch "$HS/run_inference_sentence_s2ip_AB_sweep_one.sh"
done

echo "Submitting sentence S2IP Variant B evals (T10_k0)..."
SWEEP_B="sentence_s2ip_varB_T10_k0_win50"
for i in "${!stocks[@]}"; do
  STOCK="${stocks[$i]}" TAG="${tags[$i]}" TEMP="$TEMP" TOPK="$TOPK" SWEEP_TAG="$SWEEP_B" TRAIN_ROOT="$TRAIN_B" \
    sbatch "$HS/run_inference_sentence_s2ip_AB_sweep_one.sh"
done

echo "Done."

