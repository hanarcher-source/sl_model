#!/bin/bash
set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model
JOB=$ROOT/scripts/hist_script/run_inference_blankgpt2_pool0709_0710_eval_0710_sweep_one.sh

mkdir -p $ROOT/logs/eval_pool0709_0710

stocks=("000617_XSHE" "000981_XSHE" "002263_XSHE" "002366_XSHE")

# Ablation 1: temperature only, top_k=0
params=(
  "T07_k0 0.7 0"
  "T13_k0 1.3 0"
)

for p in "${params[@]}"; do
  read -r sweep_tag temp topk <<<"$p"
  for stock in "${stocks[@]}"; do
    tag="${stock/_/}"
    name="sw_${sweep_tag}_${tag}"
    out="$ROOT/logs/eval_pool0709_0710/run_sweep_${sweep_tag}_${tag}.out"
    STOCK="$stock" TAG="$tag" TEMP="$temp" TOPK="$topk" SWEEP_TAG="$sweep_tag" \
      sbatch -J "$name" -o "$out" "$JOB"
  done
done

echo "Submitted ablation1 (temp-only) sweep jobs."

