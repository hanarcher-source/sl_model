#!/bin/bash
set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model
JOB=$ROOT/scripts/hist_script/run_inference_blankgpt2_pool0709_0710_eval_0710_sweep_one.sh

mkdir -p $ROOT/logs/eval_pool0709_0710

stocks=("000617_XSHE" "000981_XSHE" "002263_XSHE" "002366_XSHE")

# Batch 2: top-k sweep, fixed temperature=1.3
params=(
  "T13_k100 1.3 100"
  "T13_k400 1.3 400"
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

echo "Submitted batch2 topk sweep (temp=1.3): k=100 and k=400."

