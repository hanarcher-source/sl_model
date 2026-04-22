#!/bin/bash
set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model
JOB=$ROOT/scripts/hist_script/run_inference_eval_pool0709_0710_from_training_one.sh

mkdir -p $ROOT/logs/eval_pool0709_0710

# Best params per stock from multi-metric aggregate over sweep settings:
# 000617: T13_k0  (temp=1.3, topk=0)
# 000981: T10_k50 (temp=1.0, topk=50)
# 002263: T13_k0  (temp=1.3, topk=0)
# 002366: T13_k0  (temp=1.3, topk=0)

stocks=("000617_XSHE" "000981_XSHE" "002263_XSHE" "002366_XSHE")

temp_000617=1.3; topk_000617=0;   set_000617="T13_k0"
temp_000981=1.0; topk_000981=50;  set_000981="T10_k50"
temp_002263=1.3; topk_002263=0;   set_002263="T13_k0"
temp_002366=1.3; topk_002366=0;   set_002366="T13_k0"

function params_for_stock() {
  local stock=$1
  case "$stock" in
    000617_XSHE) echo "$temp_000617 $topk_000617 $set_000617" ;;
    000981_XSHE) echo "$temp_000981 $topk_000981 $set_000981" ;;
    002263_XSHE) echo "$temp_002263 $topk_002263 $set_002263" ;;
    002366_XSHE) echo "$temp_002366 $topk_002366 $set_002366" ;;
    *) echo "Unknown stock $stock" ; exit 2 ;;
  esac
}

# Variant A: pretrained backbone, window_len=50
TRAIN_ROOT_A=$ROOT/training_runs/pool_0709_0710_train0709_pretrained_gpt2_win50
WIN_A=50

# Variant B: blank backbone, window_len=200
TRAIN_ROOT_B=$ROOT/training_runs/pool_0709_0710_train0709_blank_gpt2_win200
WIN_B=200

for stock in "${stocks[@]}"; do
  tag="${stock/_/}"
  read -r temp topk setting <<<"$(params_for_stock "$stock")"

  sweep_tag_a="best_${setting}_pretrained_win${WIN_A}"
  name_a="inf_${sweep_tag_a}_${tag}"
  out_a="$ROOT/logs/eval_pool0709_0710/run_inference_${sweep_tag_a}_${tag}.out"
  STOCK="$stock" TAG="$tag" TEMP="$temp" TOPK="$topk" SWEEP_TAG="$sweep_tag_a" WINDOW_LEN="$WIN_A" TRAIN_ROOT="$TRAIN_ROOT_A" \
    sbatch -J "$name_a" -o "$out_a" "$JOB"

  sweep_tag_b="best_${setting}_blank_win${WIN_B}"
  name_b="inf_${sweep_tag_b}_${tag}"
  out_b="$ROOT/logs/eval_pool0709_0710/run_inference_${sweep_tag_b}_${tag}.out"
  STOCK="$stock" TAG="$tag" TEMP="$temp" TOPK="$topk" SWEEP_TAG="$sweep_tag_b" WINDOW_LEN="$WIN_B" TRAIN_ROOT="$TRAIN_ROOT_B" \
    sbatch -J "$name_b" -o "$out_b" "$JOB"
done

echo "Submitted 8 inference+eval jobs (pretrained win50 + blank win200) using best per-stock sampling params."

