#!/bin/bash
# Submit inference + LOBSTER replay + eval for blank GPT-2 trained with window 100 and 200
# on pooled 0709+0710 preprocess, evaluated on 0710. Sampling params = per-stock winners from
# LOB-Bench-style W_mean aggregate over the temp/top-k sweep (see select_best_sweep_setting_lobbench_aggregate.py).
#
# 000617 / 002263 / 002366: T13_k0  (temp=1.3, top_k=0)
# 000981: T13_k400 (temp=1.3, top_k=400)
#
# Output dirs: saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/lbmean_<SETTING>_blank_win<W>/
# Logs: logs/eval_pool0709_0710/run_inference_lbmean_<SETTING>_blank_win<W>_<TAG>.out

set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model
JOB=$ROOT/scripts/hist_script/run_inference_eval_pool0709_0710_from_training_one.sh

mkdir -p "$ROOT/logs/eval_pool0709_0710"

stocks=("000617_XSHE" "000981_XSHE" "002263_XSHE" "002366_XSHE")

temp_000617=1.3; topk_000617=0;   set_000617="T13_k0"
temp_000981=1.3; topk_000981=400; set_000981="T13_k400"
temp_002263=1.3; topk_002263=0;   set_002263="T13_k0"
temp_002366=1.3; topk_002366=0;   set_002366="T13_k0"

params_for_stock() {
  local stock=$1
  case "$stock" in
    000617_XSHE) echo "$temp_000617 $topk_000617 $set_000617" ;;
    000981_XSHE) echo "$temp_000981 $topk_000981 $set_000981" ;;
    002263_XSHE) echo "$temp_002263 $topk_002263 $set_002263" ;;
    002366_XSHE) echo "$temp_002366 $topk_002366 $set_002366" ;;
    *) echo "Unknown stock $stock" >&2; exit 2 ;;
  esac
}

TRAIN_WIN100=$ROOT/training_runs/pool_0709_0710_train0709_blank_gpt2_win100
TRAIN_WIN200=$ROOT/training_runs/pool_0709_0710_train0709_blank_gpt2_win200

for stock in "${stocks[@]}"; do
  tag="${stock/_/}"
  read -r temp topk setting <<<"$(params_for_stock "$stock")"

  for pair in "100:$TRAIN_WIN100" "200:$TRAIN_WIN200"; do
    win="${pair%%:*}"
    trroot="${pair##*:}"
    sweep_tag="lbmean_${setting}_blank_win${win}"
    name="inf_${sweep_tag}_${tag}"
    out="$ROOT/logs/eval_pool0709_0710/run_inference_${sweep_tag}_${tag}.out"
    STOCK="$stock" TAG="$tag" TEMP="$temp" TOPK="$topk" SWEEP_TAG="$sweep_tag" \
      WINDOW_LEN="$win" TRAIN_ROOT="$trroot" \
      sbatch -J "$name" -o "$out" "$JOB"
  done
done

echo "Submitted 8 inference+eval jobs: blank GPT-2 win100 + win200, LOB-Bench W_mean best sampling per stock."
