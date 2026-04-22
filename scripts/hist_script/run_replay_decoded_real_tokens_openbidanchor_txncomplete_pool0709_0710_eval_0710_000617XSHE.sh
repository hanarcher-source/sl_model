#!/bin/bash
#SBATCH -J reotcP_617
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:0
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_replay_decoded_real_tokens_openbidanchor_txncomplete_pool0709_0710_eval_0710_000617XSHE.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
PROC=$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete/final_result_for_merge_realflow_openbidanchor_txncomplete_20250710_000617XSHE_20260406_1233.joblib
BIN=$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete/bin_record_realflow_openbidanchor_txncomplete_20250710_000617XSHE_20260406_1233.json
REF=$ROOT/saved_LOB_stream/fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_000617XSHE_20260405_132652
OUT=$ROOT/saved_LOB_stream/pool_0709_0710_eval_0710

cd "$ROOT"
python -u scripts/hist_script/replay_decoded_real_tokens_openbidanchor_txncomplete_fixed_start.py \
  --stock 000617_XSHE \
  --processed-real-flow-path "$PROC" \
  --bin-record-path "$BIN" \
  --real-ref-dir "$REF" \
  --lob-snap-path /finance_ML/zhanghaohan/LOB_data/20250710/mdl_6_28_0.csv \
  --trade-date-str 2025-07-10 \
  --start-time 10:00:00 \
  --sim-lookahead-minutes 10 \
  --base-out-dir "$OUT"

