#!/bin/bash
#SBATCH -J inf709b_2366
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_inference_blankgpt2_0710_txncomplete_002366XSHE.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

STOCK=002366_XSHE
TAG=002366XSHE
ROOT=/finance_ML/zhanghaohan/stock_language_model
PROC=$ROOT/saved_LOB_stream/processed_real_flow/final_result_for_merge_realflow_openbidanchor_txncomplete_20250710_${TAG}_20260404_2207.joblib
BIN=$ROOT/saved_LOB_stream/processed_real_flow/bin_record_realflow_openbidanchor_txncomplete_20250710_${TAG}_20260404_2207.json
REF=$ROOT/saved_LOB_stream/fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_${TAG}_20260405_132652
BASELINE=$ROOT/saved_LOB_stream/fixed_start_decoded_real_tokens_openbidanchor_txncomplete_${TAG}_20260404_222446/metrics_summary.json
CKPT=$ROOT/training_runs/20250709_openbidanchor_txncomplete_blank_gpt2/${TAG}/model_cache/blankGPT2_20250709_txncomplete_${TAG}_win50_20260406_103000_best.pt
OUT=$ROOT/training_runs/20250710_blankgpt2_model_inference_eval

cd "$ROOT"
python -u scripts/hist_script/inference_replay_blankgpt2_openbidanchor_txncomplete_fixed_start.py \
  --stock "$STOCK" \
  --checkpoint "$CKPT" \
  --processed-real-flow-path "$PROC" \
  --bin-record-path "$BIN" \
  --real-ref-dir "$REF" \
  --baseline-metrics-json "$BASELINE" \
  --lob-snap-path /finance_ML/zhanghaohan/LOB_data/20250710/mdl_6_28_0.csv \
  --trade-date-str 2025-07-10 \
  --start-time 10:00:00 \
  --sim-lookahead-minutes 10 \
  --base-out-dir "$OUT" \
  --window-len 50 \
  --vocab-size 40560
