#!/bin/bash
#SBATCH -J infP_2263
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_inference_blankgpt2_pool0709_0710_eval_0710_002263XSHE.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

STOCK=002263_XSHE
TAG=002263XSHE
ROOT=/finance_ML/zhanghaohan/stock_language_model

PROC=$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete/final_result_for_merge_realflow_openbidanchor_txncomplete_20250710_${TAG}_20260406_1233.joblib
BIN=$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete/bin_record_realflow_openbidanchor_txncomplete_20250710_${TAG}_20260406_1233.json
REF=$ROOT/saved_LOB_stream/fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_${TAG}_20260405_132652
CKPT=$ROOT/training_runs/pool_0709_0710_train0709_blank_gpt2/${TAG}/model_cache/blankGPT2_20250709_txncomplete_${TAG}_win50_20260406_132212_best.pt
OUT=$ROOT/saved_LOB_stream/pool_0709_0710_eval_0710

cd "$ROOT"
python -u scripts/hist_script/inference_replay_blankgpt2_openbidanchor_txncomplete_fixed_start.py \
  --stock "$STOCK" \
  --checkpoint "$CKPT" \
  --processed-real-flow-path "$PROC" \
  --bin-record-path "$BIN" \
  --real-ref-dir "$REF" \
  --lob-snap-path /finance_ML/zhanghaohan/LOB_data/20250710/mdl_6_28_0.csv \
  --trade-date-str 2025-07-10 \
  --start-time 10:00:00 \
  --sim-lookahead-minutes 10 \
  --base-out-dir "$OUT" \
  --window-len 50 \
  --vocab-size 40560 \
  --sample \
  --temperature 1.0

