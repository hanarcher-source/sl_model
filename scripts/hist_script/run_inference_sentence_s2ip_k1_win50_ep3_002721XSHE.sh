#!/bin/bash
#SBATCH -J ie_s2ip_k1_2721
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_inference_sentence_s2ip_k1_win50_ep3_002721XSHE_%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
PROC_DIR=$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete
TAG=002721XSHE
STOCK=002721_XSHE

PROC=$(ls -t $PROC_DIR/final_result_for_merge_realflow_openbidanchor_txncomplete_20250710_${TAG}_*.joblib 2>/dev/null | head -n 1 || true)
BIN=$(ls -t $PROC_DIR/bin_record_realflow_openbidanchor_txncomplete_20250710_${TAG}_*.json 2>/dev/null | head -n 1 || true)
REF=$(ls -td $ROOT/saved_LOB_stream/fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_${TAG}_* 2>/dev/null | head -n 1 || true)

CKPT=$ROOT/training_runs/pool_0709_0710_train0709_sentence_preset_s2ip_win50_k1/${TAG}/model_cache/preGPT2sentS2IP_20250709_txncomplete_${TAG}_win50_k1_20260410_203045_best.pt

OUT=$ROOT/saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/sentence_preset_s2ip_k1/sentence_s2ip_k1_T10_k0_win50
mkdir -p "$OUT"

if [[ -z "$PROC" || -z "$BIN" || -z "$REF" || ! -f "$CKPT" ]]; then
  echo "Could not resolve required inputs."
  echo "PROC=$PROC"
  echo "BIN=$BIN"
  echo "REF=$REF"
  echo "CKPT=$CKPT"
  exit 3
fi

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
  --model-variant sentence_preset_s2ip \
  --topk-anchors 1 \
  --sample \
  --temperature 1.0 \
  --top-k 0

