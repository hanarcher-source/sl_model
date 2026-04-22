#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --reservation=finai

set -euo pipefail

if [[ -z "${STOCK:-}" || -z "${TAG:-}" || -z "${TEMP:-}" || -z "${TOPK:-}" || -z "${SWEEP_TAG:-}" ]]; then
  echo "Missing env vars. Need: STOCK, TAG, TEMP, TOPK, SWEEP_TAG"
  exit 2
fi

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model

PROC_DIR=$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete
PROC=$(ls -t $PROC_DIR/final_result_for_merge_realflow_openbidanchor_txncomplete_20250710_${TAG}_*.joblib 2>/dev/null | head -n 1 || true)
BIN=$(ls -t $PROC_DIR/bin_record_realflow_openbidanchor_txncomplete_20250710_${TAG}_*.json 2>/dev/null | head -n 1 || true)
REF=$(ls -td $ROOT/saved_LOB_stream/fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_${TAG}_* 2>/dev/null | head -n 1 || true)

CKPT=$(ls -t $ROOT/training_runs/pool_0709_0710_train0709_blank_gpt2_dynamic_anchor_variants_win50/${TAG}/dyn_attnpool_topk/model_cache/*_best.pt 2>/dev/null | head -n 1 || true)

if [[ -z "$PROC" || -z "$BIN" || -z "$REF" || -z "$CKPT" ]]; then
  echo "Could not resolve required inputs for TAG=$TAG"
  echo "PROC=$PROC"
  echo "BIN=$BIN"
  echo "REF=$REF"
  echo "CKPT=$CKPT"
  exit 3
fi

OUT=$ROOT/saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/dyn_attnpool_topk4/${SWEEP_TAG}
mkdir -p "$OUT"

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
  --model-variant dyn_attnpool_topk \
  --topk-anchors 4 \
  --anchor-count 128 \
  --sample \
  --temperature "$TEMP" \
  --top-k "$TOPK"

