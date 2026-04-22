#!/bin/bash
#SBATCH -J inf_pool_one
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --reservation=finai
#
# Generic pooled-0710 inference+replay+eval runner.
# Intended to be launched via `sbatch ... this_script.sh` with environment variables set:
#   STOCK      e.g. 000617_XSHE
#   TAG        e.g. 000617XSHE
#   TEMP       e.g. 1.3
#   TOPK       e.g. 400
#   SWEEP_TAG  e.g. lbmean_T13_k400_pretrained_win50  (used only for output directory tagging)
#   WINDOW_LEN e.g. 50
#   TRAIN_ROOT e.g. /.../training_runs/pool_0709_0710_train0709_pretrained_gpt2_win50
#
set -euo pipefail

if [ -z "${STOCK:-}" ] || [ -z "${TAG:-}" ] || [ -z "${TEMP:-}" ] || [ -z "${TOPK:-}" ] || [ -z "${SWEEP_TAG:-}" ] || [ -z "${WINDOW_LEN:-}" ] || [ -z "${TRAIN_ROOT:-}" ]; then
  echo "Missing env vars. Need: STOCK TAG TEMP TOPK SWEEP_TAG WINDOW_LEN TRAIN_ROOT"
  exit 2
fi

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model

PROC_DIR="$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete"
PROC="$PROC_DIR/final_result_for_merge_realflow_openbidanchor_txncomplete_20250710_${TAG}_*.joblib"
BIN="$PROC_DIR/bin_record_realflow_openbidanchor_txncomplete_20250710_${TAG}_*.json"
REF="$ROOT/saved_LOB_stream/fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_${TAG}_*"

shopt -s nullglob
PROC_ARR=($PROC)
BIN_ARR=($BIN)
REF_ARR=($REF)
if [ ${#PROC_ARR[@]} -eq 0 ] || [ ${#BIN_ARR[@]} -eq 0 ] || [ ${#REF_ARR[@]} -eq 0 ]; then
  echo "Missing PROC/BIN/REF for TAG=$TAG under pooled preprocess."
  echo "PROC glob: $PROC"
  echo "BIN  glob: $BIN"
  echo "REF  glob: $REF"
  exit 3
fi
PROC_PATH="${PROC_ARR[0]}"
BIN_PATH="${BIN_ARR[0]}"
REF_DIR="${REF_ARR[0]}"

CKPT_GLOB="$TRAIN_ROOT/${TAG}/model_cache/"*"_best.pt"
CKPT_ARR=($CKPT_GLOB)
if [ ${#CKPT_ARR[@]} -eq 0 ]; then
  echo "No checkpoint found: $CKPT_GLOB"
  exit 4
fi
CKPT_PATH="${CKPT_ARR[0]}"

OUT="$ROOT/saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/${SWEEP_TAG}"
mkdir -p "$OUT"

cd "$ROOT"
python -u scripts/hist_script/inference_replay_blankgpt2_openbidanchor_txncomplete_fixed_start.py \
  --stock "$STOCK" \
  --checkpoint "$CKPT_PATH" \
  --processed-real-flow-path "$PROC_PATH" \
  --bin-record-path "$BIN_PATH" \
  --real-ref-dir "$REF_DIR" \
  --lob-snap-path /finance_ML/zhanghaohan/LOB_data/20250710/mdl_6_28_0.csv \
  --trade-date-str 2025-07-10 \
  --start-time 10:00:00 \
  --sim-lookahead-minutes 10 \
  --base-out-dir "$OUT" \
  --window-len "$WINDOW_LEN" \
  --vocab-size 40560 \
  --sample \
  --temperature "$TEMP" \
  --top-k "$TOPK"

