#!/bin/bash
#SBATCH -J ie_bsph_2721
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_inference_eval_bookstate_parallelheads_1s_fixed_start_002721XSHE_%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"

TAG=002721XSHE
STOCK=002721_XSHE
PROC_DIR="$ROOT/saved_LOB_stream/processed_book_state/pool_0709_bookstate_anchor5m_mdl628"
DATA_JOB=$(ls -t "$PROC_DIR"/bookstate_20250709_mdl628_anchor5m_P41_V31_*.joblib 2>/dev/null | head -n 1 || true)
META=$(ls -t "$PROC_DIR"/bookstate_20250709_mdl628_anchor5m_P41_V31_*_meta.json 2>/dev/null | head -n 1 || true)

TRAIN_ROOT="$ROOT/training_runs/pool_0709_bookstate_parallelheads_anchor5m"
CKPT=$(ls -t "$TRAIN_ROOT/$TAG/"*best.pt 2>/dev/null | head -n 1 || true)

OUT="$ROOT/saved_LOB_stream/pool_0709_bookstate_eval_1s/bsph_T600_ctx60"
mkdir -p "$OUT"

if [[ -z "$DATA_JOB" || -z "$CKPT" ]]; then
  echo "Missing input TAG=$TAG DATA_JOB=$DATA_JOB CKPT=$CKPT META=$META"; exit 3
fi

python -u scripts/inference_eval_bookstate_parallelheads_1s_fixed_start.py \
  --stock "$STOCK" \
  --checkpoint "$CKPT" \
  --data-joblib "$DATA_JOB" \
  --output-dir "$OUT" \
  --start-time 10:00:00 \
  --context-sec 60 \
  --horizon-sec 600 \
  --codebook-size 1271 \
  --P 41 \
  --V 31 \
  --sample \
  --temperature 1.0

