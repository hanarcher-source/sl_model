#!/bin/bash
#SBATCH -J ie_bsphP_2263_t07
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_eval_bookstate_1s_0710_temp0p7_pretrained_002263XSHE_%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"

TAG=002263XSHE
STOCK=002263_XSHE
PROC_DIR="$ROOT/saved_LOB_stream/processed_book_state/pool_0710_bookstate_anchor5m_mdl628_apply0709bins"
DATA_JOB=$(ls -t "$PROC_DIR"/bookstate_20250710_mdl628_anchor5m_P41_V31_*.joblib 2>/dev/null | head -n 1 || true)
META=$(ls -t "$PROC_DIR"/bookstate_20250710_mdl628_anchor5m_P41_V31_*_meta.json 2>/dev/null | head -n 1 || true)

TRAIN_ROOT="$ROOT/training_runs/pool_0709_bookstate_parallelheads_anchor5m_pretrained"
CKPT=$(ls -t "$TRAIN_ROOT/$TAG/"*best.pt 2>/dev/null | head -n 1 || true)

OUT="$ROOT/saved_LOB_stream/pool_0710_bookstate_eval_1s/bsph_temp0p7_blank_vs_pretrained"
mkdir -p "$OUT"

if [[ -z "$DATA_JOB" || -z "$CKPT" || -z "$META" ]]; then
  echo "Missing input TAG=$TAG DATA_JOB=$DATA_JOB CKPT=$CKPT META=$META"; exit 3
fi

python -u scripts/inference_eval_bookstate_parallelheads_1s_fixed_start.py \
  --stock "$STOCK" \
  --checkpoint "$CKPT" \
  --data-joblib "$DATA_JOB" \
  --meta-json "$META" \
  --output-dir "$OUT" \
  --start-time 10:00:00 \
  --context-sec 60 \
  --horizon-sec 300 \
  --codebook-size 1271 \
  --P 41 \
  --V 31 \
  --sample \
  --temperature 0.7 \
  --run-tag pretrained_temp0p7

