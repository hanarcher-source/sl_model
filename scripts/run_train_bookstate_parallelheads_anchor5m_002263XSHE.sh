#!/bin/bash
#SBATCH -J tr_bsph_2263
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_train_bookstate_parallelheads_anchor5m_002263XSHE_%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"

TAG=002263XSHE
PROC_DIR="$ROOT/saved_LOB_stream/processed_book_state/pool_0709_bookstate_anchor5m_mdl628"
DATA_JOB=$(ls -t "$PROC_DIR"/bookstate_20250709_mdl628_anchor5m_P41_V31_*.joblib 2>/dev/null | head -n 1)
if [ -z "${DATA_JOB:-}" ]; then
  echo "[error] No preprocessed joblib found in $PROC_DIR"
  exit 1
fi

python -u scripts/train_bookstate_parallelheads_anchor5m.py \
  --data-joblib "$DATA_JOB" \
  --stock 002263_XSHE \
  --codebook-size 1271 \
  --context-sec 60 \
  --stride-sec 5 \
  --epochs 3 \
  --patience 3 \
  --batch-size 256 \
  --lr 2e-4 \
  --amp \
  --output-root "$ROOT/training_runs/pool_0709_bookstate_parallelheads_anchor5m"

