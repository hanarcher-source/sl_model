#!/bin/bash

#SBATCH -J rgrot_002721
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:0
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_generate_lobster_stream_real_openbidanchor_txncomplete_002721XSHE_%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
PROC_DIR=$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete
TAG=002721XSHE
STOCK=002721_XSHE

PROC=$(ls -t $PROC_DIR/final_result_for_merge_realflow_openbidanchor_txncomplete_20250710_${TAG}_*.joblib 2>/dev/null | head -n 1 || true)
if [[ -z "$PROC" ]]; then
  echo "Could not resolve processed realflow for TAG=$TAG"
  echo "PROC=$PROC"
  exit 3
fi

cd "$ROOT"
python -u scripts/hist_script/generate_lobster_stream_real_openbidanchor_txncomplete_fixed_start.py \
  --stock "$STOCK" \
  --processed-real-flow-path "$PROC"

