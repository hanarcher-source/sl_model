#!/bin/bash
#SBATCH -J pp_bs628_0709
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_preprocess_bookstate_mdl628_anchor5m_20250709_%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"

python -u scripts/preprocess_bookstate_mdl628_anchor5m_bins_20250709.py \
  --day 20250709 \
  --stocks 000617_XSHE,002263_XSHE,002721_XSHE \
  --anchor-minutes 5 \
  --max-tick 20 \
  --vol-bins 31 \
  --chunksize 500000 \
  --output-dir "$ROOT/saved_LOB_stream/processed_book_state/pool_0709_bookstate_anchor5m_mdl628"

