#!/bin/bash
#SBATCH -J pp_bs628_0710
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_preprocess_bookstate_mdl628_anchor5m_20250710_apply0709bins_%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"

PROC_0709="$ROOT/saved_LOB_stream/processed_book_state/pool_0709_bookstate_anchor5m_mdl628"
META_0709=$(ls -t "$PROC_0709"/bookstate_20250709_mdl628_anchor5m_P41_V31_*_meta.json 2>/dev/null | head -n 1 || true)
if [[ -z "${META_0709:-}" ]]; then
  echo "[error] Could not find 0709 meta.json in $PROC_0709"
  exit 2
fi

python -u scripts/preprocess_bookstate_mdl628_anchor5m_bins_20250709.py \
  --day 20250710 \
  --stocks 000617_XSHE,002263_XSHE,002721_XSHE \
  --anchor-minutes 5 \
  --max-tick 20 \
  --vol-bins 31 \
  --vol-edges-from-meta "$META_0709" \
  --keep-raw \
  --chunksize 500000 \
  --output-dir "$ROOT/saved_LOB_stream/processed_book_state/pool_0710_bookstate_anchor5m_mdl628_apply0709bins"

