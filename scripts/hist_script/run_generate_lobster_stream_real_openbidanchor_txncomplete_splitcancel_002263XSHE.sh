#!/bin/bash
#SBATCH -J rgrotsc_2263
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:0
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_generate_lobster_stream_real_openbidanchor_txncomplete_splitcancel_002263XSHE.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
PROCESSED="/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/processed_real_flow"
STOCK_TAG="002263XSHE"
FLOW="openbidanchor_txncomplete_splitcancel"
DAY="20250710"
JLIB=$(ls -t "$PROCESSED"/final_result_for_merge_realflow_${FLOW}_${DAY}_${STOCK_TAG}_*.joblib 2>/dev/null | head -1)
if [[ -z "$JLIB" ]]; then
  echo "No preprocess joblib matching ${FLOW}_${DAY}_${STOCK_TAG}; run preprocess splitcancel first."
  exit 1
fi

python -u scripts/hist_script/generate_lobster_stream_real_openbidanchor_txncomplete_fixed_start.py \
  --stock 002263_XSHE \
  --split-cancel-sides \
  --processed-real-flow-path "$JLIB"
