#!/bin/bash
#SBATCH -J reotcsc_2263
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:0
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_replay_decoded_real_tokens_openbidanchor_txncomplete_splitcancel_002263XSHE.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
PROCESSED="/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/processed_real_flow"
STREAM="/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream"
STOCK_TAG="002263XSHE"
FLOW="openbidanchor_txncomplete_splitcancel"
DAY="20250710"
JLIB=$(ls -t "$PROCESSED"/final_result_for_merge_realflow_${FLOW}_${DAY}_${STOCK_TAG}_*.joblib 2>/dev/null | head -1)
BIN=$(ls -t "$PROCESSED"/bin_record_realflow_${FLOW}_${DAY}_${STOCK_TAG}_*.json 2>/dev/null | head -1)
REF=$(ls -td "$STREAM"/fixed_start_realflow_generate_lobster_${FLOW}_${STOCK_TAG}_* 2>/dev/null | head -1)
if [[ -z "$JLIB" || -z "$BIN" || -z "$REF" ]]; then
  echo "Missing preprocess or clean-ref dir for ${STOCK_TAG} (need ${FLOW})."
  exit 1
fi

python -u scripts/hist_script/replay_decoded_real_tokens_openbidanchor_txncomplete_fixed_start.py \
  --stock 002263_XSHE \
  --split-cancel-sides \
  --processed-real-flow-path "$JLIB" \
  --bin-record-path "$BIN" \
  --real-ref-dir "$REF"
