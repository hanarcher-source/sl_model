#!/bin/bash
#SBATCH -J egctxsc_2263
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/eval_against_clean_refs/002263XSHE_txncomplete_splitcancel_vs_cleanref.log
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
mkdir -p logs/eval_against_clean_refs

STREAM="/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream"
STOCK_TAG="002263XSHE"
GEN=$(ls -td "$STREAM"/fixed_start_decoded_real_tokens_openbidanchor_txncomplete_splitcancel_${STOCK_TAG}_* 2>/dev/null | head -1)
REF=$(ls -td "$STREAM"/fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_splitcancel_${STOCK_TAG}_* 2>/dev/null | head -1)
if [[ -z "$GEN" || -z "$REF" ]]; then
  echo "Missing replay or clean-ref experiment dirs for ${STOCK_TAG}."
  exit 1
fi

python -u scripts/hist_script/eval_generated_stream.py "$GEN" --real_ref_dir "$REF"
