#!/bin/bash

#SBATCH -J egceq_617
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/eval_against_clean_refs/000617XSHE_exactpq_vs_cleanref.log
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
mkdir -p logs/eval_against_clean_refs

python -u scripts/hist_script/eval_generated_stream.py \
  /finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/fixed_start_decoded_real_tokens_openbidanchor_exactprice_exactqty_000617XSHE_20260405_123424 \
  --real_ref_dir /finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_000617XSHE_20260405_132652