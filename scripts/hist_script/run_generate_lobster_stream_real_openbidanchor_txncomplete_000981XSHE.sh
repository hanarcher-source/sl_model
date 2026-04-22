#!/bin/bash

#SBATCH -J rgrot_000981
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:0
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_generate_lobster_stream_real_openbidanchor_txncomplete_000981XSHE.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
python -u scripts/hist_script/generate_lobster_stream_real_openbidanchor_txncomplete_fixed_start.py \
  --stock 000981_XSHE \
  --processed-real-flow-path /finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/processed_real_flow/final_result_for_merge_realflow_openbidanchor_txncomplete_20250710_000981XSHE_20260404_2207.joblib