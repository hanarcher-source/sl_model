#!/bin/bash

#SBATCH -J reoeq_002366
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:0
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_replay_decoded_real_tokens_openbidanchor_exactprice_exactqty_002366XSHE.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
python -u scripts/hist_script/replay_decoded_real_tokens_openbidanchor_exactprice_fixed_start.py \
  --stock 002366_XSHE \
  --processed-real-flow-path /finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/processed_real_flow/final_result_for_merge_realflow_openbidanchor_20250710_002366XSHE_20260404_2101.joblib \
  --bin-record-path /finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/processed_real_flow/bin_record_realflow_openbidanchor_20250710_002366XSHE_20260404_2101.json \
  --real-ref-dir /finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/fixed_start_realflow_generate_lobster_002366XSHE_20260403_232134 \
  --use-exact-qty