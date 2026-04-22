#!/bin/bash

#SBATCH -J eval_raw_000617
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_streamlined_lob_eval_000617XSHE_rawsampling0710.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
python -u scripts/run_streamlined_lob_eval_pipeline.py \
  /finance_ML/zhanghaohan/GPT2_new_head_multi_d/model_cache/blankGPT2_multiday_continue_617_stock_win50_20250709_20260325_202035.pt \
  --stock 000617_XSHE \
  --processed-real-flow-path /finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/processed_real_flow/final_result_for_merge_realflow_20250710_000617XSHE_20260402_2219.joblib \
  --temperature 1.0 \
  --top-p 1.0 \
  --run-tag raw_sampling