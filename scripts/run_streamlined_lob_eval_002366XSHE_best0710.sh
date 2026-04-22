#!/bin/bash

#SBATCH -J eval_002366
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_streamlined_lob_eval_002366XSHE_best0710.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
python -u scripts/run_streamlined_lob_eval_pipeline.py \
  /finance_ML/zhanghaohan/GPT2_new_head_multi_d/model_cache/blankGPT2_multiday_continue_2366_stock_win50_20250709_20260325_202032.pt \
  --stock 002366_XSHE \
  --processed-real-flow-path /finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/processed_real_flow/final_result_for_merge_realflow_20250710_002366XSHE_20260403_2232.joblib
