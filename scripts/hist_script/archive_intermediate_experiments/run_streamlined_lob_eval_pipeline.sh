#!/bin/bash

#SBATCH -J slm_streamlined_lob_eval
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_streamlined_lob_eval_pipeline.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model

CKPT_PATH="/finance_ML/zhanghaohan/stock_language_model/model_cache/blankGPT2_multiday_continue_617_stock_win50_20260325_202035.pt"
PROCESSED_REAL_FLOW_PATH="/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/processed_real_flow/final_result_for_merge_realflow_20250710_20260402_2219.joblib"

python -u scripts/run_streamlined_lob_eval_pipeline.py "$CKPT_PATH" \
  --processed-real-flow-path "$PROCESSED_REAL_FLOW_PATH"
