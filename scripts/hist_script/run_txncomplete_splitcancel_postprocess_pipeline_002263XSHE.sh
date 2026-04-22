#!/bin/bash
#SBATCH -J pipe_tcsc_2263
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:0
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_txncomplete_splitcancel_postprocess_pipeline_002263XSHE.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
PIPELINE_TS=$(date +%Y%m%d_%H%M%S)
python -u scripts/hist_script/run_txncomplete_splitcancel_postprocess_pipeline.py \
  --stock 002263_XSHE \
  --day 20250710 \
  --pipeline-out-dir "/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/pipeline_txncomplete_splitcancel_002263XSHE_${PIPELINE_TS}"
