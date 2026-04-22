#!/bin/bash
# One Slurm job: all four stocks sequential in one log (run_txncomplete_splitcancel_postprocess_pipeline.out).
# For separate logs per ticker, use: run_txncomplete_splitcancel_postprocess_pipeline_parallel_stocks.sh
#
# After split-cancel preprocess. Submit preprocess separately:
#   run_preprocess_real_lob_20250710_openbidanchor_txncomplete_splitcancel_parallel.sh
#
#SBATCH -J pipe_tcsc
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:0
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_txncomplete_splitcancel_postprocess_pipeline.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
python -u scripts/hist_script/run_txncomplete_splitcancel_postprocess_pipeline.py \
  --stocks 000617_XSHE,000981_XSHE,002263_XSHE,002366_XSHE \
  --day 20250710
