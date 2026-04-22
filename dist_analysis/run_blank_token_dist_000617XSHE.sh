#!/bin/bash

#SBATCH -J tokdist_000617
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/dist_analysis/logs/run_blank_token_dist_000617XSHE.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
python -u dist_analysis/run_blank_token_distribution_analysis.py \
  --stock 000617_XSHE \
  --run-dir /finance_ML/zhanghaohan/stock_language_model/dist_analysis/blank_token_dist_run_parallel_20260404