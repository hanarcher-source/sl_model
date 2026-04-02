#!/bin/bash
#SBATCH -J slm_eval_stream
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_eval_generated_stream.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model

python -u scripts/eval_generated_stream.py
