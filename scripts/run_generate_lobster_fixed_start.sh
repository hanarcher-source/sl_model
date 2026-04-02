#!/bin/bash

#SBATCH -J slm_lobster_gen_fixed
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_generate_lobster_fixed_start.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

python -u /finance_ML/zhanghaohan/stock_language_model/scripts/generate_lobster_stream_fixed_start.py
