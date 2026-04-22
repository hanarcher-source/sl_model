#!/bin/bash
#SBATCH -J tr709b_981
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_train_blankgpt2_20250709_txncomplete_000981XSHE.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
python -u scripts/train_blankgpt2_openbidanchor_txncomplete_single_day.py \
  --stock 000981_XSHE \
  --epochs 3 \
  --vocab-size 40560
