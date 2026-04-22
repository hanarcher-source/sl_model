#!/bin/bash
#SBATCH -J tr_oa_ep10_617
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_train_blankgpt2_pool0709_openanchor_ep10pat3_000617XSHE.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"
python -u scripts/train_blankgpt2_openbidanchor_txncomplete_single_day.py \
  --stock 000617_XSHE \
  --day 20250709 \
  --data-dir "$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete" \
  --output-root "$ROOT/training_runs/pool_0709_0710_train0709_blank_gpt2_win50_ep10pat3" \
  --epochs 10 \
  --patience 3 \
  --window-len 50 \
  --vocab-size 40560
