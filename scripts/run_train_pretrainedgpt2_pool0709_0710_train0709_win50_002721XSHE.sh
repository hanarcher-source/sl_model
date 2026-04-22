#!/bin/bash
#SBATCH -J trpt_2721
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_train_pretrainedgpt2_pool0709_0710_train0709_win50_002721XSHE.out
#SBATCH --reservation=finai

# Pooled-bin preprocess; train on 20250709 joblib only, pretrained GPT2 backbone, window_len=50.
source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"
python -u scripts/train_blankgpt2_openbidanchor_txncomplete_single_day.py \
  --stock 002721_XSHE \
  --day 20250709 \
  --data-dir "$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete" \
  --output-root "$ROOT/training_runs/pool_0709_0710_train0709_pretrained_gpt2_win50" \
  --epochs 3 \
  --vocab-size 40560 \
  --window-len 50 \
  --init-pretrained-backbone \
  --pretrained-backbone-name gpt2 \
  --pretrained-local-only

