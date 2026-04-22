#!/bin/bash
#SBATCH -J tr_s2ip_k3_617
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_train_pool0709_sentence_s2ip_k3_ep3_000617XSHE.out
#SBATCH --reservation=finai

# Sentence preset anchors + S2IP align; train on 20250709 pool joblib; top-K=3 prepend; max 3 epochs.
source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"
python -u scripts/train_blankgpt2_sentence_preset_anchor_s2ip_txncomplete_single_day.py \
  --stock 000617_XSHE \
  --day 20250709 \
  --data-dir "$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete" \
  --output-root "$ROOT/training_runs/pool_0709_0710_train0709_sentence_preset_s2ip_win50_k3" \
  --epochs 3 \
  --patience 3 \
  --window-len 50 \
  --vocab-size 40560 \
  --topk-anchors 3 \
  --pretrained-local-only
