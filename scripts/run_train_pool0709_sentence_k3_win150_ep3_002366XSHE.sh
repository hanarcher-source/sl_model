#!/bin/bash
#SBATCH -J tr_sentk3_w150_2366
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_train_pool0709_sentence_k3_win150_ep3_002366XSHE.out
#SBATCH --reservation=finai

# win150 uses smaller batch to reduce GPU memory pressure.
source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"

python -u scripts/train_blankgpt2_sentence_preset_anchor_s2ip_txncomplete_single_day.py \
  --stock 002366_XSHE \
  --day 20250709 \
  --data-dir "$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete" \
  --output-root "$ROOT/training_runs/pool_0709_0710_train0709_sentence_k3_baseline_win150_ep3" \
  --epochs 3 \
  --patience 3 \
  --window-len 150 \
  --batch-size 128 \
  --vocab-size 40560 \
  --topk-anchors 3 \
  --pretrained-local-only

