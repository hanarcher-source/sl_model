#!/bin/bash
#SBATCH -J tr_g2v_l_981
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_train_pool0709_gpt2vocab_mlp_k3_win50_ep3_lowid_000981XSHE_%j.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"

python -u scripts/train_blankgpt2_sentence_preset_anchor_s2ip_txncomplete_single_day.py \
  --stock 000981_XSHE \
  --day 20250709 \
  --data-dir "$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete" \
  --output-root "$ROOT/training_runs/pool_0709_0710_train0709_gpt2vocab_mlp_k3_win50_ep3_lowid" \
  --epochs 3 \
  --patience 3 \
  --window-len 50 \
  --batch-size 256 \
  --vocab-size 40560 \
  --anchor-source gpt2_vocab \
  --anchor-vocab-select low_id \
  --anchor-map mlp \
  --topk-anchors 3 \
  --pretrained-local-only

