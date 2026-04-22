#!/bin/bash
#SBATCH -J tr_sentk3_w100_981_d1
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_train_pool0709_sentence_k3_win100_ep3_phase1diag_000981XSHE_%j.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"

python -u scripts/train_blankgpt2_sentence_preset_anchor_s2ip_txncomplete_single_day.py \
  --stock 000981_XSHE \
  --day 20250709 \
  --data-dir "$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete" \
  --output-root "$ROOT/training_runs/pool_0709_0710_train0709_sentence_k3_baseline_win100_ep3_phase1diag" \
  --epochs 3 \
  --patience 3 \
  --window-len 100 \
  --batch-size 256 \
  --vocab-size 40560 \
  --topk-anchors 3 \
  --pretrained-local-only \
  --router-diag \
  --router-diag-every-steps 200 \
  --router-diag-softmax-temp 1.0 \
  --router-diag-jsonl "$ROOT/training_runs/pool_0709_0710_train0709_sentence_k3_baseline_win100_ep3_phase1diag/000981XSHE/router_diag.jsonl" \
  --prefix-shuffle-probe \
  --prefix-shuffle-probe-batches 2

