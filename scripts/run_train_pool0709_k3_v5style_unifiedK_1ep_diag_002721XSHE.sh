#!/bin/bash
#SBATCH -J tr_k3v5uni_721
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_train_pool0709_k3_v5style_unifiedK_1ep_diag_002721XSHE.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"
STOCK_TAG=002721XSHE
OUT_ROOT="$ROOT/training_runs/pool_0709_0710_train0709_sentence_preset_s2ip_win50_k3_v5style_unifiedK_1ep_diag"
JSONL="$OUT_ROOT/$STOCK_TAG/router_diag.jsonl"
mkdir -p "$(dirname "$JSONL")"

python -u scripts/train_blankgpt2_sentence_preset_anchor_s2ip_txncomplete_single_day_variant_margin_sep_k3_regk5_v5match.py \
  --stock 002721_XSHE \
  --day 20250709 \
  --data-dir "$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete" \
  --output-root "$OUT_ROOT" \
  --epochs 1 \
  --patience 1 \
  --window-len 50 \
  --vocab-size 40560 \
  --topk-anchors 3 \
  --align-warmup-steps 20 \
  --sep-mode k_vs_k1 \
  --sep-lambda 1e-2 \
  --sep-margin 0.2 \
  --sep-warmup-steps 20 \
  --router-diag \
  --router-diag-every-steps 100 \
  --router-diag-softmax-temp 1.0 \
  --router-diag-jsonl "$JSONL" \
  --pretrained-local-only
