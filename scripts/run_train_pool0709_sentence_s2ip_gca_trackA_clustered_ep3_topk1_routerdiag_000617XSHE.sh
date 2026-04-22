#!/bin/bash
#SBATCH -J tr_s2ip_gca1_617
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_train_pool0709_sentence_s2ip_gca_trackA_clustered_ep3_topk1_routerdiag_000617XSHE_%j.out
#SBATCH --reservation=finai

# Gated cross-attn + sentence S2IP. Track A clustered presets; prepend K=1; router diagnostics JSONL.

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"

TAG=000617XSHE
PRESET="$ROOT/cluster_trackA/data/cluster_runs/pool0709_train_54357/${TAG}/preset_anchors_clustered_k8.txt"
OUT_ROOT="$ROOT/training_runs/pool_0709_0710_train0709_sentence_preset_s2ip_gated_crossattn_trackA_clustered_topk1_routerdiag"
mkdir -p "$OUT_ROOT/$TAG"

python -u scripts/train_blankgpt2_sentence_preset_anchor_s2ip_gated_crossattn_txncomplete_single_day.py \
  --stock 000617_XSHE \
  --day 20250709 \
  --data-dir "$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete" \
  --output-root "$OUT_ROOT" \
  --preset-anchors "$PRESET" \
  --anchor-count 8 \
  --epochs 3 \
  --patience 3 \
  --window-len 50 \
  --vocab-size 40560 \
  --topk-anchors 1 \
  --cross-attn-heads 12 \
  --gate-init-bias -5.0 \
  --pretrained-local-only \
  --router-diag \
  --router-diag-every-steps 100 \
  --router-diag-jsonl "$OUT_ROOT/$TAG/router_diag.jsonl"
