#!/bin/bash
#SBATCH -J tr_s2ip_tA_721
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_train_pool0709_sentence_s2ip_trackA_clustered_ep3_002721XSHE_%j.out
#SBATCH --reservation=finai

# Sentence preset anchors (Track A clustered captions) + S2IP align.
# Same training regime as sentence_k3 baseline (ep3/pat3/win50/topk=3), only preset anchor list changes.

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"

TAG=002721XSHE
PRESET="$ROOT/cluster_trackA/data/cluster_runs/pool0709_train_54357/${TAG}/preset_anchors_clustered_k12.txt"
OUT_ROOT="$ROOT/training_runs/pool_0709_0710_train0709_sentence_preset_s2ip_win50_trackA_clustered"

python -u scripts/train_blankgpt2_sentence_preset_anchor_s2ip_txncomplete_single_day.py \
  --stock 002721_XSHE \
  --day 20250709 \
  --data-dir "$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete" \
  --output-root "$OUT_ROOT" \
  --preset-anchors "$PRESET" \
  --anchor-count 12 \
  --epochs 3 \
  --patience 3 \
  --window-len 50 \
  --vocab-size 40560 \
  --topk-anchors 3 \
  --pretrained-local-only

