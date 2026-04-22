#!/bin/bash
# Track A — train-window KMeans; Slurm array 0–2 (one stock per task). Same headers as train jobs.
#
#   cd /finance_ML/zhanghaohan/stock_language_model
#   sbatch scripts/run_cluster_track_a_pool0709_train_array.sh
#
#SBATCH -J ctA_km_arr
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH --array=0-2
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_cluster_track_a_pool0709_train_array_%A_%a.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"

ARRAY_J="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
RUN_NAME="${RUN_NAME:-pool0709_train_${ARRAY_J}}"
OUT_ROOT="$ROOT/cluster_trackA/data/cluster_runs/${RUN_NAME}"
mkdir -p "$OUT_ROOT"

STOCK_TAGS=(000617XSHE 002263XSHE 002721XSHE)
TAG="${STOCK_TAGS[$SLURM_ARRAY_TASK_ID]}"

echo "### CLUSTER_TRACK_A_ARRAY_META array_job=${ARRAY_J} task_id=${SLURM_ARRAY_TASK_ID} task_job=${SLURM_JOB_ID} stock=${TAG} out_root=${OUT_ROOT}"
date -Is

STRIDE="${STRIDE:-2}"
K_MIN="${K_MIN:-8}"
K_MAX="${K_MAX:-48}"
K_STEP="${K_STEP:-4}"

python -u "$ROOT/cluster_trackA/scripts/build_train_window_clusters_track_a.py" \
  --stock-tag "$TAG" \
  --day 20250709 \
  --data-dir "$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete" \
  --window-len 50 \
  --stride "$STRIDE" \
  --k-min "$K_MIN" \
  --k-max "$K_MAX" \
  --k-step "$K_STEP" \
  --minibatch \
  --out-dir "$OUT_ROOT"

echo "### CLUSTER_TRACK_A_ARRAY_DONE stock=${TAG} out=${OUT_ROOT}/${TAG}"
date -Is
