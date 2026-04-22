#!/bin/bash
# Track A — train-window KMeans (3 stocks sequential, one log). Matches repo Slurm headers.
#
#   cd /finance_ML/zhanghaohan/stock_language_model
#   sbatch scripts/run_cluster_track_a_pool0709_train_onejob.sh
#
#SBATCH -J ctA_km_1job
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_cluster_track_a_pool0709_train_onejob_%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"

ARRAY_J="${SLURM_JOB_ID:-local}"
RUN_NAME="${RUN_NAME:-pool0709_train_${ARRAY_J}}"
OUT_ROOT="$ROOT/cluster_trackA/data/cluster_runs/${RUN_NAME}"
mkdir -p "$OUT_ROOT" "$ROOT/cluster_trackA/logs"

STRIDE="${STRIDE:-2}"
K_MIN="${K_MIN:-8}"
K_MAX="${K_MAX:-48}"
K_STEP="${K_STEP:-4}"

echo "### CLUSTER_TRACK_A_META job_id=${SLURM_JOB_ID:-na} run_name=${RUN_NAME} out_root=${OUT_ROOT}"
echo "### CLUSTER_TRACK_A_META stride=${STRIDE} k=${K_MIN}:${K_MAX}:${K_STEP}"
date -Is

for TAG in 000617XSHE 002263XSHE 002721XSHE; do
  echo "### STOCK_BEGIN tag=${TAG}"
  date -Is
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
  echo "### STOCK_END tag=${TAG} exit=$?"
  date -Is
done

echo "### CLUSTER_TRACK_A_DONE out_root=${OUT_ROOT}"
date -Is
