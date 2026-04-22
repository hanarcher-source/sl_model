#!/usr/bin/env bash
# Local sequential run (no Slurm). For Slurm (same headers as train jobs):
#   cd .. && sbatch scripts/run_cluster_track_a_pool0709_train_onejob.sh
#
# Run Track A clustering (train windows only) for the three pool-0709 eval stocks.
set -euo pipefail
ROOT="/finance_ML/zhanghaohan/stock_language_model"
PY="$ROOT/cluster_trackA/scripts/build_train_window_clusters_track_a.py"
OUT="$ROOT/cluster_trackA/data/cluster_runs/pool0709_train_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
LOG="$ROOT/cluster_trackA/logs"
mkdir -p "$LOG"

# Stride>1 speeds up very long days; increase if needed.
STRIDE="${STRIDE:-2}"
KMIN="${KMIN:-8}"
KMAX="${KMAX:-48}"
KSTEP="${KSTEP:-4}"

for TAG in 000617XSHE 002263XSHE 002721XSHE; do
  echo "==== $TAG ===="
  python3 "$PY" \
    --stock-tag "$TAG" \
    --day 20250709 \
    --data-dir "$ROOT/saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete" \
    --window-len 50 \
    --stride "$STRIDE" \
    --k-min "$KMIN" \
    --k-max "$KMAX" \
    --k-step "$KSTEP" \
    --minibatch \
    --out-dir "$OUT" \
    2>&1 | tee "$LOG/cluster_track_a_${TAG}.log"
done

echo "[all done] outputs under $OUT"
