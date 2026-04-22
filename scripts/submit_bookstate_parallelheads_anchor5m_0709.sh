#!/usr/bin/env bash
# Submit preprocess + 3 training jobs for 0709 book-state parallel-heads model.
#
# IMPORTANT:
#   Run this script with:  bash submit_bookstate_parallelheads_anchor5m_0709.sh
#   Do NOT submit this wrapper with sbatch.
#
# This wrapper submits:
#   1) preprocess job (CPU)
#   2) 3 training jobs (GPU:1 each) with dependency on preprocess success
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
chmod +x "$ROOT"/run_preprocess_bookstate_mdl628_anchor5m_20250709.sh 2>/dev/null || true
chmod +x "$ROOT"/run_train_bookstate_parallelheads_anchor5m_*.sh 2>/dev/null || true

PP_JOBID="$(sbatch --parsable "$ROOT/run_preprocess_bookstate_mdl628_anchor5m_20250709.sh")"
echo "[submitted] preprocess jobid=$PP_JOBID"

sbatch --dependency=afterok:"$PP_JOBID" "$ROOT/run_train_bookstate_parallelheads_anchor5m_000617XSHE.sh"
sbatch --dependency=afterok:"$PP_JOBID" "$ROOT/run_train_bookstate_parallelheads_anchor5m_002263XSHE.sh"
sbatch --dependency=afterok:"$PP_JOBID" "$ROOT/run_train_bookstate_parallelheads_anchor5m_002721XSHE.sh"
echo "[submitted] train jobs with dependency afterok:$PP_JOBID"

