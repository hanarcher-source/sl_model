#!/usr/bin/env bash
# Submit: preprocess 0710 snapshots (apply 0709 bins) then eval 1Hz from 10:00 for 5 minutes.
#
# Run with: bash submit_bookstate_eval_1s_fixed_start_0710_start1000_T300.sh
# (Do NOT sbatch this wrapper.)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
chmod +x "$ROOT"/run_preprocess_bookstate_mdl628_anchor5m_20250710_apply0709bins.sh 2>/dev/null || true
chmod +x "$ROOT"/run_inference_eval_bookstate_parallelheads_1s_fixed_start_0710_*.sh 2>/dev/null || true

PP_JOBID="$(sbatch --parsable "$ROOT/run_preprocess_bookstate_mdl628_anchor5m_20250710_apply0709bins.sh")"
echo "[submitted] preprocess0710 jobid=$PP_JOBID"

sbatch --dependency=afterok:"$PP_JOBID" "$ROOT/run_inference_eval_bookstate_parallelheads_1s_fixed_start_0710_000617XSHE.sh"
sbatch --dependency=afterok:"$PP_JOBID" "$ROOT/run_inference_eval_bookstate_parallelheads_1s_fixed_start_0710_002263XSHE.sh"
sbatch --dependency=afterok:"$PP_JOBID" "$ROOT/run_inference_eval_bookstate_parallelheads_1s_fixed_start_0710_002721XSHE.sh"
echo "[submitted] eval jobs afterok:$PP_JOBID"

