#!/usr/bin/env bash
# Re-run 0710 1Hz eval (10:00 start, 5min horizon) after updating evaluator to emit lobbench6 aggregates.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
chmod +x "$ROOT"/run_inference_eval_bookstate_parallelheads_1s_fixed_start_0710_*.sh 2>/dev/null || true

sbatch "$ROOT/run_inference_eval_bookstate_parallelheads_1s_fixed_start_0710_000617XSHE.sh"
sbatch "$ROOT/run_inference_eval_bookstate_parallelheads_1s_fixed_start_0710_002263XSHE.sh"
sbatch "$ROOT/run_inference_eval_bookstate_parallelheads_1s_fixed_start_0710_002721XSHE.sh"

