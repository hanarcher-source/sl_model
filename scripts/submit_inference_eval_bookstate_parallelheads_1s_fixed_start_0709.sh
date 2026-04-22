#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
chmod +x "$ROOT"/run_inference_eval_bookstate_parallelheads_1s_fixed_start_*.sh 2>/dev/null || true

sbatch "$ROOT/run_inference_eval_bookstate_parallelheads_1s_fixed_start_000617XSHE.sh"
sbatch "$ROOT/run_inference_eval_bookstate_parallelheads_1s_fixed_start_002263XSHE.sh"
sbatch "$ROOT/run_inference_eval_bookstate_parallelheads_1s_fixed_start_002721XSHE.sh"

