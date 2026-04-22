#!/bin/bash
# Compare latest split-cancel token replay vs latest split-cancel clean realflow reference (same naming convention as eval_against_clean_refs).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$ROOT/run_eval_generated_stream_cleanref_txncomplete_splitcancel_000617XSHE.sh"
sbatch "$ROOT/run_eval_generated_stream_cleanref_txncomplete_splitcancel_000981XSHE.sh"
sbatch "$ROOT/run_eval_generated_stream_cleanref_txncomplete_splitcancel_002263XSHE.sh"
sbatch "$ROOT/run_eval_generated_stream_cleanref_txncomplete_splitcancel_002366XSHE.sh"
