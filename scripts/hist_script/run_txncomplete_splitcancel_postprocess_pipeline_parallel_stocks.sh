#!/bin/bash
# Four independent Slurm jobs (one per ticker). Each has its own stdout log under logs/.
# Use this instead of run_txncomplete_splitcancel_postprocess_pipeline_slurm.sh for parallel runs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$ROOT/run_txncomplete_splitcancel_postprocess_pipeline_000617XSHE.sh"
sbatch "$ROOT/run_txncomplete_splitcancel_postprocess_pipeline_000981XSHE.sh"
sbatch "$ROOT/run_txncomplete_splitcancel_postprocess_pipeline_002263XSHE.sh"
sbatch "$ROOT/run_txncomplete_splitcancel_postprocess_pipeline_002366XSHE.sh"
