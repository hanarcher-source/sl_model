#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$ROOT/run_generate_lobster_stream_real_openbidanchor_txncomplete_splitcancel_000617XSHE.sh"
sbatch "$ROOT/run_generate_lobster_stream_real_openbidanchor_txncomplete_splitcancel_000981XSHE.sh"
sbatch "$ROOT/run_generate_lobster_stream_real_openbidanchor_txncomplete_splitcancel_002263XSHE.sh"
sbatch "$ROOT/run_generate_lobster_stream_real_openbidanchor_txncomplete_splitcancel_002366XSHE.sh"
