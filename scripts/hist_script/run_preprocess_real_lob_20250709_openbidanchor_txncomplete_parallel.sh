#!/bin/bash
# Parallel preprocess: 20250709, open-bid anchor + txn-complete (129/130), single cancel (99), n_side=5.
# Outputs: saved_LOB_stream/processed_real_flow/20250709_openbidanchor_txncomplete/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$ROOT/run_preprocess_real_lob_20250709_openbidanchor_txncomplete_000617XSHE.sh"
sbatch "$ROOT/run_preprocess_real_lob_20250709_openbidanchor_txncomplete_000981XSHE.sh"
sbatch "$ROOT/run_preprocess_real_lob_20250709_openbidanchor_txncomplete_002263XSHE.sh"
sbatch "$ROOT/run_preprocess_real_lob_20250709_openbidanchor_txncomplete_002366XSHE.sh"
