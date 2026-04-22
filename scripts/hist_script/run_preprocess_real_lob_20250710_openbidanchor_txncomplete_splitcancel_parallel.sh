#!/bin/bash
# Submit preprocess jobs for txn-complete + split-cancel (6-way sides, vocab 26*26*12*6).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$ROOT/run_preprocess_real_lob_20250710_openbidanchor_txncomplete_splitcancel_000617XSHE.sh"
sbatch "$ROOT/run_preprocess_real_lob_20250710_openbidanchor_txncomplete_splitcancel_000981XSHE.sh"
sbatch "$ROOT/run_preprocess_real_lob_20250710_openbidanchor_txncomplete_splitcancel_002263XSHE.sh"
sbatch "$ROOT/run_preprocess_real_lob_20250710_openbidanchor_txncomplete_splitcancel_002366XSHE.sh"
