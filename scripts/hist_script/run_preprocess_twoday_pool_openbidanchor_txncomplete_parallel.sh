#!/bin/bash
# Pooled bin fit on 20250709+20250710 per stock; outputs under pool_0709_0710_openbidanchor_txncomplete/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$ROOT/run_preprocess_twoday_pool_openbidanchor_txncomplete_000617XSHE.sh"
sbatch "$ROOT/run_preprocess_twoday_pool_openbidanchor_txncomplete_000981XSHE.sh"
sbatch "$ROOT/run_preprocess_twoday_pool_openbidanchor_txncomplete_002263XSHE.sh"
sbatch "$ROOT/run_preprocess_twoday_pool_openbidanchor_txncomplete_002366XSHE.sh"
