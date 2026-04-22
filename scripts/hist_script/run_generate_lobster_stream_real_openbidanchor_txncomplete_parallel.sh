#!/bin/bash
set -euo pipefail

ROOT="/finance_ML/zhanghaohan/stock_language_model/scripts/hist_script"

sbatch "$ROOT/run_generate_lobster_stream_real_openbidanchor_txncomplete_000617XSHE.sh"
sbatch "$ROOT/run_generate_lobster_stream_real_openbidanchor_txncomplete_000981XSHE.sh"
sbatch "$ROOT/run_generate_lobster_stream_real_openbidanchor_txncomplete_002263XSHE.sh"
sbatch "$ROOT/run_generate_lobster_stream_real_openbidanchor_txncomplete_002366XSHE.sh"