#!/bin/bash
# Run after preprocess + clean-reference generate for split-cancel. Picks latest matching artifacts per ticker.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$ROOT/run_replay_decoded_real_tokens_openbidanchor_txncomplete_splitcancel_000617XSHE.sh"
sbatch "$ROOT/run_replay_decoded_real_tokens_openbidanchor_txncomplete_splitcancel_000981XSHE.sh"
sbatch "$ROOT/run_replay_decoded_real_tokens_openbidanchor_txncomplete_splitcancel_002263XSHE.sh"
sbatch "$ROOT/run_replay_decoded_real_tokens_openbidanchor_txncomplete_splitcancel_002366XSHE.sh"
