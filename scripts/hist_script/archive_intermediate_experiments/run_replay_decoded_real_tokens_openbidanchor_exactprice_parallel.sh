#!/bin/bash
set -euo pipefail

ROOT="/finance_ML/zhanghaohan/stock_language_model/scripts/hist_script"

sbatch "$ROOT/run_replay_decoded_real_tokens_openbidanchor_exactprice_000617XSHE.sh"
sbatch "$ROOT/run_replay_decoded_real_tokens_openbidanchor_exactprice_000981XSHE.sh"
sbatch "$ROOT/run_replay_decoded_real_tokens_openbidanchor_exactprice_002263XSHE.sh"
sbatch "$ROOT/run_replay_decoded_real_tokens_openbidanchor_exactprice_002366XSHE.sh"