#!/bin/bash
# Rerun sentence_preset_s2ip inference+replay+eval for stocks whose sampling settings changed
# under the repick-by-six rule (derived from the blank sweep).
#
# Changed stocks:
#   - 000617: from T13_k0 -> T13_k400 (temp=1.3, topk=400)
#   - 000981: from T13_k400 -> T12_k200 (temp=1.2, topk=200)
#
# We submit for both K_PREPEND in {3,5} using the same trained checkpoints.

set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model/scripts/hist_script

submit_one () {
  local stock="$1"
  local tag="$2"
  local k_prepend="$3"
  local temp="$4"
  local topk="$5"
  local sweep_tag="$6"
  STOCK="$stock" TAG="$tag" TEMP="$temp" TOPK="$topk" SWEEP_TAG="$sweep_tag" K_PREPEND="$k_prepend" \
    sbatch "$ROOT/run_inference_sentence_preset_s2ip_sweep_one.sh"
}

echo "Submitting sentence_preset_s2ip reruns (repick-by-six)..."

# 000617XSHE: T13_k400
submit_one "000617_XSHE" "000617XSHE" 3 1.3 400 "sentence_preset_s2ip_k3_repick6_T13_k400"
submit_one "000617_XSHE" "000617XSHE" 5 1.3 400 "sentence_preset_s2ip_k5_repick6_T13_k400"

# 000981XSHE: T12_k200
submit_one "000981_XSHE" "000981XSHE" 3 1.2 200 "sentence_preset_s2ip_k3_repick6_T12_k200"
submit_one "000981_XSHE" "000981XSHE" 5 1.2 200 "sentence_preset_s2ip_k5_repick6_T12_k200"

echo "Done."

