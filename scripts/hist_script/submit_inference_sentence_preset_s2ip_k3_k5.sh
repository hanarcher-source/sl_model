#!/bin/bash
# Submit 8 inference+replay+eval jobs:
#   4 stocks × (sentence_preset_s2ip prepend K=3 vs K=5)
#
# Uses the per-stock best sampling settings from your lobbench aggregate:
#   000617 / 002263 / 002366: temp=1.3, topk=0 (T13_k0)
#   000981: temp=1.3, topk=400 (T13_k400)

set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model/scripts/hist_script

stocks=("000617_XSHE" "000981_XSHE" "002263_XSHE" "002366_XSHE")
tags=("000617XSHE" "000981XSHE" "002263XSHE" "002366XSHE")

params_for_stock () {
  local s="$1"
  case "$s" in
    000617_XSHE) echo "1.3 0" ;;
    000981_XSHE) echo "1.3 400" ;;
    002263_XSHE) echo "1.3 0" ;;
    002366_XSHE) echo "1.3 0" ;;
    *) echo "1.0 0" ;;
  esac
}

submit_one () {
  local stock="$1"
  local tag="$2"
  local k_prepend="$3"
  local sweep_tag="$4"
  read -r temp topk <<<"$(params_for_stock "$stock")"
  STOCK="$stock" TAG="$tag" TEMP="$temp" TOPK="$topk" SWEEP_TAG="$sweep_tag" K_PREPEND="$k_prepend" \
    sbatch "$ROOT/run_inference_sentence_preset_s2ip_sweep_one.sh"
}

SWEEP_TAG_K3="sentence_preset_s2ip_k3_ep3"
SWEEP_TAG_K5="sentence_preset_s2ip_k5_ep3"

echo "Submitting sentence_preset_s2ip K=3..."
for i in "${!stocks[@]}"; do
  submit_one "${stocks[$i]}" "${tags[$i]}" 3 "$SWEEP_TAG_K3"
done

echo "Submitting sentence_preset_s2ip K=5..."
for i in "${!stocks[@]}"; do
  submit_one "${stocks[$i]}" "${tags[$i]}" 5 "$SWEEP_TAG_K5"
done

echo "Done."

