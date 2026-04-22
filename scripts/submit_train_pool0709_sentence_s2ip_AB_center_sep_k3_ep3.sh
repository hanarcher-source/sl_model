#!/bin/bash
# Submit 8 jobs: 4 stocks × 2 variants, all with K_PREPEND=3, epochs=3.
#
# Variant A: frozen + centered + separate projections (q_proj/a_proj)
# Variant B: trainable + centered + separate projections (q_proj/a_proj)

set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model/scripts

echo "Submitting sentence preset S2IP Variant A (frozen, centered, separate-proj)..."
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_A_k3_ep3_000617XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_A_k3_ep3_000981XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_A_k3_ep3_002263XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_A_k3_ep3_002366XSHE.sh"

echo "Submitting sentence preset S2IP Variant B (trainable, centered, separate-proj)..."
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_B_k3_ep3_000617XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_B_k3_ep3_000981XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_B_k3_ep3_002263XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_B_k3_ep3_002366XSHE.sh"

echo "Done (8 sbatch calls)."

