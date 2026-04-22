#!/bin/bash
# Submit 8 jobs: 4 stocks × (prepend K=3 vs K=5). Same pool/day/window/epochs otherwise.
ROOT=/finance_ML/zhanghaohan/stock_language_model/scripts

echo "Submitting sentence preset S2IP: K=3 (4 stocks)..."
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_k3_ep3_000617XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_k3_ep3_000981XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_k3_ep3_002263XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_k3_ep3_002366XSHE.sh"

echo "Submitting sentence preset S2IP: K=5 (4 stocks)..."
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_k5_ep3_000617XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_k5_ep3_000981XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_k5_ep3_002263XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_k5_ep3_002366XSHE.sh"

echo "Done (8 sbatch calls)."
