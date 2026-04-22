#!/bin/bash
set -euo pipefail

ROOT=/finance_ML/zhanghaohan/stock_language_model/scripts

sbatch "$ROOT/run_train_pool0709_sentence_s2ip_trackA_clustered_ep3_000617XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_trackA_clustered_ep3_002263XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_trackA_clustered_ep3_002721XSHE.sh"

