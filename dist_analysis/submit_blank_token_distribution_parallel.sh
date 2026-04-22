#!/bin/bash

set -euo pipefail

cd /finance_ML/zhanghaohan/stock_language_model

sbatch dist_analysis/run_blank_token_dist_000617XSHE.sh
sbatch dist_analysis/run_blank_token_dist_000981XSHE.sh
sbatch dist_analysis/run_blank_token_dist_002263XSHE.sh
sbatch dist_analysis/run_blank_token_dist_002366XSHE.sh