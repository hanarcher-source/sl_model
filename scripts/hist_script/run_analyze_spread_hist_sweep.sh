#!/bin/bash
#SBATCH -J spread_sweep
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/eval_pool0709_0710/run_analyze_spread_hist_sweep.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
mkdir -p $ROOT/logs/eval_pool0709_0710

python -u $ROOT/scripts/hist_script/analyze_spread_hist_sweep.py \
  --sweep-root $ROOT/saved_LOB_stream/pool_0709_0710_eval_0710_sweep \
  --settings T07_k0,T13_k0,T10_k50,T10_k200 \
  --stocks 000617_XSHE,000981_XSHE,002263_XSHE,002366_XSHE

