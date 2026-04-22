#!/bin/bash
#SBATCH -J reeval_lb_win
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/eval_pool0709_0710/run_reeval_lobbench_then_analyze_window.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
LOGDIR=$ROOT/logs/eval_pool0709_0710
mkdir -p "$LOGDIR"

# 8× lbmean win100/win200 (no lobbench_style_overall) + 4× sweep win50 (same sampling) → 12 re-evals
python -u "$ROOT/scripts/hist_script/reeval_experiments_for_lobbench_overall.py" \
  --root "$ROOT" \
  --include-sweep-win50

python -u "$ROOT/scripts/hist_script/analyze_blank_window_lobbench_overall.py" \
  --root "$ROOT" \
  --out-csv "$LOGDIR/blank_window_lobbench_w_aggregate.csv"

echo "Finished re-eval + window analysis."
