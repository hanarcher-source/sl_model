#!/bin/bash
#SBATCH -J sweep_best_lb
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/eval_pool0709_0710/run_select_best_sweep_lobbench_aggregate.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
OUT=$ROOT/logs/eval_pool0709_0710/sweep_best_lobbench_aggregate_slurm.json
mkdir -p "$ROOT/logs/eval_pool0709_0710"

# Primary = LOB-Bench paper emphasizes reporting mean/median/IQM; default here is W_mean.
python -u "$ROOT/scripts/hist_script/select_best_sweep_setting_lobbench_aggregate.py" \
  --sweep-root "$ROOT/saved_LOB_stream/pool_0709_0710_eval_0710_sweep" \
  --criterion mean \
  --out-json "$OUT"

echo ""
echo "=== Same selection with primary criterion W_median (for comparison) ==="
python -u "$ROOT/scripts/hist_script/select_best_sweep_setting_lobbench_aggregate.py" \
  --sweep-root "$ROOT/saved_LOB_stream/pool_0709_0710_eval_0710_sweep" \
  --criterion median \
  --out-json "${OUT%.json}_median.json"
