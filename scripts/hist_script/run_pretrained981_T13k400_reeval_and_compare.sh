#!/bin/bash
#SBATCH -J pre981_t13k4
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/eval_pool0709_0710/run_pretrained981_T13k400_reeval_compare.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
LOGDIR=$ROOT/logs/eval_pool0709_0710
mkdir -p "$LOGDIR"

JOB=$ROOT/scripts/hist_script/run_inference_eval_pool0709_0710_from_training_one.sh

# ---------------------------------------------------------------------------
# No retraining: same pretrained win50 checkpoint; only sampling matches
# LOB-Bench-mean optimum for 981 (T13_k400 @ temp=1.3, top_k=400).
# Old stream: best_T10_k50_pretrained_win50 (z-score pick) — superseded for 981.
# ---------------------------------------------------------------------------
echo "=== Inference + eval: 000981_XSHE pretrained win50, T=1.3, top_k=400 ==="
STOCK="000981_XSHE" TAG="000981XSHE" TEMP="1.3" TOPK="400" \
  SWEEP_TAG="lbmean_T13_k400_pretrained_win50" WINDOW_LEN="50" \
  TRAIN_ROOT="$ROOT/training_runs/pool_0709_0710_train0709_pretrained_gpt2_win50" \
  bash "$JOB"

echo ""
echo "=== Re-eval pretrained win50 dirs (add lobbench_style_overall to JSON if missing) ==="
python -u "$ROOT/scripts/hist_script/reeval_experiments_for_lobbench_overall.py" \
  --root "$ROOT" \
  --only-pretrained-win50-lbmean

echo ""
echo "=== Blank vs pretrained table (LOB-Bench W_mean) ==="
python -u "$ROOT/scripts/hist_script/analyze_blank_vs_pretrained_lobbench_overall.py" \
  --root "$ROOT" \
  --out-csv "$LOGDIR/blank_vs_pretrained_lobbench_w_mean_win50.csv"

echo "Done."
