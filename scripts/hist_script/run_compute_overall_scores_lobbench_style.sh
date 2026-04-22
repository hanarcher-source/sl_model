#!/bin/bash
#SBATCH -J lobbench_overall
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/eval_pool0709_0710/run_compute_overall_scores_lobbench_style.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
mkdir -p $ROOT/logs/eval_pool0709_0710

# BEFORE: blank GPT-2 win50 from the sampling sweep (same trained model, different sampling settings).
BEFORE_ROOT=$ROOT/saved_LOB_stream/pool_0709_0710_eval_0710_sweep

# AFTER: pretrained GPT-2 backbone win50 (best-per-stock sampling settings), saved under model variants.
AFTER_ROOT=$ROOT/saved_LOB_stream/pool_0709_0710_eval_0710_model_variants
AFTER_SUBDIR=best_T13_k0_pretrained_win50

# Best settings per stock (from our earlier multi-metric sweep selection)
BEFORE_MAP="000617XSHE=T13_k0,000981XSHE=T10_k50,002263XSHE=T13_k0,002366XSHE=T13_k0"

python -u $ROOT/scripts/hist_script/compute_overall_scores_lobbench_style.py \
  --before-root "$BEFORE_ROOT" \
  --after-root "$AFTER_ROOT" \
  --after-subdir "$AFTER_SUBDIR" \
  --stocks "000617XSHE,002263XSHE,002366XSHE" \
  --before-setting "000617XSHE=T13_k0,002263XSHE=T13_k0,002366XSHE=T13_k0"

echo ""
echo "NOTE: 000981 uses AFTER variant best_T10_k50_pretrained_win50, so we run it separately too."

python -u $ROOT/scripts/hist_script/compute_overall_scores_lobbench_style.py \
  --before-root "$BEFORE_ROOT" \
  --after-root "$AFTER_ROOT" \
  --after-subdir "best_T10_k50_pretrained_win50" \
  --stocks "000981XSHE" \
  --before-setting "000981XSHE=T10_k50"

