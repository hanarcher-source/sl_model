#!/bin/bash
#SBATCH -J preo709_617
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:0
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_preprocess_real_lob_20250709_openbidanchor_txncomplete_000617XSHE.out
#SBATCH --reservation=finai

# Txn-complete (129/130), single cancel (99), n_side=5. Day 20250709.
source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
python -u scripts/hist_script/preprocess_real_lob_20250710_openbidanchor_txncomplete.py \
  --stock 000617_XSHE --day 20250709 \
  --output-parent-subdir 20250709_openbidanchor_txncomplete
