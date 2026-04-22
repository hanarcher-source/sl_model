#!/bin/bash
#SBATCH -J p2d_2366
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:0
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_preprocess_twoday_pool_openbidanchor_txncomplete_002366XSHE.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
python -u scripts/hist_script/preprocess_real_lob_twoday_pool_openbidanchor_txncomplete.py \
  --stock 002366_XSHE \
  --day-a 20250709 --day-b 20250710 \
  --output-parent-subdir pool_0709_0710_openbidanchor_txncomplete
