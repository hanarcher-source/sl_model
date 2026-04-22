#!/bin/bash

#SBATCH -J slm_preproc_20250710
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:0
#SBATCH --reservation=finai

if [ -z "$TARGET_STOCK" ]; then
  echo "TARGET_STOCK is not set"
  exit 1
fi

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
python -u scripts/hist_script/preprocess_real_lob_20250710.py --stock "$TARGET_STOCK" --day 20250710
