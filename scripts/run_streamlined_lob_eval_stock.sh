#!/bin/bash

#SBATCH -J slm_streamlined_lob_eval
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --reservation=finai

if [ -z "$CKPT_PATH" ]; then
  echo "CKPT_PATH is not set"
  exit 1
fi

if [ -z "$TARGET_STOCK" ]; then
  echo "TARGET_STOCK is not set"
  exit 1
fi

if [ -z "$PROCESSED_REAL_FLOW_PATH" ]; then
  echo "PROCESSED_REAL_FLOW_PATH is not set"
  exit 1
fi

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model
python -u scripts/run_streamlined_lob_eval_pipeline.py "$CKPT_PATH" \
  --stock "$TARGET_STOCK" \
  --processed-real-flow-path "$PROCESSED_REAL_FLOW_PATH"
