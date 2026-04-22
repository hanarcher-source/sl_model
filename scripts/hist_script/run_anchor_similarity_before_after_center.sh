#!/bin/bash
#SBATCH -J anchor_sim_center
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_anchor_similarity_before_after_center_%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

ROOT=/finance_ML/zhanghaohan/stock_language_model
cd "$ROOT"

python -u scripts/hist_script/anchor_similarity_before_after_center.py \
  --anchor-count 128 \
  --anchor-max-tokens 128 \
  --pretrained-name gpt2 \
  --local-files-only

