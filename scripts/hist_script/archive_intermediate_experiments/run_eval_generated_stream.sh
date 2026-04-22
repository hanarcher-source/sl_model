#!/bin/bash
#SBATCH -J slm_eval_stream
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH -o /finance_ML/zhanghaohan/stock_language_model/logs/run_eval_generated_stream.out
#SBATCH --reservation=finai

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/stock_language_model

# Set both directories explicitly for evaluation.
GEN_EXP_DIR="/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/fixed_start_617_blank_generate_lobster_20260401_161834"
REAL_REF_DIR="/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/fixed_start_realflow_generate_lobster_20260402_223215"

if [ -z "$GEN_EXP_DIR" ]; then
	echo "GEN_EXP_DIR is empty. Please set a generated experiment directory."
	exit 1
fi

if [ ! -d "$GEN_EXP_DIR" ]; then
	echo "GEN_EXP_DIR does not exist: $GEN_EXP_DIR"
	exit 1
fi

if [ -z "$REAL_REF_DIR" ]; then
	echo "REAL_REF_DIR is empty. Please set a real-reference directory."
	exit 1
fi

if [ ! -d "$REAL_REF_DIR" ]; then
	echo "REAL_REF_DIR does not exist: $REAL_REF_DIR"
	exit 1
fi

if [ "$GEN_EXP_DIR" = "$REAL_REF_DIR" ]; then
	echo "GEN_EXP_DIR and REAL_REF_DIR must be different directories."
	exit 1
fi

python -u scripts/eval_generated_stream.py "$GEN_EXP_DIR" --real_ref_dir "$REAL_REF_DIR"
