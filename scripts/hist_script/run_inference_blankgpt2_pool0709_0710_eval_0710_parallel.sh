#!/bin/bash
# Submit 4 model-inference replays on pooled-bin 0710 preprocess.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$ROOT/run_inference_blankgpt2_pool0709_0710_eval_0710_000617XSHE.sh"
sbatch "$ROOT/run_inference_blankgpt2_pool0709_0710_eval_0710_000981XSHE.sh"
sbatch "$ROOT/run_inference_blankgpt2_pool0709_0710_eval_0710_002263XSHE.sh"
sbatch "$ROOT/run_inference_blankgpt2_pool0709_0710_eval_0710_002366XSHE.sh"

