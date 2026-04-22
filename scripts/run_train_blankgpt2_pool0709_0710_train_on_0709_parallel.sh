#!/bin/bash
# Train blank GPT2 on pooled-bin 20250709 joblibs (4 stocks, 3 epochs, 1 GPU each).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$ROOT/run_train_blankgpt2_pool0709_0710_train_on_0709_000617XSHE.sh"
sbatch "$ROOT/run_train_blankgpt2_pool0709_0710_train_on_0709_000981XSHE.sh"
sbatch "$ROOT/run_train_blankgpt2_pool0709_0710_train_on_0709_002263XSHE.sh"
sbatch "$ROOT/run_train_blankgpt2_pool0709_0710_train_on_0709_002366XSHE.sh"
