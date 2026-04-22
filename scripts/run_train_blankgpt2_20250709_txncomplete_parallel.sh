#!/bin/bash
# Blank GPT2, 20250709 openbidanchor_txncomplete joblibs, 3 epochs, vocab 40560 (26*26*12*5).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$ROOT/run_train_blankgpt2_20250709_txncomplete_000617XSHE.sh"
sbatch "$ROOT/run_train_blankgpt2_20250709_txncomplete_000981XSHE.sh"
sbatch "$ROOT/run_train_blankgpt2_20250709_txncomplete_002263XSHE.sh"
sbatch "$ROOT/run_train_blankgpt2_20250709_txncomplete_002366XSHE.sh"
