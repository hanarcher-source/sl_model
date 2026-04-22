#!/bin/bash
# Model autoregressive replay on 20250710 vs same clean ref as direct token replay; writes comparison JSON.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$ROOT/run_inference_blankgpt2_0710_txncomplete_000617XSHE.sh"
sbatch "$ROOT/run_inference_blankgpt2_0710_txncomplete_000981XSHE.sh"
sbatch "$ROOT/run_inference_blankgpt2_0710_txncomplete_002263XSHE.sh"
sbatch "$ROOT/run_inference_blankgpt2_0710_txncomplete_002366XSHE.sh"
