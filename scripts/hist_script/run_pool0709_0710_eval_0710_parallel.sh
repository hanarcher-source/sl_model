#!/bin/bash
# Submit all 8 jobs: 4 direct replays + 4 model inference replays.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$ROOT/run_replay_decoded_real_tokens_openbidanchor_txncomplete_pool0709_0710_eval_0710_parallel.sh"
bash "$ROOT/run_inference_blankgpt2_pool0709_0710_eval_0710_parallel.sh"

