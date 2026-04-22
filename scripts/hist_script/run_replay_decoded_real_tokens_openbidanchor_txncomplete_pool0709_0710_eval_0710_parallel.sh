#!/bin/bash
# Submit 4 direct (real-token) replays on pooled-bin 0710 preprocess.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$ROOT/run_replay_decoded_real_tokens_openbidanchor_txncomplete_pool0709_0710_eval_0710_000617XSHE.sh"
sbatch "$ROOT/run_replay_decoded_real_tokens_openbidanchor_txncomplete_pool0709_0710_eval_0710_000981XSHE.sh"
sbatch "$ROOT/run_replay_decoded_real_tokens_openbidanchor_txncomplete_pool0709_0710_eval_0710_002263XSHE.sh"
sbatch "$ROOT/run_replay_decoded_real_tokens_openbidanchor_txncomplete_pool0709_0710_eval_0710_002366XSHE.sh"

