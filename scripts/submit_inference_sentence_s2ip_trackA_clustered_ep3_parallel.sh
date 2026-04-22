#!/usr/bin/env bash
# Submit replay+eval for Track A clustered S2IP checkpoints (T=1.0, top_k=0, win50).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
sbatch "$ROOT/scripts/run_inference_sentence_s2ip_trackA_clustered_ep3_000617XSHE.sh"
sbatch "$ROOT/scripts/run_inference_sentence_s2ip_trackA_clustered_ep3_002263XSHE.sh"
sbatch "$ROOT/scripts/run_inference_sentence_s2ip_trackA_clustered_ep3_002721XSHE.sh"
