#!/usr/bin/env bash
# Replay+eval (T10 k0 win50) for vocabcode-from-clusters topk1 checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
chmod +x "$ROOT"/run_inference_vocabcode_cluster_topk1_*.sh 2>/dev/null || true

sbatch "$ROOT/run_inference_vocabcode_cluster_topk1_000617XSHE.sh"
sbatch "$ROOT/run_inference_vocabcode_cluster_topk1_002263XSHE.sh"
sbatch "$ROOT/run_inference_vocabcode_cluster_topk1_002721XSHE.sh"

