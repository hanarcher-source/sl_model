#!/usr/bin/env bash
# Submit 3 training jobs: vocab-code anchors from cluster centroids, prepend K=1, router diagnostics.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
chmod +x "$ROOT"/run_train_pool0709_vocabcode_cluster_ep3_topk1_routerdiag_*.sh 2>/dev/null || true

sbatch "$ROOT/run_train_pool0709_vocabcode_cluster_ep3_topk1_routerdiag_000617XSHE.sh"
sbatch "$ROOT/run_train_pool0709_vocabcode_cluster_ep3_topk1_routerdiag_002263XSHE.sh"
sbatch "$ROOT/run_train_pool0709_vocabcode_cluster_ep3_topk1_routerdiag_002721XSHE.sh"

