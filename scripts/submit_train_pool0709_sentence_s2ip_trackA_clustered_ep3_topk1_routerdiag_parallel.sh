#!/usr/bin/env bash
# Track A clustered anchors, prepend K=1 only, router diagnostics JSONL per stock.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
chmod +x "$ROOT"/run_train_pool0709_sentence_s2ip_trackA_clustered_ep3_topk1_routerdiag_*.sh 2>/dev/null || true
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_trackA_clustered_ep3_topk1_routerdiag_000617XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_trackA_clustered_ep3_topk1_routerdiag_002263XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_trackA_clustered_ep3_topk1_routerdiag_002721XSHE.sh"
