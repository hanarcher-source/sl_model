#!/usr/bin/env bash
# Gated cross-attn S2IP + Track A clustered presets, prepend K=1, router JSONL.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
chmod +x "$ROOT"/run_train_pool0709_sentence_s2ip_gca_trackA_clustered_ep3_topk1_routerdiag_*.sh 2>/dev/null || true
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_gca_trackA_clustered_ep3_topk1_routerdiag_000617XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_gca_trackA_clustered_ep3_topk1_routerdiag_002263XSHE.sh"
sbatch "$ROOT/run_train_pool0709_sentence_s2ip_gca_trackA_clustered_ep3_topk1_routerdiag_002721XSHE.sh"
