#!/usr/bin/env bash
# Replay+eval (T10 k0 win50) for 6 checkpoints: Track A topk1 S2IP + gated cross-attn topk1.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
chmod +x "$D"/run_inference_sentence_s2ip_trackA_clustered_topk1_*.sh "$D"/run_inference_sentence_s2ip_gca_trackA_clustered_topk1_*.sh 2>/dev/null || true
sbatch "$D/run_inference_sentence_s2ip_trackA_clustered_topk1_000617XSHE.sh"
sbatch "$D/run_inference_sentence_s2ip_trackA_clustered_topk1_002263XSHE.sh"
sbatch "$D/run_inference_sentence_s2ip_trackA_clustered_topk1_002721XSHE.sh"
sbatch "$D/run_inference_sentence_s2ip_gca_trackA_clustered_topk1_000617XSHE.sh"
sbatch "$D/run_inference_sentence_s2ip_gca_trackA_clustered_topk1_002263XSHE.sh"
sbatch "$D/run_inference_sentence_s2ip_gca_trackA_clustered_topk1_002721XSHE.sh"
