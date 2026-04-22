#!/usr/bin/env python3
"""
Compare vs pretrain on LOB-bench six aggregates for:
  - TrackA clustered preset S2IP with prepend K=1
  - TrackA clustered preset S2IP + gated cross-attn with prepend K=1

By default uses intersection pooling (finite metric names common to both runs) so W_n/L1_n match.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

ROOT = "/finance_ML/zhanghaohan/stock_language_model"

PRE_ROOT = os.path.join(ROOT, "saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/pretrained_T10_k0_win50")
TOPK1_ROOT = os.path.join(
    ROOT,
    "saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/"
    "sentence_preset_s2ip_trackA_clustered_topk1_routerdiag/sentence_s2ip_trackA_T10_k0_win50",
)
GCA_ROOT = os.path.join(
    ROOT,
    "saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/"
    "sentence_preset_s2ip_gca_trackA_clustered_topk1_routerdiag/sentence_s2ip_gca_T10_k0_win50",
)

STOCKS = ["000617XSHE", "002263XSHE", "002721XSHE"]

_HIST = os.path.join(ROOT, "scripts/hist_script")
if _HIST not in sys.path:
    sys.path.insert(0, _HIST)

from compute_overall_scores_lobbench_style import (  # noqa: E402
    _aggregate,
    _collect_metric_losses,
    _load_metrics_summary,
)


def _find_latest_metrics(root: str, stock_tag: str) -> str:
    hits: List[Tuple[float, str]] = []
    for dirpath, _, filenames in os.walk(root):
        if "metrics_summary.json" not in filenames:
            continue
        if stock_tag not in dirpath:
            continue
        p = os.path.join(dirpath, "metrics_summary.json")
        try:
            hits.append((os.path.getmtime(p), p))
        except OSError:
            pass
    if not hits:
        raise FileNotFoundError(f"No metrics_summary.json for {stock_tag} under {root}")
    hits.sort(reverse=True)
    return hits[0][1]


def _counts_from_ref(summary: Dict[str, Any]) -> Tuple[int, int]:
    ref = summary.get("reference_comparison") or {}
    metrics = ref.get("metrics") or {}
    n_gen = 0
    n_real = 0
    for v in metrics.values():
        if not isinstance(v, dict):
            continue
        g = v.get("n_generated")
        r = v.get("n_real")
        if isinstance(g, (int, float)):
            n_gen = max(n_gen, int(g))
        if isinstance(r, (int, float)):
            n_real = max(n_real, int(r))
    return n_real, n_gen


def _six_intersection(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    w0, l0 = _collect_metric_losses(a)
    w1, l1 = _collect_metric_losses(b)
    names_w = sorted(set(w0.keys()) & set(w1.keys()))
    names_l1 = sorted(set(l0.keys()) & set(l1.keys()))
    meta = {
        "common_w_n": len(names_w),
        "common_l1_n": len(names_l1),
        "w_only_a": sorted(set(w0.keys()) - set(w1.keys())),
        "w_only_b": sorted(set(w1.keys()) - set(w0.keys())),
        "l1_only_a": sorted(set(l0.keys()) - set(l1.keys())),
        "l1_only_b": sorted(set(l1.keys()) - set(l0.keys())),
    }
    aw0 = _aggregate(w0[k] for k in names_w)
    aw1 = _aggregate(w1[k] for k in names_w)
    al0 = _aggregate(l0[k] for k in names_l1)
    al1 = _aggregate(l1[k] for k in names_l1)

    def pack(w, l1) -> Dict[str, Any]:
        return {
            "W_n": w.n_metrics,
            "W_mean": w.mean,
            "W_median": w.median,
            "W_iqm": w.iqm,
            "L1_n": l1.n_metrics,
            "L1_mean": l1.mean,
            "L1_median": l1.median,
            "L1_iqm": l1.iqm,
        }

    return pack(aw0, al0), pack(aw1, al1), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-intersection", action="store_true", help="Use embedded lobbench_style_overall directly.")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for tag in STOCKS:
        pre_p = _find_latest_metrics(PRE_ROOT, tag)
        t1_p = _find_latest_metrics(TOPK1_ROOT, tag)
        gca_p = _find_latest_metrics(GCA_ROOT, tag)
        pre = _load_metrics_summary(pre_p)
        t1 = _load_metrics_summary(t1_p)
        gca = _load_metrics_summary(gca_p)

        if args.no_intersection:
            def six_emb(ms: Dict[str, Any]) -> Dict[str, Any]:
                lob = ms.get("lobbench_style_overall") or {}
                w = lob.get("wasserstein") or {}
                l1 = lob.get("l1_by_group") or {}
                return {
                    "W_n": w.get("n"),
                    "W_mean": w.get("mean"),
                    "W_median": w.get("median"),
                    "W_iqm": w.get("iqm"),
                    "L1_n": l1.get("n"),
                    "L1_mean": l1.get("mean"),
                    "L1_median": l1.get("median"),
                    "L1_iqm": l1.get("iqm"),
                }
            s_pre = six_emb(pre)
            s_t1 = six_emb(t1)
            s_gca = six_emb(gca)
        else:
            s_pre, s_t1, _ = _six_intersection(pre, t1)
            _, s_gca, _ = _six_intersection(pre, gca)

        n_real, n_gen = _counts_from_ref(pre)
        rows.append({"stock": tag, "variant": "pretrain", "n_real": n_real, "n_gen": n_gen, **s_pre})
        n_real, n_gen = _counts_from_ref(t1)
        rows.append({"stock": tag, "variant": "trackA_topk1", "n_real": n_real, "n_gen": n_gen, **s_t1})
        n_real, n_gen = _counts_from_ref(gca)
        rows.append({"stock": tag, "variant": "gca_trackA_topk1", "n_real": n_real, "n_gen": n_gen, **s_gca})

    print("| stock | variant | n_real | n_gen | W_n | W_mean | W_median | W_iqm | L1_n | L1_mean | L1_median | L1_iqm |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    order = {"pretrain": 0, "trackA_topk1": 1, "gca_trackA_topk1": 2}
    for r in sorted(rows, key=lambda x: (x["stock"], order.get(x["variant"], 99))):
        print(
            f"| {r['stock']} | {r['variant']} | {r['n_real']} | {r['n_gen']} | "
            f"{r['W_n']} | {r['W_mean']:.6f} | {r['W_median']:.6f} | {r['W_iqm']:.6f} | "
            f"{r['L1_n']} | {r['L1_mean']:.6f} | {r['L1_median']:.6f} | {r['L1_iqm']:.6f} |"
        )


if __name__ == "__main__":
    main()

