#!/usr/bin/env python3
"""Print markdown table: pretrain vs Track A clustered S2IP (six LOB-bench aggregates + n_real / n_gen).

With --intersection (recommended for apples-to-apples): W_mean/median/IQM use the same metric-name set
for both variants (intersection of finite W losses); L1_* uses the intersection for L1. W_n and L1_n
then match across variants within each stock.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

ROOT = "/finance_ML/zhanghaohan/stock_language_model"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HIST = os.path.join(ROOT, "scripts/hist_script")
if _HIST not in sys.path:
    sys.path.insert(0, _HIST)

from compute_overall_scores_lobbench_style import (  # noqa: E402
    _aggregate,
    _collect_metric_losses,
    _load_metrics_summary,
)
PRE_ROOT = os.path.join(ROOT, "saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/pretrained_T10_k0_win50")
TRACK_ROOT = os.path.join(
    ROOT,
    "saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/sentence_preset_s2ip_trackA_clustered_ep3/sentence_s2ip_trackA_T10_k0_win50",
)
STOCKS = ["000617XSHE", "002263XSHE", "002721XSHE"]


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


def _six_embedded(summary: Dict[str, Any]) -> Dict[str, Any]:
    lob = summary.get("lobbench_style_overall") or {}
    if not lob:
        raise KeyError("lobbench_style_overall missing; re-run inference eval")
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


def _six_intersection_pair(pre: Dict[str, Any], trk: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    w0, l0 = _collect_metric_losses(pre)
    w1, l1 = _collect_metric_losses(trk)
    names_w = sorted(set(w0.keys()) & set(w1.keys()))
    names_l1 = sorted(set(l0.keys()) & set(l1.keys()))
    meta = {
        "common_w_metrics": names_w,
        "common_l1_metrics": names_l1,
        "w_only_pretrain": sorted(set(w0.keys()) - set(w1.keys())),
        "w_only_trackA": sorted(set(w1.keys()) - set(w0.keys())),
        "l1_only_pretrain": sorted(set(l0.keys()) - set(l1.keys())),
        "l1_only_trackA": sorted(set(l1.keys()) - set(l0.keys())),
    }
    aw0 = _aggregate(w0[k] for k in names_w)
    aw1 = _aggregate(w1[k] for k in names_w)
    al0 = _aggregate(l0[k] for k in names_l1)
    al1 = _aggregate(l1[k] for k in names_l1)

    def _pack_w(a) -> Dict[str, Any]:
        return {"W_n": a.n_metrics, "W_mean": a.mean, "W_median": a.median, "W_iqm": a.iqm}

    def _pack_l1(a) -> Dict[str, Any]:
        return {"L1_n": a.n_metrics, "L1_mean": a.mean, "L1_median": a.median, "L1_iqm": a.iqm}

    row_pre = {**_pack_w(aw0), **_pack_l1(al0)}
    row_trk = {**_pack_w(aw1), **_pack_l1(al1)}
    return row_pre, row_trk, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--intersection",
        action="store_true",
        help="Recompute six-pack on metric names present with finite W (resp. L1) in BOTH variants.",
    )
    ap.add_argument(
        "--list-dropped",
        action="store_true",
        help="With --intersection, print symmetric metric diffs to stderr.",
    )
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    if not args.intersection:
        for tag in STOCKS:
            for label, base in [("pretrain", PRE_ROOT), ("trackA_clustered", TRACK_ROOT)]:
                try:
                    path = _find_latest_metrics(base, tag)
                except FileNotFoundError as e:
                    print(f"<!-- skip {label} {tag}: {e} -->", file=sys.stderr)
                    continue
                with open(path, encoding="utf-8") as fh:
                    summary = json.load(fh)
                n_real, n_gen = _counts_from_ref(summary)
                s = _six_embedded(summary)
                rows.append({"stock": tag, "variant": label, "metrics_path": path, "n_real": n_real, "n_gen": n_gen, **s})
    else:
        for tag in STOCKS:
            try:
                pre_path = _find_latest_metrics(PRE_ROOT, tag)
                trk_path = _find_latest_metrics(TRACK_ROOT, tag)
            except FileNotFoundError as e:
                print(f"<!-- skip {tag}: {e} -->", file=sys.stderr)
                continue
            pre = _load_metrics_summary(pre_path)
            trk = _load_metrics_summary(trk_path)
            n_real_p, n_gen_p = _counts_from_ref(pre)
            n_real_t, n_gen_t = _counts_from_ref(trk)
            rp, rt, meta = _six_intersection_pair(pre, trk)
            if args.list_dropped:
                print(f"[{tag}] W intersection n={rp['W_n']}: only_pre={meta['w_only_pretrain']} only_trackA={meta['w_only_trackA']}", file=sys.stderr)
                print(f"[{tag}] L1 intersection n={rp['L1_n']}: only_pre={meta['l1_only_pretrain']} only_trackA={meta['l1_only_trackA']}", file=sys.stderr)
            rows.append(
                {
                    "stock": tag,
                    "variant": "pretrain",
                    "metrics_path": pre_path,
                    "n_real": n_real_p,
                    "n_gen": n_gen_p,
                    **rp,
                }
            )
            rows.append(
                {
                    "stock": tag,
                    "variant": "trackA_clustered",
                    "metrics_path": trk_path,
                    "n_real": n_real_t,
                    "n_gen": n_gen_t,
                    **rt,
                }
            )

    hdr = (
        "| stock | variant | n_real | n_gen | W_n | W_mean | W_median | W_iqm | "
        "L1_n | L1_mean | L1_median | L1_iqm |"
    )
    sep = "| --- | --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|"
    print(hdr)
    print(sep)
    for r in sorted(rows, key=lambda x: (x["stock"], x["variant"] != "pretrain")):
        print(
            f"| {r['stock']} | {r['variant']} | {r['n_real']} | {r['n_gen']} | "
            f"{r['W_n']} | {r['W_mean']:.6f} | {r['W_median']:.6f} | {r['W_iqm']:.6f} | "
            f"{r['L1_n']} | {r['L1_mean']:.6f} | {r['L1_median']:.6f} | {r['L1_iqm']:.6f} |"
        )


if __name__ == "__main__":
    main()
