#!/usr/bin/env python3
"""
For each stock, scan all temp/top-k sweep runs under a root folder and pick the setting
that minimizes LOB-Bench-style aggregate Wasserstein loss (mean/median/IQM over
reference_comparison metrics in metrics_summary.json).

Compares the winner to the prior multi-metric z-score picks used for downstream jobs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_compute_module():
    here = Path(__file__).resolve().parent
    path = here / "compute_overall_scores_lobbench_style.py"
    name = "lobbench_compute_internal"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_metrics_for_stock(setting_root: str, stock_tag: str) -> Optional[str]:
    """Latest metrics_summary.json under setting_root whose path contains stock_tag."""
    hits: List[Tuple[float, str]] = []
    for dirpath, _, filenames in os.walk(setting_root):
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
        return None
    hits.sort(reverse=True)
    return hits[0][1]


def _sort_key(
    criterion: str,
    w_mean: Optional[float],
    w_median: Optional[float],
    w_iqm: Optional[float],
    setting: str,
) -> Tuple[float, float, float, str]:
    if criterion == "mean":
        prim = w_mean if w_mean is not None else float("inf")
        sec = w_median if w_median is not None else float("inf")
        ter = w_iqm if w_iqm is not None else float("inf")
    elif criterion == "median":
        prim = w_median if w_median is not None else float("inf")
        sec = w_mean if w_mean is not None else float("inf")
        ter = w_iqm if w_iqm is not None else float("inf")
    elif criterion == "iqm":
        prim = w_iqm if w_iqm is not None else float("inf")
        sec = w_mean if w_mean is not None else float("inf")
        ter = w_median if w_median is not None else float("inf")
    else:
        raise ValueError(f"Unknown criterion {criterion}")
    return (prim, sec, ter, setting)


def main() -> None:
    mod = _load_compute_module()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sweep-root",
        default="/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/pool_0709_0710_eval_0710_sweep",
        help="Root containing one subfolder per sweep setting (e.g. T13_k0).",
    )
    ap.add_argument(
        "--stocks",
        default="000617XSHE,000981XSHE,002263XSHE,002366XSHE",
        help="Comma-separated stock path tags.",
    )
    ap.add_argument(
        "--criterion",
        choices=("mean", "median", "iqm"),
        default="mean",
        help="Primary objective: minimize Wasserstein aggregate mean, median, or IQM.",
    )
    ap.add_argument(
        "--prior-zscore-best",
        default="000617XSHE=T13_k0,000981XSHE=T10_k50,002263XSHE=T13_k0,002366XSHE=T13_k0",
        help="Comma-separated stock=setting from the earlier z-score winner table (for diff only).",
    )
    ap.add_argument(
        "--out-json",
        default="",
        help="If set, write full per-(stock,setting) table + winners to this path.",
    )
    args = ap.parse_args()

    stocks = [s.strip() for s in args.stocks.split(",") if s.strip()]
    prior: Dict[str, str] = {}
    for item in args.prior_zscore_best.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        prior[k.strip()] = v.strip()

    if not os.path.isdir(args.sweep_root):
        raise SystemExit(f"Missing sweep root: {args.sweep_root}")

    settings = sorted(
        d for d in os.listdir(args.sweep_root) if os.path.isdir(os.path.join(args.sweep_root, d))
    )

    all_rows: List[Dict[str, Any]] = []
    winners: List[Dict[str, Any]] = []

    for st in stocks:
        per_setting: List[Dict[str, Any]] = []
        for setting in settings:
            setting_root = os.path.join(args.sweep_root, setting)
            mp = _find_metrics_for_stock(setting_root, st)
            if not mp:
                continue
            summary = mod._load_metrics_summary(mp)
            w, l1d = mod._collect_metric_losses(summary)
            w_agg = mod._aggregate(w.values())
            n_w = w_agg.n_metrics
            w_mean, w_med, w_iqm = w_agg.mean, w_agg.median, w_agg.iqm
            n_l1 = len(l1d)
            row = {
                "stock": st,
                "setting": setting,
                "metrics_path": mp,
                "n_w": n_w,
                "n_l1": n_l1,
                "W_mean": w_mean,
                "W_median": w_med,
                "W_iqm": w_iqm,
            }
            per_setting.append(row)
            all_rows.append(row)

        if not per_setting:
            winners.append({"stock": st, "error": "no_metrics_found"})
            continue

        per_setting.sort(
            key=lambda r: _sort_key(
                args.criterion,
                r["W_mean"],
                r["W_median"],
                r["W_iqm"],
                r["setting"],
            )
        )
        best = per_setting[0]
        prev = prior.get(st)
        same = prev == best["setting"] if prev else None
        winners.append(
            {
                "stock": st,
                "best_setting": best["setting"],
                "prior_zscore_setting": prev,
                "same_as_prior": same,
                "criterion": args.criterion,
                "W_mean": best["W_mean"],
                "W_median": best["W_median"],
                "W_iqm": best["W_iqm"],
                "n_w": best["n_w"],
            }
        )

    print("LOB-Bench-style sweep selection (lower W aggregate is better)")
    print(f"Primary criterion: W_{args.criterion}  (tie-break: other W aggregates, then setting name)")
    print(f"Sweep root: {args.sweep_root}")
    print(f"Settings scanned: {len(settings)}")
    print("")
    hdr = [
        "stock",
        "best_setting",
        f"W_{args.criterion}_best",
        "W_mean",
        "W_median",
        "W_iqm",
        "n_w",
        "prior_zscore",
        "same?",
    ]
    print("\t".join(hdr))
    for w in winners:
        if "error" in w:
            print(f"{w['stock']}\tERROR\t\t\t\t\t\t")
            continue
        prim = w["W_mean"] if args.criterion == "mean" else w["W_median"] if args.criterion == "median" else w["W_iqm"]
        prim_s = f"{float(prim):.6g}" if prim is not None else "NA"
        print(
            "\t".join(
                [
                    w["stock"],
                    w["best_setting"],
                    prim_s,
                    f"{float(w['W_mean']):.6g}" if w["W_mean"] is not None else "NA",
                    f"{float(w['W_median']):.6g}" if w["W_median"] is not None else "NA",
                    f"{float(w['W_iqm']):.6g}" if w["W_iqm"] is not None else "NA",
                    str(w["n_w"]),
                    str(w.get("prior_zscore_setting") or ""),
                    "YES" if w.get("same_as_prior") else "NO" if w.get("same_as_prior") is False else "?",
                ]
            )
        )

    if args.out_json:
        payload = {"winners": winners, "rows": all_rows, "settings": settings, "criterion": args.criterion}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
