#!/usr/bin/env python3
"""
Compare blank GPT-2 vs sentence-preset S2IP (K=3) under fixed sampling (T=1.0, top_k=0)
for window_len=100. Prints LOB-Bench-style six scalars:
  - Wasserstein: mean/median/IQM
  - L1_by_group: mean/median/IQM

This script discovers the newest metrics_summary.json for each stock under each root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from compute_overall_scores_lobbench_style import (  # noqa: E402
    _aggregate,
    _collect_metric_losses,
    _load_metrics_summary,
)


def _find_latest_metrics_summary(root: str, stock_tag: str) -> str:
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


def six_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    w_map, l1_map = _collect_metric_losses(summary)
    w_agg = _aggregate(w_map.values())
    l1_agg = _aggregate(l1_map.values())
    return {
        "wasserstein": {"n": w_agg.n_metrics, "mean": w_agg.mean, "median": w_agg.median, "iqm": w_agg.iqm},
        "l1_by_group": {"n": l1_agg.n_metrics, "mean": l1_agg.mean, "median": l1_agg.median, "iqm": l1_agg.iqm},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="", help="Optional: write full rows JSON here.")
    ap.add_argument("--out-tsv", default="", help="Optional: write TSV here.")
    args = ap.parse_args()

    root = "/finance_ML/zhanghaohan/stock_language_model"
    blank_root = os.path.join(
        root, "saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/blank_fixed_sampling_win100/blank_T10_k0_win100"
    )
    sentence_root = os.path.join(
        root, "saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/sentence_k3_win100/sentence_k3_win100_T10_k0_ep3"
    )

    stocks = ["000617XSHE", "000981XSHE", "002263XSHE", "002366XSHE"]
    rows: List[Dict[str, Any]] = []

    for tag in stocks:
        for variant, base in [("blank_win100", blank_root), ("sentence_k3_win100", sentence_root)]:
            path = _find_latest_metrics_summary(base, tag)
            summary = _load_metrics_summary(path)
            six = six_from_summary(summary)
            rows.append(
                {
                    "stock": tag,
                    "variant": variant,
                    "metrics_path": path,
                    "derived_six": six,
                }
            )

    def fmt(x: Any) -> str:
        return "" if x is None else f"{float(x):.10g}"

    lines = [
        "\t".join(
            [
                "stock",
                "variant",
                "W_n",
                "W_mean",
                "W_median",
                "W_iqm",
                "L1_n",
                "L1_mean",
                "L1_median",
                "L1_iqm",
                "metrics_path",
            ]
        )
    ]
    for r in rows:
        s = r["derived_six"]
        w, l1 = s["wasserstein"], s["l1_by_group"]
        lines.append(
            "\t".join(
                [
                    r["stock"],
                    r["variant"],
                    str(w["n"]),
                    fmt(w["mean"]),
                    fmt(w["median"]),
                    fmt(w["iqm"]),
                    str(l1["n"]),
                    fmt(l1["mean"]),
                    fmt(l1["median"]),
                    fmt(l1["iqm"]),
                    r["metrics_path"],
                ]
            )
        )

    table = "\n".join(lines) + "\n"
    print(table, end="")

    if args.out_tsv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_tsv)) or ".", exist_ok=True)
        with open(args.out_tsv, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"Wrote TSV: {args.out_tsv}", file=sys.stderr)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote JSON: {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()

