#!/usr/bin/env python3
"""
Emit LOB-Bench-style six scalars (W and L1: mean, median, IQM) from metrics_summary.json
for blank open-anchor vs sentence_preset_s2ip K=3 vs K=5 (per stock).

Uses the same pooling rules as compute_overall_scores_lobbench_style.py.
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
        "wasserstein": {
            "n": w_agg.n_metrics,
            "mean": w_agg.mean,
            "median": w_agg.median,
            "iqm": w_agg.iqm,
        },
        "l1_by_group": {
            "n": l1_agg.n_metrics,
            "mean": l1_agg.mean,
            "median": l1_agg.median,
            "iqm": l1_agg.iqm,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-json",
        default="",
        help="Write full table JSON here (default: print to stdout only).",
    )
    p.add_argument(
        "--out-tsv",
        default="",
        help="Write TSV table here.",
    )
    args = p.parse_args()

    root = "/finance_ML/zhanghaohan/stock_language_model"
    blank_root = os.path.join(
        root, "saved_LOB_stream/pool_0709_0710_eval_0710"
    )
    k3_root = os.path.join(
        root,
        "saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/sentence_preset_s2ip_k3/sentence_preset_s2ip_k3_ep3",
    )
    k5_root = os.path.join(
        root,
        "saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/sentence_preset_s2ip_k5/sentence_preset_s2ip_k5_ep3",
    )

    stocks = ["000617XSHE", "000981XSHE", "002263XSHE", "002366XSHE"]

    blank_fixed: Dict[str, str] = {
        "000617XSHE": f"{blank_root}/fixed_start_model_blankgpt2_tokens_openbidanchor_txncomplete_000617XSHE_20260406_182708/metrics_summary.json",
        "000981XSHE": f"{blank_root}/fixed_start_model_blankgpt2_tokens_openbidanchor_txncomplete_000981XSHE_20260406_182710/metrics_summary.json",
        "002263XSHE": f"{blank_root}/fixed_start_model_blankgpt2_tokens_openbidanchor_txncomplete_002263XSHE_20260406_182710/metrics_summary.json",
        "002366XSHE": f"{blank_root}/fixed_start_model_blankgpt2_tokens_openbidanchor_txncomplete_002366XSHE_20260406_182710/metrics_summary.json",
    }

    rows: List[Dict[str, Any]] = []

    for tag in stocks:
        for variant, path_getter in [
            ("blank_3ep_openanchor", lambda t: blank_fixed[t]),
            ("sentence_s2ip_k3", lambda t: _find_latest_metrics_summary(k3_root, t)),
            ("sentence_s2ip_k5", lambda t: _find_latest_metrics_summary(k5_root, t)),
        ]:
            path = path_getter(tag)
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            summary = _load_metrics_summary(path)
            six = six_from_summary(summary)
            embedded = summary.get("lobbench_style_overall")
            rows.append(
                {
                    "stock": tag,
                    "variant": variant,
                    "metrics_path": path,
                    "derived_six": six,
                    "json_lobbench_style_overall": embedded,
                }
            )

    # TSV header
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
            ]
        )
    ]
    for r in rows:
        s = r["derived_six"]
        w, l1 = s["wasserstein"], s["l1_by_group"]

        def fmt(x: Any) -> str:
            return "" if x is None else f"{float(x):.10g}"

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
