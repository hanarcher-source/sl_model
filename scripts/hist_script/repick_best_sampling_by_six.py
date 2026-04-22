#!/usr/bin/env python3
"""
Repick best (temperature, top-k) sampling setting per stock using all 6 aggregate scores:
  Wasserstein: mean / median / IQM
  L1_by_group: mean / median / IQM

We reuse the same extraction rules as compute_overall_scores_lobbench_style.py.

Selection rule (scale-free, uses all six):
  - For each stock, compute a rank for every setting on each of the six scalars (lower is better).
  - Sum ranks across available scalars; choose the lowest rank-sum.
  - Also report the original best-by-W_mean winner for comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from compute_overall_scores_lobbench_style import (  # noqa: E402
    _aggregate,
    _collect_metric_losses,
    _load_metrics_summary,
)


def _six_from_path(path: str) -> Dict[str, Dict[str, Any]]:
    s = _load_metrics_summary(path)
    w_map, l1_map = _collect_metric_losses(s)
    w = _aggregate(w_map.values())
    l1 = _aggregate(l1_map.values())
    return {
        "wasserstein": {"n": w.n_metrics, "mean": w.mean, "median": w.median, "iqm": w.iqm},
        "l1_by_group": {"n": l1.n_metrics, "mean": l1.mean, "median": l1.median, "iqm": l1.iqm},
    }


def _rank(values: List[Tuple[str, Optional[float]]]) -> Dict[str, int]:
    """
    values: list of (key, value) where lower is better; None means missing.
    Returns: key -> rank starting at 1 (ties get the same rank-min style).
    """
    present = [(k, float(v)) for k, v in values if v is not None]
    present.sort(key=lambda kv: kv[1])
    out: Dict[str, int] = {k: 10**9 for k, v in values}  # big for missing
    rank = 1
    i = 0
    while i < len(present):
        j = i + 1
        while j < len(present) and present[j][1] == present[i][1]:
            j += 1
        for k, _ in present[i:j]:
            out[k] = rank
        rank += (j - i)
        i = j
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sweep-json",
        default="/finance_ML/zhanghaohan/stock_language_model/logs/eval_pool0709_0710/sweep_best_lobbench_aggregate.json",
        help="JSON that lists per-stock rows with setting + metrics_summary path.",
    )
    ap.add_argument(
        "--out-json",
        default="/finance_ML/zhanghaohan/stock_language_model/logs/eval_pool0709_0710/repick_best_sampling_by_six.json",
        help="Write results JSON here.",
    )
    args = ap.parse_args()

    with open(args.sweep_json, encoding="utf-8") as f:
        sweep = json.load(f)
    rows = sweep.get("rows") or []
    winners = {w["stock"]: w for w in (sweep.get("winners") or []) if isinstance(w, dict) and "stock" in w}

    by_stock: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        st = str(r.get("stock", "")).strip()
        setting = str(r.get("setting", "")).strip()
        path = str(r.get("metrics_path", "")).strip()
        if not st or not setting or not path:
            continue
        by_stock.setdefault(st, []).append({"setting": setting, "metrics_path": path})

    report: Dict[str, Any] = {
        "selection_rule": "rank_sum_over_six (W mean/median/iqm + L1 mean/median/iqm; lower is better)",
        "stocks": {},
        "source_sweep_json": args.sweep_json,
    }

    for st, items in sorted(by_stock.items()):
        # compute six for each setting
        enriched: List[Dict[str, Any]] = []
        for it in items:
            path = it["metrics_path"]
            six = _six_from_path(path)
            enriched.append({**it, "six": six})

        # gather scalar lists per dimension
        dims = [
            ("W_mean", ("wasserstein", "mean")),
            ("W_median", ("wasserstein", "median")),
            ("W_iqm", ("wasserstein", "iqm")),
            ("L1_mean", ("l1_by_group", "mean")),
            ("L1_median", ("l1_by_group", "median")),
            ("L1_iqm", ("l1_by_group", "iqm")),
        ]

        ranks_per_dim: Dict[str, Dict[str, int]] = {}
        for dim_name, (block, key) in dims:
            vals = [(e["setting"], e["six"][block][key]) for e in enriched]
            ranks_per_dim[dim_name] = _rank(vals)

        # rank-sum
        rank_sum: Dict[str, int] = {e["setting"]: 0 for e in enriched}
        for dim_name, _ in dims:
            rk = ranks_per_dim[dim_name]
            for setting in rank_sum:
                rank_sum[setting] += int(rk.get(setting, 10**9))

        best_setting = min(rank_sum.items(), key=lambda kv: kv[1])[0]

        # attach top few
        def _scalar(e, block, key):
            return e["six"][block][key]

        table = []
        for e in sorted(enriched, key=lambda e: rank_sum[e["setting"]])[:8]:
            table.append(
                {
                    "setting": e["setting"],
                    "rank_sum": int(rank_sum[e["setting"]]),
                    "W": e["six"]["wasserstein"],
                    "L1": e["six"]["l1_by_group"],
                    "metrics_path": e["metrics_path"],
                }
            )

        report["stocks"][st] = {
            "best_by_rank_sum_over_six": best_setting,
            "best_rank_sum": int(rank_sum[best_setting]),
            "original_best_by_W_mean": (winners.get(st) or {}).get("best_setting"),
            "top_candidates": table,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()

