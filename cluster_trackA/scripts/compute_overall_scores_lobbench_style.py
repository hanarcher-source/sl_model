#!/usr/bin/env python3
"""
Compute LOB-Bench-style aggregate scores (mean/median/IQM across scoring functions)
from our `metrics_summary.json` artifacts.

LOB-Bench paper text (sec 5) states aggregate model scores are reported via mean,
median and IQM across all conditional + unconditional scoring functions.

Our `metrics_summary.json` is not strict JSON (may contain NaN), so we load it by
sanitizing NaN/Infinity tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_metrics_summary(path: str) -> Dict[str, Any]:
    raw = _read_text(path)
    # Our file is JSON-like but may contain NaN, Infinity; make it JSON.
    raw = raw.replace("NaN", "null")
    raw = raw.replace("Infinity", "null").replace("-Infinity", "null")
    return json.loads(raw)


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(float(x)) or math.isinf(float(x)))


def _iqm(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    if n == 1:
        return float(s[0])

    def _pct(p: float) -> float:
        # linear interpolation percentile, similar to numpy default
        idx = (n - 1) * p
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return float(s[lo])
        w = idx - lo
        return float(s[lo] * (1.0 - w) + s[hi] * w)

    q25 = _pct(0.25)
    q75 = _pct(0.75)
    core = [v for v in s if (v >= q25 and v <= q75)]
    if not core:
        return float(sum(s)) / float(len(s))
    return float(sum(core)) / float(len(core))


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


@dataclass
class AggregateScores:
    n_metrics: int
    mean: Optional[float]
    median: Optional[float]
    iqm: Optional[float]


def _collect_metric_losses(summary: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns:
      - wasserstein_losses: metric_name -> loss_value
      - l1_losses: metric_name -> loss_value

    We look for dicts that contain per-metric comparative losses:
      - unconditional: {"wasserstein": ..., "l1_by_group": ...}
      - conditional: {"weighted_wasserstein": ...} (may also contain per_group ...)

    For conditional metrics, if weighted_wasserstein is null, we skip (because the
    benchmark metric isn't well-defined for that run in our pipeline).
    """
    w: Dict[str, float] = {}
    l1: Dict[str, float] = {}

    # Our pipeline stores real-vs-generated losses under:
    #   summary["reference_comparison"]["metrics"][metric_name] = {...}
    # (This corresponds to the block logged as "[real-vs-generated comparative metrics]".)
    ref = summary.get("reference_comparison")
    metrics = None
    if isinstance(ref, dict) and isinstance(ref.get("metrics"), dict):
        metrics = ref.get("metrics")
    else:
        metrics = {}

    for metric_name, payload in metrics.items():
        if not isinstance(payload, dict):
            continue

        # Prefer weighted conditional score where present and finite.
        if "weighted_wasserstein" in payload and _is_finite_number(payload.get("weighted_wasserstein")):
            w[metric_name] = float(payload["weighted_wasserstein"])
        elif _is_finite_number(payload.get("wasserstein")):
            w[metric_name] = float(payload["wasserstein"])

        if _is_finite_number(payload.get("l1_by_group")):
            l1[metric_name] = float(payload["l1_by_group"])

    return w, l1


def _aggregate(vals: Iterable[float]) -> AggregateScores:
    xs = [float(x) for x in vals if _is_finite_number(x)]
    if not xs:
        return AggregateScores(n_metrics=0, mean=None, median=None, iqm=None)
    return AggregateScores(
        n_metrics=len(xs),
        mean=float(sum(xs)) / float(len(xs)),
        median=_median(xs),
        iqm=_iqm(xs),
    )


def _find_latest_metrics_summary(root: str, stock_tag: str) -> str:
    """
    Find latest `metrics_summary.json` under root matching the stock tag.
    """
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
        raise FileNotFoundError(f"Could not find metrics_summary.json for stock_tag={stock_tag} under {root}")
    hits.sort(reverse=True)
    return hits[0][1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-root", required=True, help="Root folder containing BEFORE runs (e.g. sweep root setting dir).")
    ap.add_argument("--after-root", required=True, help="Root folder containing AFTER runs (e.g. pretrained variant dir).")
    ap.add_argument(
        "--stocks",
        required=True,
        help="Comma-separated stock tags (e.g. 000617XSHE,000981XSHE,002263XSHE,002366XSHE).",
    )
    ap.add_argument(
        "--before-setting",
        required=True,
        help=(
            "Comma-separated mapping stock_tag=SETTING where SETTING is a subdir under --before-root "
            "(e.g. 000617XSHE=T13_k0,000981XSHE=T10_k50,002263XSHE=T13_k0,002366XSHE=T13_k0)"
        ),
    )
    ap.add_argument(
        "--after-subdir",
        default="",
        help="Optional subdir inside --after-root to search within (useful if after-root contains multiple variants).",
    )
    args = ap.parse_args()

    stocks = [s.strip() for s in args.stocks.split(",") if s.strip()]
    before_map: Dict[str, str] = {}
    for item in args.before_setting.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad --before-setting entry: {item}")
        st, setting = item.split("=", 1)
        before_map[st.strip()] = setting.strip()

    after_root = os.path.join(args.after_root, args.after_subdir) if args.after_subdir else args.after_root

    rows: List[Dict[str, Any]] = []
    for st in stocks:
        if st not in before_map:
            raise ValueError(f"Missing before setting for stock {st} in --before-setting")
        before_setting = before_map[st]
        before_root = os.path.join(args.before_root, before_setting)

        before_path = _find_latest_metrics_summary(before_root, st)
        after_path = _find_latest_metrics_summary(after_root, st)

        before_sum = _load_metrics_summary(before_path)
        after_sum = _load_metrics_summary(after_path)

        bw, bl1 = _collect_metric_losses(before_sum)
        aw, al1 = _collect_metric_losses(after_sum)

        bw_agg = _aggregate(bw.values())
        aw_agg = _aggregate(aw.values())
        bl1_agg = _aggregate(bl1.values())
        al1_agg = _aggregate(al1.values())

        rows.append(
            {
                "stock": st,
                "before_setting": before_setting,
                "before_w_n": bw_agg.n_metrics,
                "before_w_mean": bw_agg.mean,
                "before_w_median": bw_agg.median,
                "before_w_iqm": bw_agg.iqm,
                "after_w_n": aw_agg.n_metrics,
                "after_w_mean": aw_agg.mean,
                "after_w_median": aw_agg.median,
                "after_w_iqm": aw_agg.iqm,
                "before_l1_n": bl1_agg.n_metrics,
                "before_l1_mean": bl1_agg.mean,
                "before_l1_median": bl1_agg.median,
                "before_l1_iqm": bl1_agg.iqm,
                "after_l1_n": al1_agg.n_metrics,
                "after_l1_mean": al1_agg.mean,
                "after_l1_median": al1_agg.median,
                "after_l1_iqm": al1_agg.iqm,
                "before_metrics_path": before_path,
                "after_metrics_path": after_path,
            }
        )

    # Print a compact table to stdout (Slurm log)
    def fmt(x: Any) -> str:
        if x is None:
            return "NA"
        if isinstance(x, (int, float)):
            return f"{float(x):.6g}"
        return str(x)

    print("LOB-Bench-style aggregate scores (lower is better)")
    print("Aggregation across metric configs with available comparative losses.")
    print("")

    headers = [
        "stock",
        "before_setting",
        "W_mean_before",
        "W_mean_after",
        "W_med_before",
        "W_med_after",
        "W_IQM_before",
        "W_IQM_after",
        "L1_mean_before",
        "L1_mean_after",
        "L1_med_before",
        "L1_med_after",
        "L1_IQM_before",
        "L1_IQM_after",
        "nW_before",
        "nW_after",
    ]
    print("\t".join(headers))
    for r in rows:
        print(
            "\t".join(
                [
                    r["stock"],
                    r["before_setting"],
                    fmt(r["before_w_mean"]),
                    fmt(r["after_w_mean"]),
                    fmt(r["before_w_median"]),
                    fmt(r["after_w_median"]),
                    fmt(r["before_w_iqm"]),
                    fmt(r["after_w_iqm"]),
                    fmt(r["before_l1_mean"]),
                    fmt(r["after_l1_mean"]),
                    fmt(r["before_l1_median"]),
                    fmt(r["after_l1_median"]),
                    fmt(r["before_l1_iqm"]),
                    fmt(r["after_l1_iqm"]),
                    str(r["before_w_n"]),
                    str(r["after_w_n"]),
                ]
            )
        )

    # Also emit a machine-readable JSON next to CWD if desired
    out_json = "overall_scores_lobbench_style.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()

