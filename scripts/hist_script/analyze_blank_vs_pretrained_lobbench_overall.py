#!/usr/bin/env python3
"""
Compare LOB-Bench-style Wasserstein aggregates: blank GPT-2 win50 vs pretrained GPT-2 win50,
using the same per-stock sampling as LOB-Bench-mean selection:
  T13_k0 (temp=1.3, k=0) for 000617 / 002263 / 002366
  T13_k400 (temp=1.3, k=400) for 000981

Blank: sweep dirs. Pretrained: best_T13_k0_pretrained_win50 + lbmean_T13_k400_pretrained_win50 (981).
Requires lobbench_style_overall in each metrics_summary.json (re-run eval_generated_stream if missing).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional


def _load_ms(path: str) -> dict:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().replace("NaN", "null").replace("Infinity", "null").replace("-Infinity", "null")
    return json.loads(raw)


def _w_mean(path: Optional[str]) -> Optional[float]:
    if not path or not os.path.isfile(path):
        return None
    ms = _load_ms(path)
    lob = ms.get("lobbench_style_overall") or {}
    w = lob.get("wasserstein") or {}
    m = w.get("mean")
    return float(m) if m is not None else None


def _latest_glob(pattern: str) -> Optional[str]:
    hits = glob.glob(pattern)
    if not hits:
        return None
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/finance_ML/zhanghaohan/stock_language_model")
    ap.add_argument("--sweep-root", default="saved_LOB_stream/pool_0709_0710_eval_0710_sweep")
    ap.add_argument("--model-variants-root", default="saved_LOB_stream/pool_0709_0710_eval_0710_model_variants")
    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    sweep = os.path.join(root, args.sweep_root)
    mv = os.path.join(root, args.model_variants_root)

    stocks = [
        ("000617XSHE", "T13_k0", "best_T13_k0_pretrained_win50"),
        ("002263XSHE", "T13_k0", "best_T13_k0_pretrained_win50"),
        ("002366XSHE", "T13_k0", "best_T13_k0_pretrained_win50"),
        ("000981XSHE", "T13_k400", "lbmean_T13_k400_pretrained_win50"),
    ]

    print("## Blank vs pretrained (win50, LOB-Bench-mean sampling per stock)")
    print("")
    print("| stock | blank W_mean | pretrained W_mean | Δ (pre − blank) |")
    print("|-------|-------------:|----------------:|----------------:|")

    rows: List[Dict[str, Any]] = []
    for tag, sweep_set, pre_sub in stocks:
        blank_p = _latest_glob(
            os.path.join(sweep, sweep_set, f"fixed_start_model_blankgpt2_tokens_openbidanchor_txncomplete_{tag}_*", "metrics_summary.json")
        )
        pre_p = _latest_glob(
            os.path.join(mv, pre_sub, f"fixed_start_model_blankgpt2_tokens_openbidanchor_txncomplete_{tag}_*", "metrics_summary.json")
        )
        bw = _w_mean(blank_p)
        pw = _w_mean(pre_p)
        delta = (pw - bw) if (pw is not None and bw is not None) else None
        rows.append(
            {
                "stock": tag,
                "blank_W_mean": bw,
                "pretrained_W_mean": pw,
                "delta_pretrained_minus_blank": delta,
                "blank_path": blank_p,
                "pretrained_path": pre_p,
            }
        )
        ds = f"{delta:+.6g}" if delta is not None else "—"
        bws = f"{bw:.6g}" if bw is not None else "—"
        pws = f"{pw:.6g}" if pw is not None else "—"
        print(f"| {tag} | {bws} | {pws} | {ds} |")

    print("")
    missing_pre = [r for r in rows if r["pretrained_W_mean"] is None]
    if missing_pre:
        print("### Missing pretrained metrics (inference + eval, then re-eval if needed)")
        for r in missing_pre:
            print(f"  - {r['stock']}: pretrained_path={r['pretrained_path']}")

    if args.out_csv:
        import csv

        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "stock",
                    "blank_W_mean",
                    "pretrained_W_mean",
                    "delta_pretrained_minus_blank",
                    "blank_path",
                    "pretrained_path",
                ],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
