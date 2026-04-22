#!/usr/bin/env python3
"""
Read metrics_summary.json from blank-GPT2 eval runs and compare LOB-Bench-style Wasserstein
aggregates across context window (50 / 100 / 200) for the same per-stock sampling settings
as the lbmean batch (T13_k0 for three stocks, T13_k400 for 000981).

Expects lobbench_style_overall in each JSON (re-run eval_generated_stream after inject landed).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().replace("NaN", "null").replace("Infinity", "null").replace("-Infinity", "null")
    return json.loads(raw)


def _w_block(ms: dict) -> Optional[dict]:
    lob = ms.get("lobbench_style_overall") or {}
    w = lob.get("wasserstein")
    if not isinstance(w, dict) or not w.get("n"):
        return None
    return w


def _latest_metrics(glob_pat: str) -> Optional[str]:
    hits = glob.glob(glob_pat)
    if not hits:
        return None
    hits.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/finance_ML/zhanghaohan/stock_language_model")
    ap.add_argument("--sweep-root", default="saved_LOB_stream/pool_0709_0710_eval_0710_sweep")
    ap.add_argument("--lbmean-root", default="saved_LOB_stream/pool_0709_0710_eval_0710_model_variants")
    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    sweep_root = os.path.join(root, args.sweep_root)
    lb_root = os.path.join(root, args.lbmean_root)

    stocks = ["000617XSHE", "000981XSHE", "002263XSHE", "002366XSHE"]
    setting_for = {
        "000617XSHE": "T13_k0",
        "002263XSHE": "T13_k0",
        "002366XSHE": "T13_k0",
        "000981XSHE": "T13_k400",
    }

    rows: List[Dict[str, Any]] = []

    for st in stocks:
        setting = setting_for[st]
        # win50: sweep
        g50 = os.path.join(sweep_root, setting, f"fixed_start_model_blankgpt2_tokens_openbidanchor_txncomplete_{st}_*", "metrics_summary.json")
        p50 = _latest_metrics(g50)
        # win100 / win200: lbmean_*_blank_win{100,200} with matching setting in folder name
        suf = f"{setting}_blank_win100"
        g100 = os.path.join(lb_root, f"lbmean_{suf}", f"fixed_start_*_{st}_*", "metrics_summary.json")
        p100 = _latest_metrics(g100)
        suf200 = f"{setting}_blank_win200"
        g200 = os.path.join(lb_root, f"lbmean_{suf200}", f"fixed_start_*_{st}_*", "metrics_summary.json")
        p200 = _latest_metrics(g200)

        def row(win: int, path: Optional[str]) -> None:
            if not path or not os.path.isfile(path):
                rows.append({"stock": st, "window": win, "missing": True, "path": path})
                return
            ms = _load_json(path)
            w = _w_block(ms)
            rows.append(
                {
                    "stock": st,
                    "window": win,
                    "missing": w is None,
                    "path": path,
                    "W_mean": w["mean"] if w else None,
                    "W_median": w["median"] if w else None,
                    "W_iqm": w["iqm"] if w else None,
                    "n_w": w["n"] if w else None,
                }
            )

        row(50, p50)
        row(100, p100)
        row(200, p200)

    # Print markdown table per stock
    print("## Blank GPT-2: LOB-Bench-style Wasserstein aggregate vs context window")
    print("(Same sampling as lbmean batch: T13_k0 @ temp=1.3,k=0 for 617/2263/2366; T13_k400 @ temp=1.3,k=400 for 981.)")
    print("")
    for st in stocks:
        sub = [r for r in rows if r["stock"] == st]
        print(f"### {st}")
        print("| window | W_mean | W_median | W_iqm | n |")
        print("|--------|--------|----------|-------|---|")
        for w in (50, 100, 200):
            r = next((x for x in sub if x["window"] == w), None)
            if not r or r.get("missing"):
                print(f"| {w} | — | — | — | — |")
                continue
            print(
                f"| {w} | {r['W_mean']:.6g} | {r['W_median']:.6g} | {r['W_iqm']:.6g} | {r['n_w']} |"
            )
        print("")

    missing = [r for r in rows if r.get("missing") or r["path"] is None]
    if missing:
        print("### Missing or incomplete (re-run reeval script / eval_generated_stream)")
        for r in missing:
            print(f"  stock={r['stock']} window={r['window']} path={r.get('path')}")

    if args.out_csv:
        import csv

        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["stock", "window", "W_mean", "W_median", "W_iqm", "n_w", "missing", "path"])
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "stock": r["stock"],
                        "window": r["window"],
                        "W_mean": r.get("W_mean"),
                        "W_median": r.get("W_median"),
                        "W_iqm": r.get("W_iqm"),
                        "n_w": r.get("n_w"),
                        "missing": r.get("missing"),
                        "path": r.get("path"),
                    }
                )
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
