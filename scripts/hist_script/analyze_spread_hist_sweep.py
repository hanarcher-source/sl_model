#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze spread (in ticks) for the pooled-bin 0709+0710 sweep runs.

For each stock:
- load clean reference LOBSTER orderbook CSV (ask1 - bid1)
- load each sweep setting's generated orderbook CSV
- compute mean/median/p95 spread in ticks and mean gap vs reference
- pick best setting by minimal absolute mean gap vs reference (tie-break: smaller abs(Wasserstein spread) if available, else smaller mean)
- write a side-by-side histogram CSV for the best setting

This script reads existing artifacts only; it does NOT rerun generation/eval.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# LOBSTER orderbook layout used by LOB_bench in this repo:
# columns are (ask1P, ask1Q, bid1P, bid1Q, ask2P, ask2Q, bid2P, bid2Q, ...)
ASK1P_COL = 0
BID1P_COL = 2

# For CN stocks with tick 0.01 and price_int scaled by 1e4:
# 1 tick (0.01 CNY) = 0.01 * 10000 = 100 price_int units.
TICK_SIZE_PRICE_INT = 100.0


@dataclass
class SpreadStats:
    mean: float
    median: float
    std: float
    p95: float
    n: int


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _spread_ticks_from_orderbook_csv(path: str) -> pd.Series:
    book = pd.read_csv(path, header=None)
    ask1 = pd.to_numeric(book.iloc[:, ASK1P_COL], errors="coerce")
    bid1 = pd.to_numeric(book.iloc[:, BID1P_COL], errors="coerce")
    spr = (ask1 - bid1) / TICK_SIZE_PRICE_INT
    spr = spr.replace([np.inf, -np.inf], np.nan).dropna()
    return spr


def _stats(s: pd.Series) -> SpreadStats:
    return SpreadStats(
        mean=float(s.mean()),
        median=float(s.median()),
        std=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        p95=float(s.quantile(0.95)),
        n=int(len(s)),
    )


def _hist_pct(s: pd.Series) -> pd.DataFrame:
    vc = s.value_counts(dropna=True).sort_index()
    pct = vc / vc.sum() * 100.0
    return pd.DataFrame({"spread_ticks": vc.index, "count": vc.values, "pct": pct.values})


def _find_single(pattern: str) -> str:
    import glob

    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No matches for pattern: {pattern}")
    if len(matches) > 1:
        # use latest by mtime
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def _extract_spread_wasserstein(metrics_summary_path: str) -> Optional[float]:
    try:
        d = _read_json(metrics_summary_path)
        m = (d.get("reference_comparison") or {}).get("metrics") or {}
        payload = m.get("spread") or {}
        v = payload.get("wasserstein")
        return float(v) if v is not None else None
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sweep-root",
        default="/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/pool_0709_0710_eval_0710_sweep",
        help="Folder containing sweep subdirs like T13_k0/, T10_k200/ ...",
    )
    p.add_argument(
        "--settings",
        default="T07_k0,T13_k0,T10_k50,T10_k200",
        help="Comma-separated sweep subdir names to compare.",
    )
    p.add_argument(
        "--stocks",
        default="000617_XSHE,000981_XSHE,002263_XSHE,002366_XSHE",
        help="Comma-separated stock codes.",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Optional output directory for histogram CSVs. Default: <sweep-root>/spread_hist_reports",
    )
    args = p.parse_args()

    sweep_root = os.path.abspath(args.sweep_root)
    settings = [s.strip() for s in str(args.settings).split(",") if s.strip()]
    stocks = [s.strip() for s in str(args.stocks).split(",") if s.strip()]
    out_dir = os.path.abspath(args.out_dir) if str(args.out_dir).strip() else os.path.join(sweep_root, "spread_hist_reports")
    os.makedirs(out_dir, exist_ok=True)

    print("Sweep root:", sweep_root)
    print("Settings:", settings)
    print("Stocks:", stocks)
    print("Output dir:", out_dir)
    print("")

    for stock in stocks:
        tag = stock.replace("_", "")
        print("=" * 90)
        print("STOCK:", stock)

        # Use any one setting's generation_notes to find the clean ref dir for this stock
        notes_path = _find_single(os.path.join(sweep_root, settings[0], f"fixed_start_model_blankgpt2_tokens_openbidanchor_txncomplete_{tag}_*/generation_notes.json"))
        notes = _read_json(notes_path)
        ref_dir = ((notes.get("paths") or {}).get("real_ref_dir")) or ""
        if not ref_dir or not os.path.isdir(ref_dir):
            # fallback: glob by naming convention
            ref_dir = _find_single(os.path.join(os.path.dirname(sweep_root), f"fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_{tag}_*"))
        ref_ob = _find_single(os.path.join(ref_dir, f"{tag}_2025-07-10_*_orderbook_10.csv"))

        ref_sp = _spread_ticks_from_orderbook_csv(ref_ob)
        ref_stats = _stats(ref_sp)
        print(f"[Clean ref] orderbook={ref_ob}")
        print(f"[Clean ref] spread_ticks mean={ref_stats.mean:.6g} median={ref_stats.median:.6g} p95={ref_stats.p95:.6g} std={ref_stats.std:.6g} n={ref_stats.n}")
        print("")

        rows: List[dict] = []
        for setting in settings:
            exp_notes = _find_single(
                os.path.join(
                    sweep_root,
                    setting,
                    f"fixed_start_model_blankgpt2_tokens_openbidanchor_txncomplete_{tag}_*/generation_notes.json",
                )
            )
            n = _read_json(exp_notes)
            ob = n.get("lobster_orderbook_csv") or ""
            ms = n.get("metrics_summary_json") or os.path.join(os.path.dirname(exp_notes), "metrics_summary.json")
            if not ob or not os.path.isfile(ob):
                # fallback: orderbook in same folder
                ob = _find_single(os.path.join(os.path.dirname(exp_notes), f"*orderbook_10.csv"))

            sp = _spread_ticks_from_orderbook_csv(ob)
            st = _stats(sp)
            gap = st.mean - ref_stats.mean
            ws = _extract_spread_wasserstein(ms) if os.path.isfile(ms) else None
            rows.append(
                {
                    "setting": setting,
                    "temperature": float(n.get("temperature")) if n.get("temperature") is not None else None,
                    "top_k": int(n.get("top_k")) if n.get("top_k") is not None else None,
                    "orderbook_csv": ob,
                    "metrics_summary_json": ms if os.path.isfile(ms) else None,
                    "spread_mean_ticks": st.mean,
                    "spread_median_ticks": st.median,
                    "spread_p95_ticks": st.p95,
                    "spread_std_ticks": st.std,
                    "n_rows": st.n,
                    "mean_gap_vs_ref_ticks": gap,
                    "abs_mean_gap_vs_ref_ticks": abs(gap),
                    "spread_wasserstein_vs_ref": ws,
                }
            )

        df = pd.DataFrame(rows).sort_values(["abs_mean_gap_vs_ref_ticks", "spread_wasserstein_vs_ref", "spread_mean_ticks"], ascending=[True, True, True])
        print("Per-setting mean spread + gap vs clean ref (ticks):")
        print(
            df[
                [
                    "setting",
                    "temperature",
                    "top_k",
                    "spread_mean_ticks",
                    "mean_gap_vs_ref_ticks",
                    "abs_mean_gap_vs_ref_ticks",
                    "spread_wasserstein_vs_ref",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.6g}")
        )
        print("")

        best = df.iloc[0].to_dict()
        best_setting = str(best["setting"])
        print(f"[BEST] setting={best_setting} temp={best.get('temperature')} top_k={best.get('top_k')}")
        print(f"[BEST] mean_gap_vs_ref_ticks={best['mean_gap_vs_ref_ticks']:.6g} | ref_mean={ref_stats.mean:.6g} gen_mean={best['spread_mean_ticks']:.6g}")

        # write histogram comparison for best setting
        best_ob = str(best["orderbook_csv"])
        best_sp = _spread_ticks_from_orderbook_csv(best_ob)
        h_gen = _hist_pct(best_sp).rename(columns={"count": "count_gen", "pct": "pct_gen"})
        h_ref = _hist_pct(ref_sp).rename(columns={"count": "count_ref", "pct": "pct_ref"})
        cmp = h_gen.merge(h_ref, on="spread_ticks", how="outer").fillna(0).sort_values("spread_ticks")
        cmp["delta_pct_gen_minus_ref"] = cmp["pct_gen"] - cmp["pct_ref"]

        out_csv = os.path.join(out_dir, f"spread_hist_best_{tag}_{best_setting}.csv")
        cmp.to_csv(out_csv, index=False)

        # print concise view (>=1% mass in either)
        view = cmp[(cmp["pct_gen"] >= 1.0) | (cmp["pct_ref"] >= 1.0)].copy()
        print("\nHistogram (pct) for spreads with >=1% mass in gen or ref:")
        if len(view) == 0:
            print("  (no bins over 1%)")
        else:
            print(view[["spread_ticks", "pct_gen", "pct_ref", "delta_pct_gen_minus_ref"]].to_string(index=False, float_format=lambda x: f"{x:.6g}"))
        print(f"\nWrote histogram CSV: {out_csv}")
        print("")


if __name__ == "__main__":
    main()

