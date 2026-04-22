#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For one stock: build txn-complete merged frames for day A and day B without binning,
pool rows (fit sides) to fit price/qty/interval bins once, then apply those bins to each
day separately and write two joblibs + two bin_records (same bin edges/distributions;
per-day anchors and immediate-aggressive stats).
"""

import argparse
import json
import os
import sys
import time

import joblib
import pandas as pd

PROJECT_ROOT = "/finance_ML/zhanghaohan/stock_language_model"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utility.sim_helper_unified import process_lob_data_real_flow_open_anchor_txn_complete  # noqa: E402


PRICE_BIN_NUM = 26
QTY_BIN_NUM = 26
INTERVAL_BIN_NUM = 12
ANCHOR_TIME = "09:31:00"


def _paths_for_day(root_dir: str, day: str):
    lob_day = os.path.join(root_dir, day)
    return (
        os.path.join(lob_day, "mdl_6_33_0.csv"),
        os.path.join(lob_day, "mdl_6_28_0.csv"),
        os.path.join(lob_day, "mdl_6_36_0.csv"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Two-day pooled bin fitting; separate per-day joblib/bin_record outputs."
    )
    parser.add_argument("--stock", required=True, help="e.g. 000617_XSHE")
    parser.add_argument("--day-a", default="20250709")
    parser.add_argument("--day-b", default="20250710")
    parser.add_argument(
        "--split-cancel-sides",
        action="store_true",
        help="6-way cancel sides (97/98); must match downstream replay.",
    )
    parser.add_argument(
        "--output-parent-subdir",
        default="pool_0709_0710_openbidanchor_txncomplete",
        help="Subdir under saved_LOB_stream/processed_real_flow/",
    )
    args = parser.parse_args()

    stock = str(args.stock).strip()
    stock_tag = stock.replace("_", "")
    day_a = args.day_a
    day_b = args.day_b
    ts = time.strftime("%Y%m%d_%H%M")

    n_side = 6 if args.split_cancel_sides else 5
    side_to_bin = (
        {49: 0, 50: 1, 97: 2, 98: 3, 129: 4, 130: 5}
        if args.split_cancel_sides
        else {49: 0, 50: 1, 99: 2, 129: 3, 130: 4}
    )
    flow_tag = "openbidanchor_txncomplete_splitcancel" if args.split_cancel_sides else "openbidanchor_txncomplete"

    root_dir = "/finance_ML/zhanghaohan/LOB_data"
    liquidity_mask_dir = "/finance_ML/zhanghaohan/LOB_data/misc_data/AVG_AMT_3M_7_1D_8390e8742c5e.csv"

    fit_sides = set(int(k) for k in side_to_bin.keys())

    def run_merge_only(day: str):
        post, snap, tr = _paths_for_day(root_dir, day)
        print(f"\n================ merge-only | {day} | {stock} | {flow_tag} ================")
        return process_lob_data_real_flow_open_anchor_txn_complete(
            order_post_dir=post,
            lob_snap_dir=snap,
            order_transac_dir=tr,
            liquidity_mask_dir=liquidity_mask_dir,
            selected_stocks=[stock],
            filter_bo=True,
            date_num_str=day,
            anchor_time=ANCHOR_TIME,
            price_bin_num=None,
            qty_bin_num=None,
            interval_bin_num=None,
            n_side=n_side,
            side_to_bin=side_to_bin,
            return_bin_record=False,
            split_cancel_sides=bool(args.split_cancel_sides),
        )

    df_a = run_merge_only(day_a)
    df_b = run_merge_only(day_b)

    fit_pool = pd_concat_fit(df_a, df_b, fit_sides)
    print(
        f"\n[Pooled fit] rows day_a={len(df_a)} day_b={len(df_b)} "
        f"fit_pool_rows={len(fit_pool)} (sides {sorted(fit_sides)})"
    )

    def run_binned(day: str):
        post, snap, tr = _paths_for_day(root_dir, day)
        print(f"\n================ binned | {day} | {stock} | {flow_tag} ================")
        return process_lob_data_real_flow_open_anchor_txn_complete(
            order_post_dir=post,
            lob_snap_dir=snap,
            order_transac_dir=tr,
            liquidity_mask_dir=liquidity_mask_dir,
            selected_stocks=[stock],
            filter_bo=True,
            date_num_str=day,
            anchor_time=ANCHOR_TIME,
            price_bin_num=PRICE_BIN_NUM,
            qty_bin_num=QTY_BIN_NUM,
            interval_bin_num=INTERVAL_BIN_NUM,
            n_side=n_side,
            side_to_bin=side_to_bin,
            return_bin_record=True,
            split_cancel_sides=bool(args.split_cancel_sides),
            external_fit_dataframe=fit_pool,
        )

    df_a_out, br_a = run_binned(day_a)
    df_b_out, br_b = run_binned(day_b)

    for br, dlabel in ((br_a, day_a), (br_b, day_b)):
        br["pooled_fit_trade_dates"] = [day_a, day_b]
        br["pooled_fit_stock"] = stock
        br["pooled_fit_note"] = (
            "Price/qty/interval bin edges and bin_value_distributions were fit on the concatenation "
            "of txn-complete rows (fit sides) from these trade dates for this stock."
        )

    base_processed = os.path.join(PROJECT_ROOT, "saved_LOB_stream", "processed_real_flow")
    subdir = str(args.output_parent_subdir).strip().strip("/").replace("..", "")
    output_dir = os.path.join(base_processed, subdir) if subdir else base_processed
    os.makedirs(output_dir, exist_ok=True)

    def write_day(df_out, br, day: str):
        df_out = df_out.copy()
        df_out["TradeDate"] = day
        job = os.path.join(
            output_dir,
            f"final_result_for_merge_realflow_{flow_tag}_{day}_{stock_tag}_{ts}.joblib",
        )
        jbr = os.path.join(
            output_dir,
            f"bin_record_realflow_{flow_tag}_{day}_{stock_tag}_{ts}.json",
        )
        joblib.dump(df_out, job, compress=3)
        with open(jbr, "w", encoding="utf-8") as f:
            json.dump(br, f, indent=2, ensure_ascii=False)
        anchor_meta = (br or {}).get("price_anchor_by_stock", {}).get(stock, {})
        summary = {
            "day": day,
            "stock": stock,
            "regime_label": f"{day}_{flow_tag}_pooled_with_{day_a}_{day_b}",
            "output_parent_subdir": subdir if subdir else None,
            "pooled_fit_trade_dates": [day_a, day_b],
            "created_at": ts,
            "rows": int(len(df_out)),
            "columns": list(df_out.columns),
            "output": {"joblib": job, "bin_record": jbr},
            "price_anchor": {"time": ANCHOR_TIME, "stock_anchor": anchor_meta},
            "event_schema": {
                "n_side": n_side,
                "side_to_bin": side_to_bin,
                "split_cancel_sides": bool(args.split_cancel_sides),
            },
        }
        js = os.path.join(
            output_dir,
            f"final_result_for_merge_realflow_{flow_tag}_{day}_{stock_tag}_{ts}.json",
        )
        with open(js, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[{day}] rows={len(df_out)} joblib={job}")
        print(f"[{day}] bin_record={jbr}")

    write_day(df_a_out, br_a, day_a)
    write_day(df_b_out, br_b, day_b)
    print("\nDone two-day pooled preprocess.")


def pd_concat_fit(df_a, df_b, fit_sides):
    a = df_a[df_a["Side"].isin(fit_sides)].copy()
    b = df_b[df_b["Side"].isin(fit_sides)].copy()
    return pd.concat([a, b], ignore_index=True)


if __name__ == "__main__":
    main()
