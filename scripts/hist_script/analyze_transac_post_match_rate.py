#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count cancel (ExecType 52) and transaction-complete (70) legs that fail to match
any row in the *filtered* post table on (ChannelNo, ApplSeqNum, SecurityID).

This mirrors process_lob_data_real_flow_open_anchor_txn_complete up to the
left-merge + dropna(subset=['Price']): unmatched rows are those that would be
dropped and never enter final_result_for_merge as cancel/txn events.

Usage:
  python scripts/hist_script/analyze_transac_post_match_rate.py
  python scripts/hist_script/analyze_transac_post_match_rate.py --stocks 000617_XSHE,000981_XSHE
"""

import argparse
import gc

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = "/finance_ML/zhanghaohan/stock_language_model"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utility.sim_helper_unified import get_l1_bid_anchor_by_time  # noqa: E402


def _build_post_keys_for_match(
    order_post_dir: str,
    liquidity_mask_dir: str,
    lob_snap_dir: str,
    date_num_str: str,
    selected_stocks: list,
    anchor_time: str,
    filter_bo: bool,
):
    col_names = [
        "ChannelNo", "ApplSeqNum", "MDStreamID", "SecurityID", "SecurityIDSource",
        "Price", "OrderQty", "Side", "TransactTime", "OrdType", "LocalTime", "SeqNo", "MISC",
    ]
    order_post = pd.read_csv(order_post_dir, header=None, names=col_names)
    order_post = order_post[1:]
    order_post["ApplSeqNum"] = pd.to_numeric(order_post["ApplSeqNum"], errors="coerce").astype(int)
    order_post["SecurityID"] = order_post["SecurityID"].astype(str).str.zfill(6) + "_XSHE"
    if selected_stocks:
        order_post = order_post[order_post["SecurityID"].isin(selected_stocks)]

    liquidity_mask = pd.read_csv(liquidity_mask_dir)
    liquidity_mask_slice = liquidity_mask[liquidity_mask["Unnamed: 0"] == int(date_num_str)].reset_index()
    melted = liquidity_mask_slice.melt(
        id_vars=["index"],
        var_name="SecurityID",
        value_name="is_high_liquidity",
    )
    result_df = melted[["SecurityID", "is_high_liquidity"]][1:]
    merged_order_post = order_post.merge(result_df, how="left", on="SecurityID")

    merged_order_post["TransactDT_MS"] = pd.to_datetime(
        merged_order_post["TransactTime"], format="%H:%M:%S.%f"
    )
    merged_order_post["TransactDT_SEC"] = merged_order_post["TransactDT_MS"].dt.floor("S")
    merged = merged_order_post[merged_order_post["is_high_liquidity"] == 1]
    merged = merged.sort_values(["SecurityID", "TransactDT_MS"], kind="mergesort")

    if filter_bo:
        merged = merged[merged["TransactDT_SEC"].dt.time > pd.to_datetime("09:30:00").time()]

    for col in ["ChannelNo", "ApplSeqNum", "Side", "OrderQty", "OrdType"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged[merged["OrdType"] == 50].copy()
    merged["Price"] = pd.to_numeric(merged["Price"], errors="coerce")
    merged = merged.dropna(subset=["Price"]).copy()

    anchor_map = get_l1_bid_anchor_by_time(lob_snap_dir, anchor_time, selected_stocks=selected_stocks)
    merged["OpenAnchorBidPrice"] = merged["SecurityID"].map({k: v["anchor_bid_price"] for k, v in anchor_map.items()})
    merged = merged.dropna(subset=["OpenAnchorBidPrice"]).copy()

    price_match_df = merged[["ChannelNo", "ApplSeqNum", "SecurityID", "Price", "Side"]].copy()
    price_match_df = price_match_df.rename(columns={"Side": "OrigSide"})
    for col in ["ChannelNo", "ApplSeqNum", "OrigSide"]:
        price_match_df[col] = pd.to_numeric(price_match_df[col], errors="coerce")
    price_match_df["Price"] = pd.to_numeric(price_match_df["Price"], errors="coerce")
    # Dedupe keys: one post row per (ChannelNo, ApplSeqNum, SecurityID) for merge semantics
    price_match_df = price_match_df.drop_duplicates(subset=["ChannelNo", "ApplSeqNum", "SecurityID"], keep="first")

    del order_post, merged_order_post, merged
    gc.collect()
    return price_match_df


def _expand_transac(order_transac_dir: str, selected_stocks: list) -> pd.DataFrame:
    order_transac_col_names = [
        "ChannelNo", "ApplSeqNum", "MDStreamID", "BidApplSeqNum", "OfferApplSeqNum",
        "SecurityID", "SecurityIDSource", "LastPx", "LastQty", "ExecType",
        "TransactTime", "LocalTime", "SeqNo", "MISC",
    ]
    order_transac = pd.read_csv(order_transac_dir, header=None, names=order_transac_col_names)
    order_transac = order_transac[1:]
    order_transac["SecurityID"] = order_transac["SecurityID"].astype(str).str.zfill(6) + "_XSHE"
    if selected_stocks:
        order_transac = order_transac[order_transac["SecurityID"].isin(selected_stocks)]

    for col in ["ExecType", "BidApplSeqNum", "OfferApplSeqNum", "ChannelNo"]:
        order_transac[col] = pd.to_numeric(order_transac[col], errors="coerce")
    order_transac = order_transac[order_transac["ExecType"].isin([52, 70])].copy()

    bid_transac = order_transac.loc[
        order_transac["BidApplSeqNum"].notna() & (order_transac["BidApplSeqNum"] != 0),
        ["ChannelNo", "SecurityID", "LastQty", "TransactTime", "ExecType", "BidApplSeqNum"],
    ].rename(columns={"BidApplSeqNum": "ApplSeqNum"})
    offer_transac = order_transac.loc[
        order_transac["OfferApplSeqNum"].notna() & (order_transac["OfferApplSeqNum"] != 0),
        ["ChannelNo", "SecurityID", "LastQty", "TransactTime", "ExecType", "OfferApplSeqNum"],
    ].rename(columns={"OfferApplSeqNum": "ApplSeqNum"})

    expanded = pd.concat([bid_transac, offer_transac], ignore_index=True)
    expanded["ApplSeqNum"] = pd.to_numeric(expanded["ApplSeqNum"], errors="coerce")
    return expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stocks",
        default="000617_XSHE,000981_XSHE,002263_XSHE,002366_XSHE",
        help="Comma-separated SecurityID list.",
    )
    parser.add_argument("--day", default="20250710")
    parser.add_argument("--anchor-time", default="09:31:00")
    args = parser.parse_args()
    stocks = [s.strip() for s in args.stocks.split(",") if s.strip()]
    day = args.day
    root = "/finance_ML/zhanghaohan/LOB_data"
    lob_day = f"{root}/{day}"
    order_post_dir = f"{lob_day}/mdl_6_33_0.csv"
    order_transac_dir = f"{lob_day}/mdl_6_36_0.csv"
    lob_snap_dir = f"{lob_day}/mdl_6_28_0.csv"
    liquidity_mask_dir = f"{root}/misc_data/AVG_AMT_3M_7_1D_8390e8742c5e.csv"

    price_match_df = _build_post_keys_for_match(
        order_post_dir,
        liquidity_mask_dir,
        lob_snap_dir,
        day,
        stocks,
        args.anchor_time,
        filter_bo=True,
    )
    expanded = _expand_transac(order_transac_dir, stocks)

    merged = expanded.merge(
        price_match_df,
        how="left",
        on=["ChannelNo", "ApplSeqNum", "SecurityID"],
    )
    unmatched = merged["Price"].isna()
    matched = ~unmatched

    def _report_slice(name: str, mask: pd.Series):
        tot = int(mask.sum())
        if tot == 0:
            print(f"{name}: n=0")
            return
        u = int((mask & unmatched).sum())
        pct = 100.0 * u / tot
        print(f"{name}: total_legs={tot}  unmatched_to_post_table={u}  ({pct:.4f}%)")

    print("=== Transaction/cancel legs vs filtered OrdType=50 post table ===")
    print(f"Stocks: {stocks}  Day: {day}")
    print(f"Unique post keys (ChannelNo, ApplSeqNum, SecurityID) in match table: {len(price_match_df)}")
    print(f"Expanded transac legs (ExecType 52 or 70, bid/offer): {len(merged)}")
    print()
    _report_slice("ALL cancel+txn legs", pd.Series(True, index=merged.index))
    _report_slice("ExecType 52 (cancel)", merged["ExecType"] == 52)
    _report_slice("ExecType 70 (transaction_complete)", merged["ExecType"] == 70)
    print()
    for sid in stocks:
        m = merged["SecurityID"] == sid
        _report_slice(f"  [{sid}] ALL", m)
        _report_slice(f"  [{sid}] cancel", m & (merged["ExecType"] == 52))
        _report_slice(f"  [{sid}] txn_complete", m & (merged["ExecType"] == 70))


if __name__ == "__main__":
    main()
