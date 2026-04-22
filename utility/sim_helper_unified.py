#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified LOB simulation helper supporting both model variants:
- model_variant="anchor"
- model_variant="no_anchor"

This file merges the common simulation/eval utilities from:
- sim_func_ng.py
- sim_func_ng_abl.py
"""

import json
import math
import os
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model
from tqdm import tqdm

# ============================================================
# BASIC HELPERS
# ============================================================

def get_lob_snapshot_by_time(lob_info, target_time, stock, stock_col="SecurityID"):
    target_ts = pd.Timestamp(f"1900-01-01 {target_time}")

    cols = [stock_col, "TransactDT_SEC"]
    cols += [f"BidPrice{i}" for i in range(1, 11)]
    cols += [f"AskPrice{i}" for i in range(1, 11)]
    cols += [f"BidVolume{i}" for i in range(1, 11)]
    cols += [f"AskVolume{i}" for i in range(1, 11)]
    cols += ["MidPrice"]

    missing_cols = [c for c in cols if c not in lob_info.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in lob_info: {missing_cols}")

    result = lob_info.loc[
        (lob_info["TransactDT_SEC"] == target_ts) &
        (lob_info[stock_col] == stock),
        cols,
    ].copy()

    return result


def get_l1_bid_anchor_by_time(lob_snap_path, target_time, selected_stocks=None):
    lob_snap = pd.read_csv(lob_snap_path)
    columns = list(lob_snap.columns) + ["MISC"]
    lob_info = pd.read_csv(
        lob_snap_path,
        header=None,
        names=columns,
        usecols=["SecurityID", "BidPrice1", "AskPrice1", "UpdateTime"],
    )
    lob_info = lob_info[1:].copy()
    lob_info["SecurityID"] = lob_info["SecurityID"].astype(str).str.zfill(6) + "_XSHE"
    if selected_stocks:
        lob_info = lob_info[lob_info["SecurityID"].isin(selected_stocks)].copy()

    lob_info["BidPrice1"] = pd.to_numeric(lob_info["BidPrice1"], errors="coerce")
    lob_info["AskPrice1"] = pd.to_numeric(lob_info["AskPrice1"], errors="coerce")
    lob_info = lob_info.dropna(subset=["BidPrice1", "AskPrice1"]).copy()
    lob_info["UpdateTime"] = lob_info["UpdateTime"].astype(str)
    lob_info = lob_info[lob_info["UpdateTime"].str.len() >= 8].copy()
    lob_info["TransactDT_SEC"] = pd.to_datetime(
        lob_info["UpdateTime"].str.slice(0, 8),
        format="%H:%M:%S",
        errors="coerce",
    ).dt.floor("S")
    lob_info = lob_info.dropna(subset=["TransactDT_SEC"]).copy()

    target_ts = pd.Timestamp(f"1900-01-01 {target_time}")
    anchors = lob_info.loc[lob_info["TransactDT_SEC"] == target_ts].copy()
    if anchors.empty:
        raise RuntimeError(f"No L1 snapshot found at {target_time} in {lob_snap_path}")

    anchors = (
        anchors.sort_values(["SecurityID", "UpdateTime"], kind="mergesort")
        .groupby("SecurityID", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    anchor_map = {}
    for row in anchors.itertuples(index=False):
        bid_price = float(row.BidPrice1)
        anchor_map[row.SecurityID] = {
            "anchor_time": target_time,
            "anchor_bid_price": bid_price,
            "anchor_bid_tick": int(round(bid_price / 0.01)),
            "anchor_ask_price": float(row.AskPrice1),
            "anchor_update_time": str(row.UpdateTime),
        }

    missing = sorted(set(selected_stocks or []) - set(anchor_map.keys()))
    if missing:
        raise RuntimeError(f"Missing 09:31 L1 anchors for stocks: {missing}")

    return anchor_map


def get_order_window_ending_at_second(processed_LOB_data, target_time, stock, order_num, stock_col="SecurityID"):
    if order_num <= 0:
        raise ValueError(f"order_num must be positive, got {order_num}")

    parsed = pd.to_datetime(target_time, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Could not parse target_time: {target_time}")

    target_ts = pd.Timestamp(
        year=1900,
        month=1,
        day=1,
        hour=parsed.hour,
        minute=parsed.minute,
        second=parsed.second,
    )

    required_cols = ["TransactDT_SEC", stock_col]
    missing_cols = [c for c in required_cols if c not in processed_LOB_data.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in processed_LOB_data: {missing_cols}")

    df_stock = processed_LOB_data.loc[processed_LOB_data[stock_col] == stock]
    if df_stock.empty:
        raise ValueError(f"No rows found for stock={stock}")

    matched_idx = df_stock.index[df_stock["TransactDT_SEC"] == target_ts]
    if len(matched_idx) == 0:
        raise ValueError(f"No rows found for stock={stock} at TransactDT_SEC={target_ts}")

    end_idx = matched_idx[-1]
    stock_indices = df_stock.index.to_list()
    end_pos = stock_indices.index(end_idx)

    start_pos = max(0, end_pos - order_num + 1)
    window_indices = stock_indices[start_pos:end_pos + 1]
    return df_stock.loc[window_indices].copy()


def apply_existing_bins(df: pd.DataFrame, field: str, bins) -> pd.Series:
    bins = np.asarray(bins, dtype=float)
    min_value = float(bins[0])

    bin_indices = pd.Series(
        np.digitize(df[field].astype(float).fillna(min_value - 1), bins, right=True) - 1,
        index=df.index,
        name=f"{field}_bin",
    )
    bin_indices = np.maximum(bin_indices, 0)
    bin_indices = np.minimum(bin_indices, len(bins) - 2)
    return pd.Series(bin_indices, index=df.index, name=f"{field}_bin").astype(np.int32)


def build_bin_converter_samp(
    df: pd.DataFrame,
    field: str,
    num_bins: int = 32,
    max_values: int = 1_000_000,
    return_bins: bool = True,
    return_distribution: bool = True,
):
    values = df[field].dropna().astype(float).values

    if len(values) == 0:
        raise ValueError(f"No non-null values found for field={field}")

    rng = np.random.default_rng(seed=42)
    if len(values) > max_values:
        values = rng.choice(values, size=max_values, replace=False)

    values.sort()
    value_freq = Counter(values)
    avg_bin_sample_count = len(values) / num_bins

    single_item_bins = set()
    for value, count in value_freq.items():
        if count > avg_bin_sample_count:
            single_item_bins.add(value)

    # Guard against degenerate cases where too many heavy hitters would consume
    # all available bins and break the edge construction logic.
    if len(single_item_bins) >= num_bins:
        ranked_singletons = sorted(
            single_item_bins,
            key=lambda value: (-int(value_freq[value]), float(value)),
        )
        single_item_bins = set(ranked_singletons[: max(0, num_bins - 1)])

    min_value = float(values.min())
    if min_value in single_item_bins:
        min_value = min_value - 1

    values_wo_single = [x for x in values if x not in single_item_bins]
    num_values = len(values_wo_single)
    # We already have one lower edge plus one edge per isolated singleton.
    # To reach exactly num_bins + 1 total edges, append only the remaining edge budget.
    available_bins = num_bins - len(single_item_bins)
    cur_index = 0

    bins = [min_value] + list(single_item_bins)

    while available_bins > 0 and cur_index < num_values:
        steps = (
            (num_values - cur_index) // (available_bins - 1)
            if available_bins > 1 else
            (num_values - cur_index)
        )

        start_value = values_wo_single[cur_index]

        for end_index in range(cur_index + steps, num_values):
            if values_wo_single[end_index] != start_value:
                break
        else:
            end_index = num_values - 1

        cur_index = end_index
        available_bins -= 1
        bins.append(values_wo_single[cur_index] if cur_index < num_values else values_wo_single[-1])

    bins = np.sort(np.asarray(bins, dtype=float))
    if len(bins) != num_bins + 1:
        raise AssertionError(f"Expected {num_bins + 1} bin edges, got {len(bins)}")

    filled = df[field].astype(float).fillna(min_value - 1)
    bin_indices = pd.Series(
        np.digitize(filled, bins, right=True) - 1,
        index=df.index,
        name=f"{field}_bin",
    )
    bin_indices = np.maximum(bin_indices, 0)
    bin_indices = np.minimum(bin_indices, len(bins) - 2)
    bin_indices = pd.Series(bin_indices, index=df.index, name=f"{field}_bin").astype(np.int32)

    bin_distribution_record = None
    if return_distribution:
        tmp = pd.DataFrame({
            "value": df[field].astype(float),
            "bin_idx": bin_indices,
        }).dropna(subset=["value"])

        bin_distribution_record = {}
        for b in range(num_bins):
            sub = tmp[tmp["bin_idx"] == b]["value"]
            vc = sub.value_counts(dropna=True, sort=False).sort_index()

            counts = vc.astype(int).tolist()
            unique_values = [float(x) for x in vc.index.tolist()]
            total = int(sum(counts))
            probs = [c / total for c in counts] if total > 0 else []

            bin_distribution_record[str(b)] = {
                "bin_left": float(bins[b]),
                "bin_right": float(bins[b + 1]),
                "total_count": total,
                "n_unique": len(unique_values),
                "unique_values": unique_values,
                "counts": counts,
                "probs": probs,
            }

    if return_bins:
        return bins, bin_indices, bin_distribution_record
    return None, bin_indices, bin_distribution_record


def make_order_token_id(
    df: pd.DataFrame,
    *,
    price_col: str = "price_bin",
    qty_col: str = "qty_bin",
    interval_col: str = "interval_bin",
    side_col: str = "Side",
    n_price: int = 32,
    n_qty: int = 32,
    n_interval: int = 16,
    n_side: int = 3,
    side_to_bin: Optional[dict] = None,
    out_col: str = "order_token",
    side_bin_col: str = "side_bin",
    valid_col: str = "tokenizable_event",
):
    if side_to_bin is None:
        side_to_bin = {49: 0, 50: 1, 99: 2}

    side_bin = df[side_col].map(side_to_bin)
    valid_mask = (
        df[price_col].notna()
        & df[qty_col].notna()
        & df[interval_col].notna()
        & side_bin.notna()
    )

    df[valid_col] = valid_mask.astype(bool)
    df[side_bin_col] = side_bin.astype("Int64")
    df[out_col] = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"), index=df.index)

    if not valid_mask.any():
        return df

    price = df.loc[valid_mask, price_col].astype(int).to_numpy()
    qty = df.loc[valid_mask, qty_col].astype(int).to_numpy()
    interval = df.loc[valid_mask, interval_col].astype(int).to_numpy()
    side = side_bin.loc[valid_mask].astype(int).to_numpy()

    assert price.min() >= 0 and price.max() < n_price
    assert qty.min() >= 0 and qty.max() < n_qty
    assert interval.min() >= 0 and interval.max() < n_interval
    assert side.min() >= 0 and side.max() < n_side

    token = (
        price * (n_qty * n_interval * n_side)
        + qty * (n_interval * n_side)
        + interval * n_side
        + side
    )
    df.loc[valid_mask, out_col] = pd.array(token.astype(np.int64), dtype="Int64")
    return df


def process_lob_data_real_flow(
    order_post_dir,
    lob_snap_dir,
    order_transac_dir,
    liquidity_mask_dir,
    selected_stocks,
    filter_bo,
    date_num_str,
    *,
    price_bin_num: Optional[int] = None,
    qty_bin_num: Optional[int] = None,
    interval_bin_num: Optional[int] = None,
    n_side: int = 3,
    side_to_bin: Optional[dict] = None,
    existing_bin_record_path: Optional[str] = None,
    return_bin_record: bool = False,
):
    import gc
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    print("Reading Order Post data...")
    col_names = [
        "ChannelNo", "ApplSeqNum", "MDStreamID", "SecurityID", "SecurityIDSource",
        "Price", "OrderQty", "Side", "TransactTime", "OrdType", "LocalTime", "SeqNo", "MISC"
    ]
    order_post = pd.read_csv(order_post_dir, header=None, names=col_names)
    print("Cleaning Order Post data...")
    order_post = order_post[1:]

    order_post["ApplSeqNum"] = pd.to_numeric(order_post["ApplSeqNum"], errors="coerce").astype(int)
    order_post["SecurityID"] = order_post["SecurityID"].astype(str).str.zfill(6) + "_XSHE"

    if len(selected_stocks) > 0:
        order_post = order_post[order_post["SecurityID"].isin(selected_stocks)]

    print("Reading Liquidity Mask data...")
    liquidity_mask = pd.read_csv(liquidity_mask_dir)
    liquidity_mask_slice = liquidity_mask[liquidity_mask["Unnamed: 0"] == int(date_num_str)].reset_index()

    print("Unpivoting Liquidity Mask data...")
    melted = liquidity_mask_slice.melt(
        id_vars=["index"],
        var_name="SecurityID",
        value_name="is_high_liquidity",
    )
    result_df = melted[["SecurityID", "is_high_liquidity"]][1:]

    print("Merging DataFrames...")
    merged_order_post = order_post.merge(result_df, how="left", on="SecurityID")
    print("Merge complete!")

    merged_order_post["TransactDT"] = pd.to_datetime(
        merged_order_post["TransactTime"].str.slice(0, 8), format="%H:%M:%S"
    )
    merged_order_post["TransactDT_MS"] = pd.to_datetime(
        merged_order_post["TransactTime"], format="%H:%M:%S.%f"
    )

    print("Filtering merged data for high liquidity...")
    merged_order_post_filtered = merged_order_post[merged_order_post["is_high_liquidity"] == 1]
    merged_order_post_filtered = merged_order_post_filtered.sort_values(
        ["SecurityID", "TransactDT_MS"], kind="mergesort"
    )

    merged_order_post_filtered["TransactDT_SEC"] = merged_order_post_filtered["TransactDT_MS"].dt.floor("S")

    if filter_bo:
        print("Filtering out pre-market data (before 09:30:00)...")
        merged_order_post_filtered = merged_order_post_filtered[
            merged_order_post_filtered["TransactDT_SEC"].dt.time > pd.to_datetime("09:30:00").time()
        ]

    int_cols = ["ChannelNo", "ApplSeqNum", "Side", "OrderQty", "OrdType"]
    for col in int_cols:
        merged_order_post_filtered[col] = pd.to_numeric(merged_order_post_filtered[col], errors="coerce")

    print("Filtering for specific OrdType...")
    merged_order_post_filtered = merged_order_post_filtered[merged_order_post_filtered["OrdType"] == 50]

    print("Reading LOB Snap data...")
    lob_snap = pd.read_csv(lob_snap_dir)
    columns = list(lob_snap.columns) + ["MISC"]
    lob_info = pd.read_csv(
        lob_snap_dir,
        header=None,
        names=columns,
        usecols=["SecurityID", "AskPrice1", "BidPrice1", "UpdateTime"],
    )
    lob_info = lob_info[1:]
    lob_info["AskPrice1"] = pd.to_numeric(lob_info["AskPrice1"], errors="coerce")
    lob_info["BidPrice1"] = pd.to_numeric(lob_info["BidPrice1"], errors="coerce")
    lob_info = lob_info.dropna(subset=["AskPrice1", "BidPrice1"])
    lob_info["MidPrice"] = (lob_info["AskPrice1"] + lob_info["BidPrice1"]) / 2
    lob_info["SecurityID"] = lob_info["SecurityID"].astype(str).str.zfill(6) + "_XSHE"

    print("Creating LOB dictionary...")
    lob_dict = lob_info[["UpdateTime", "SecurityID", "MidPrice"]].copy()
    lob_dict["UpdateTime"] = lob_dict["UpdateTime"].astype(str)
    lob_dict = lob_dict[lob_dict["UpdateTime"].str.len() >= 8]

    print("Converting UpdateTime to datetime...")
    lob_dict["TransactDT_SEC"] = pd.to_datetime(
        lob_dict["UpdateTime"].str.slice(0, 8), format="%H:%M:%S", errors="coerce"
    ).dt.floor("S")
    lob_dict = lob_dict.dropna(subset=["TransactDT_SEC", "MidPrice"])

    lob_key = lob_dict[["SecurityID", "TransactDT_SEC", "MidPrice"]].copy()
    lob_key = (
        lob_key.sort_values(["SecurityID", "TransactDT_SEC"], kind="mergesort")
        .groupby(["SecurityID", "TransactDT_SEC"], as_index=False)
        .tail(1)
    )
    lob_key = lob_key.sort_values(["TransactDT_SEC", "SecurityID"], kind="mergesort").reset_index(drop=True)

    print("Performing merge_asof operation on chunks...")
    chunk_size = 1_000_000
    out_dfs = []
    n = len(merged_order_post_filtered)

    for start in tqdm(range(0, n, chunk_size), desc="Asof-joining MidPrice chunks"):
        end = min(start + chunk_size, n)
        chunk = merged_order_post_filtered.iloc[start:end].copy()
        chunk = chunk.sort_values(["TransactDT_SEC", "SecurityID"], kind="mergesort").reset_index(drop=True)

        chunk = pd.merge_asof(
            chunk,
            lob_key,
            on="TransactDT_SEC",
            by="SecurityID",
            direction="backward",
            allow_exact_matches=True,
            tolerance=pd.Timedelta("5s"),
        )
        out_dfs.append(chunk)

    print("Concatenating final result...")
    final_result = pd.concat(out_dfs, ignore_index=True)

    del lob_dict, lob_info, lob_snap, lob_key, merged_order_post, merged_order_post_filtered, order_post, out_dfs
    gc.collect()

    final_result.dropna(subset=["MidPrice"], inplace=True)
    final_result["Price"] = final_result["Price"].astype(float)
    final_result["MidPrice"] = final_result["MidPrice"].astype(float)
    half_tick = 0.005
    final_result["Price_Mid_diff"] = np.rint(
        (final_result["Price"] - final_result["MidPrice"]) / half_tick
    ).astype(np.int32)

    order_transac_col_names = [
        "ChannelNo", "ApplSeqNum", "MDStreamID", "BidApplSeqNum", "OfferApplSeqNum",
        "SecurityID", "SecurityIDSource", "LastPx", "LastQty", "ExecType",
        "TransactTime", "LocalTime", "SeqNo", "MISC"
    ]
    print("Reading Order Transaction data...")
    order_transac = pd.read_csv(order_transac_dir, header=None, names=order_transac_col_names)
    order_transac = order_transac[1:]
    order_transac["SecurityID"] = order_transac["SecurityID"].astype(str).str.zfill(6) + "_XSHE"

    if len(selected_stocks) > 0:
        order_transac = order_transac[order_transac["SecurityID"].isin(selected_stocks)]

    for col in ["ExecType", "BidApplSeqNum", "OfferApplSeqNum", "ChannelNo"]:
        order_transac[col] = pd.to_numeric(order_transac[col], errors="coerce")

    # keep both cancel (52) and execution (70)
    order_transac = order_transac[order_transac["ExecType"].isin([52, 70])].copy()

    gc.collect()

    print("Expanding transaction rows for bid/offer order matches...")
    bid_transac = order_transac.loc[
        order_transac["BidApplSeqNum"].notna() & (order_transac["BidApplSeqNum"] != 0),
        ["ChannelNo", "SecurityID", "LastQty", "TransactTime", "ExecType", "BidApplSeqNum"],
    ].rename(columns={"BidApplSeqNum": "ApplSeqNum"})

    offer_transac = order_transac.loc[
        order_transac["OfferApplSeqNum"].notna() & (order_transac["OfferApplSeqNum"] != 0),
        ["ChannelNo", "SecurityID", "LastQty", "TransactTime", "ExecType", "OfferApplSeqNum"],
    ].rename(columns={"OfferApplSeqNum": "ApplSeqNum"})

    order_transac_expanded = pd.concat([bid_transac, offer_transac], ignore_index=True)
    order_transac_expanded["ApplSeqNum"] = pd.to_numeric(order_transac_expanded["ApplSeqNum"], errors="coerce")

    print("Matching transaction rows back to posted orders...")
    price_match_df = final_result[
        ["ChannelNo", "ApplSeqNum", "SecurityID", "Price", "MidPrice", "Price_Mid_diff", "Side"]
    ].copy()
    price_match_df = price_match_df.rename(columns={"Side": "OrigSide"})
    for col in ["ChannelNo", "ApplSeqNum", "OrigSide"]:
        price_match_df[col] = pd.to_numeric(price_match_df[col], errors="coerce")
    price_match_df["Price"] = pd.to_numeric(price_match_df["Price"], errors="coerce")

    order_transac_expanded = order_transac_expanded.merge(
        price_match_df,
        how="left",
        on=["ChannelNo", "ApplSeqNum", "SecurityID"],
    )
    order_transac_expanded.dropna(subset=["Price"], inplace=True)

    order_transac_expanded["TransactDT"] = pd.to_datetime(
        order_transac_expanded["TransactTime"].str.slice(0, 8), format="%H:%M:%S"
    )
    order_transac_expanded["TransactDT_MS"] = pd.to_datetime(
        order_transac_expanded["TransactTime"], format="%H:%M:%S.%f"
    )
    order_transac_expanded["TransactDT_SEC"] = order_transac_expanded["TransactDT_MS"].dt.floor("S")
    order_transac_expanded = order_transac_expanded.rename(columns={"LastQty": "OrderQty"})
    order_transac_expanded = order_transac_expanded.drop(columns=["TransactTime"])

    # overwrite Side to distinguish event types
    # 52 -> cancel -> 99
    # 70 -> execution -> 129
    order_transac_expanded["Side"] = np.where(
        order_transac_expanded["ExecType"] == 52,
        99,
        np.where(order_transac_expanded["ExecType"] == 70, 129, np.nan)
    )

    final_result_for_merge = final_result[
        [
            "ChannelNo", "ApplSeqNum", "SecurityID", "OrderQty", "Side",
            "TransactDT", "TransactDT_MS", "TransactDT_SEC", "Price", "MidPrice", "Price_Mid_diff"
        ]
    ].copy()
    final_result_for_merge["ExecType"] = np.nan
    final_result_for_merge["OrigSide"] = final_result_for_merge["Side"]

    order_transac_expanded = order_transac_expanded[
        [
            "ChannelNo", "ApplSeqNum", "SecurityID", "OrderQty", "Side",
            "TransactDT", "TransactDT_MS", "TransactDT_SEC", "Price", "MidPrice", "Price_Mid_diff",
            "ExecType", "OrigSide"
        ]
    ].copy()

    final_result_for_merge = pd.concat([final_result_for_merge, order_transac_expanded], ignore_index=True)

    final_result_for_merge = (
        final_result_for_merge
        .sort_values(["SecurityID", "TransactDT_MS", "ChannelNo", "ApplSeqNum"], kind="mergesort")
        .reset_index(drop=True)
    )

    final_result_for_merge = final_result_for_merge[
        (final_result_for_merge["TransactDT"].dt.time >= pd.to_datetime("09:25").time())
        & (final_result_for_merge["TransactDT"].dt.time <= pd.to_datetime("15:00").time())
    ].reset_index(drop=True)

    final_result_for_merge["interval_ms"] = (
        final_result_for_merge.groupby("SecurityID")["TransactDT_MS"]
        .diff()
        .dt.total_seconds() * 1000
    )
    final_result_for_merge["interval_ms"] = final_result_for_merge["interval_ms"].fillna(0).astype(int)

    final_result_for_merge = final_result_for_merge.dropna(subset=["Side"])

    bin_record = None
    if price_bin_num is not None and qty_bin_num is not None and interval_bin_num is not None:
        if side_to_bin is None:
            side_to_bin = {49: 0, 50: 1, 99: 2}

        fit_sides = set(int(k) for k in side_to_bin.keys())
        fit_df = final_result_for_merge[final_result_for_merge["Side"].isin(fit_sides)].copy()
        if fit_df.empty:
            raise RuntimeError("No rows available for training-compatible bin fitting in real-flow data.")

        bin_record = None
        if existing_bin_record_path:
            with open(existing_bin_record_path, "r", encoding="utf-8") as fh:
                bin_record = json.load(fh)

            final_result_for_merge["price_bin"] = apply_existing_bins(
                final_result_for_merge,
                "Price_Mid_diff",
                bin_record["price_mid_diff"]["bins"],
            )
            final_result_for_merge["qty_bin"] = apply_existing_bins(
                final_result_for_merge,
                "OrderQty",
                bin_record["order_qty"]["bins"],
            )
            final_result_for_merge["interval_bin"] = apply_existing_bins(
                final_result_for_merge,
                "interval_ms",
                bin_record["interval_ms"]["bins"],
            )
            bin_record = dict(bin_record)
            bin_record["realflow_binning_mode"] = "applied_existing_bins"
            bin_record["realflow_binning_source"] = existing_bin_record_path
            bin_record["realflow_fit_side_values"] = sorted(fit_sides)
        else:
            bin_record = {
                "half_tick": float(half_tick),
                "price_mid_diff": None,
                "order_qty": None,
                "interval_ms": None,
                "realflow_binning_mode": "fit_on_realflow",
                "realflow_fit_side_values": sorted(fit_sides),
            }

            price_bins, _, price_dist = build_bin_converter_samp(
                fit_df,
                "Price_Mid_diff",
                num_bins=price_bin_num,
                return_bins=True,
                return_distribution=True,
            )
            qty_bins, _, qty_dist = build_bin_converter_samp(
                fit_df,
                "OrderQty",
                num_bins=qty_bin_num,
                return_bins=True,
                return_distribution=True,
            )
            interval_bins, _, interval_dist = build_bin_converter_samp(
                fit_df,
                "interval_ms",
                num_bins=interval_bin_num,
                return_bins=True,
                return_distribution=True,
            )

            final_result_for_merge["price_bin"] = apply_existing_bins(final_result_for_merge, "Price_Mid_diff", price_bins)
            final_result_for_merge["qty_bin"] = apply_existing_bins(final_result_for_merge, "OrderQty", qty_bins)
            final_result_for_merge["interval_bin"] = apply_existing_bins(final_result_for_merge, "interval_ms", interval_bins)

            bin_record["price_mid_diff"] = {
                "field": "Price_Mid_diff",
                "num_bins": int(price_bin_num),
                "bins": [float(x) for x in np.asarray(price_bins).tolist()],
                "unit": "half_ticks",
                "notes": "Price_Mid_diff is rounded to integer half-ticks before binning.",
                "bin_value_distributions": price_dist,
            }
            bin_record["order_qty"] = {
                "field": "OrderQty",
                "num_bins": int(qty_bin_num),
                "bins": [float(x) for x in np.asarray(qty_bins).tolist()],
                "unit": "shares/contracts(?)",
                "notes": "Equal-frequency-ish binning with single-item heavy hitters isolated.",
                "bin_value_distributions": qty_dist,
            }
            bin_record["interval_ms"] = {
                "field": "interval_ms",
                "num_bins": int(interval_bin_num),
                "bins": [float(x) for x in np.asarray(interval_bins).tolist()],
                "unit": "milliseconds",
                "notes": "Computed on the unified real-flow stream within each SecurityID.",
                "bin_value_distributions": interval_dist,
            }

        final_result_for_merge = make_order_token_id(
            final_result_for_merge,
            price_col="price_bin",
            qty_col="qty_bin",
            interval_col="interval_bin",
            side_col="Side",
            n_price=int(price_bin_num),
            n_qty=int(qty_bin_num),
            n_interval=int(interval_bin_num),
            n_side=int(n_side),
            side_to_bin=side_to_bin,
            out_col="order_token",
            side_bin_col="side_bin",
            valid_col="tokenizable_event",
        )

    print("Real-flow processing complete!")
    if return_bin_record:
        return final_result_for_merge, bin_record
    return final_result_for_merge


def process_lob_data_real_flow_open_anchor(
    order_post_dir,
    lob_snap_dir,
    order_transac_dir,
    liquidity_mask_dir,
    selected_stocks,
    filter_bo,
    date_num_str,
    *,
    anchor_time: str = "09:31:00",
    price_bin_num: Optional[int] = None,
    qty_bin_num: Optional[int] = None,
    interval_bin_num: Optional[int] = None,
    n_side: int = 3,
    side_to_bin: Optional[dict] = None,
    return_bin_record: bool = False,
):
    import gc
    import numpy as np
    import pandas as pd

    print("Reading Order Post data...")
    col_names = [
        "ChannelNo", "ApplSeqNum", "MDStreamID", "SecurityID", "SecurityIDSource",
        "Price", "OrderQty", "Side", "TransactTime", "OrdType", "LocalTime", "SeqNo", "MISC"
    ]
    order_post = pd.read_csv(order_post_dir, header=None, names=col_names)
    print("Cleaning Order Post data...")
    order_post = order_post[1:]

    order_post["ApplSeqNum"] = pd.to_numeric(order_post["ApplSeqNum"], errors="coerce").astype(int)
    order_post["SecurityID"] = order_post["SecurityID"].astype(str).str.zfill(6) + "_XSHE"

    if len(selected_stocks) > 0:
        order_post = order_post[order_post["SecurityID"].isin(selected_stocks)]

    print("Reading Liquidity Mask data...")
    liquidity_mask = pd.read_csv(liquidity_mask_dir)
    liquidity_mask_slice = liquidity_mask[liquidity_mask["Unnamed: 0"] == int(date_num_str)].reset_index()

    print("Unpivoting Liquidity Mask data...")
    melted = liquidity_mask_slice.melt(
        id_vars=["index"],
        var_name="SecurityID",
        value_name="is_high_liquidity",
    )
    result_df = melted[["SecurityID", "is_high_liquidity"]][1:]

    print("Merging DataFrames...")
    merged_order_post = order_post.merge(result_df, how="left", on="SecurityID")
    print("Merge complete!")

    merged_order_post["TransactDT"] = pd.to_datetime(
        merged_order_post["TransactTime"].str.slice(0, 8), format="%H:%M:%S"
    )
    merged_order_post["TransactDT_MS"] = pd.to_datetime(
        merged_order_post["TransactTime"], format="%H:%M:%S.%f"
    )

    print("Filtering merged data for high liquidity...")
    merged_order_post_filtered = merged_order_post[merged_order_post["is_high_liquidity"] == 1]
    merged_order_post_filtered = merged_order_post_filtered.sort_values(
        ["SecurityID", "TransactDT_MS"], kind="mergesort"
    )

    merged_order_post_filtered["TransactDT_SEC"] = merged_order_post_filtered["TransactDT_MS"].dt.floor("S")

    if filter_bo:
        print("Filtering out pre-market data (before 09:30:00)...")
        merged_order_post_filtered = merged_order_post_filtered[
            merged_order_post_filtered["TransactDT_SEC"].dt.time > pd.to_datetime("09:30:00").time()
        ]

    int_cols = ["ChannelNo", "ApplSeqNum", "Side", "OrderQty", "OrdType"]
    for col in int_cols:
        merged_order_post_filtered[col] = pd.to_numeric(merged_order_post_filtered[col], errors="coerce")

    print("Filtering for specific OrdType...")
    merged_order_post_filtered = merged_order_post_filtered[merged_order_post_filtered["OrdType"] == 50].copy()
    merged_order_post_filtered["Price"] = pd.to_numeric(merged_order_post_filtered["Price"], errors="coerce")
    merged_order_post_filtered = merged_order_post_filtered.dropna(subset=["Price"]).copy()

    print(f"Looking up fixed L1 bid anchors at {anchor_time}...")
    anchor_map = get_l1_bid_anchor_by_time(lob_snap_dir, anchor_time, selected_stocks=selected_stocks)

    final_result = merged_order_post_filtered.copy()
    final_result["OpenAnchorBidPrice"] = final_result["SecurityID"].map(
        {k: v["anchor_bid_price"] for k, v in anchor_map.items()}
    )
    final_result["OpenAnchorBidTick"] = final_result["SecurityID"].map(
        {k: v["anchor_bid_tick"] for k, v in anchor_map.items()}
    )
    final_result = final_result.dropna(subset=["OpenAnchorBidPrice", "OpenAnchorBidTick"]).copy()
    final_result["Price_OpenBid_diff"] = np.rint(
        (final_result["Price"].astype(float) - final_result["OpenAnchorBidPrice"].astype(float)) / 0.01
    ).astype(np.int32)

    del merged_order_post, merged_order_post_filtered, order_post
    gc.collect()

    order_transac_col_names = [
        "ChannelNo", "ApplSeqNum", "MDStreamID", "BidApplSeqNum", "OfferApplSeqNum",
        "SecurityID", "SecurityIDSource", "LastPx", "LastQty", "ExecType",
        "TransactTime", "LocalTime", "SeqNo", "MISC"
    ]
    print("Reading Order Transaction data...")
    order_transac = pd.read_csv(order_transac_dir, header=None, names=order_transac_col_names)
    order_transac = order_transac[1:]
    order_transac["SecurityID"] = order_transac["SecurityID"].astype(str).str.zfill(6) + "_XSHE"

    if len(selected_stocks) > 0:
        order_transac = order_transac[order_transac["SecurityID"].isin(selected_stocks)]

    for col in ["ExecType", "BidApplSeqNum", "OfferApplSeqNum", "ChannelNo"]:
        order_transac[col] = pd.to_numeric(order_transac[col], errors="coerce")

    order_transac = order_transac[order_transac["ExecType"].isin([52, 70])].copy()
    gc.collect()

    print("Expanding transaction rows for bid/offer order matches...")
    bid_transac = order_transac.loc[
        order_transac["BidApplSeqNum"].notna() & (order_transac["BidApplSeqNum"] != 0),
        ["ChannelNo", "SecurityID", "LastQty", "TransactTime", "ExecType", "BidApplSeqNum"],
    ].rename(columns={"BidApplSeqNum": "ApplSeqNum"})
    offer_transac = order_transac.loc[
        order_transac["OfferApplSeqNum"].notna() & (order_transac["OfferApplSeqNum"] != 0),
        ["ChannelNo", "SecurityID", "LastQty", "TransactTime", "ExecType", "OfferApplSeqNum"],
    ].rename(columns={"OfferApplSeqNum": "ApplSeqNum"})

    order_transac_expanded = pd.concat([bid_transac, offer_transac], ignore_index=True)
    order_transac_expanded["ApplSeqNum"] = pd.to_numeric(order_transac_expanded["ApplSeqNum"], errors="coerce")

    print("Matching transaction rows back to posted orders...")
    price_match_df = final_result[
        ["ChannelNo", "ApplSeqNum", "SecurityID", "Price", "Price_OpenBid_diff", "OpenAnchorBidPrice", "OpenAnchorBidTick", "Side"]
    ].copy()
    price_match_df = price_match_df.rename(columns={"Side": "OrigSide"})
    for col in ["ChannelNo", "ApplSeqNum", "OrigSide"]:
        price_match_df[col] = pd.to_numeric(price_match_df[col], errors="coerce")
    price_match_df["Price"] = pd.to_numeric(price_match_df["Price"], errors="coerce")

    order_transac_expanded = order_transac_expanded.merge(
        price_match_df,
        how="left",
        on=["ChannelNo", "ApplSeqNum", "SecurityID"],
    )
    order_transac_expanded.dropna(subset=["Price"], inplace=True)

    order_transac_expanded["TransactDT"] = pd.to_datetime(
        order_transac_expanded["TransactTime"].str.slice(0, 8), format="%H:%M:%S"
    )
    order_transac_expanded["TransactDT_MS"] = pd.to_datetime(
        order_transac_expanded["TransactTime"], format="%H:%M:%S.%f"
    )
    order_transac_expanded["TransactDT_SEC"] = order_transac_expanded["TransactDT_MS"].dt.floor("S")
    order_transac_expanded = order_transac_expanded.rename(columns={"LastQty": "OrderQty"})
    order_transac_expanded = order_transac_expanded.drop(columns=["TransactTime"])

    order_transac_expanded["Side"] = np.where(
        order_transac_expanded["ExecType"] == 52,
        99,
        np.where(order_transac_expanded["ExecType"] == 70, 129, np.nan),
    )

    final_result_for_merge = final_result[
        [
            "ChannelNo", "ApplSeqNum", "SecurityID", "OrderQty", "Side",
            "TransactDT", "TransactDT_MS", "TransactDT_SEC", "Price", "Price_OpenBid_diff",
            "OpenAnchorBidPrice", "OpenAnchorBidTick"
        ]
    ].copy()
    final_result_for_merge["ExecType"] = np.nan
    final_result_for_merge["OrigSide"] = final_result_for_merge["Side"]

    order_transac_expanded = order_transac_expanded[
        [
            "ChannelNo", "ApplSeqNum", "SecurityID", "OrderQty", "Side",
            "TransactDT", "TransactDT_MS", "TransactDT_SEC", "Price", "Price_OpenBid_diff",
            "OpenAnchorBidPrice", "OpenAnchorBidTick", "ExecType", "OrigSide"
        ]
    ].copy()

    final_result_for_merge = pd.concat([final_result_for_merge, order_transac_expanded], ignore_index=True)
    final_result_for_merge = (
        final_result_for_merge
        .sort_values(["SecurityID", "TransactDT_MS", "ChannelNo", "ApplSeqNum"], kind="mergesort")
        .reset_index(drop=True)
    )
    final_result_for_merge = final_result_for_merge[
        (final_result_for_merge["TransactDT"].dt.time >= pd.to_datetime("09:25").time())
        & (final_result_for_merge["TransactDT"].dt.time <= pd.to_datetime("15:00").time())
    ].reset_index(drop=True)

    final_result_for_merge["interval_ms"] = (
        final_result_for_merge.groupby("SecurityID")["TransactDT_MS"]
        .diff()
        .dt.total_seconds() * 1000
    )
    final_result_for_merge["interval_ms"] = final_result_for_merge["interval_ms"].fillna(0).astype(int)
    final_result_for_merge = final_result_for_merge.dropna(subset=["Side"]) 

    bin_record = None
    if price_bin_num is not None and qty_bin_num is not None and interval_bin_num is not None:
        if side_to_bin is None:
            side_to_bin = {49: 0, 50: 1, 99: 2}

        fit_sides = set(int(k) for k in side_to_bin.keys())
        fit_df = final_result_for_merge[final_result_for_merge["Side"].isin(fit_sides)].copy()
        if fit_df.empty:
            raise RuntimeError("No rows available for open-anchor bin fitting in real-flow data.")

        price_bins, _, price_dist = build_bin_converter_samp(
            fit_df,
            "Price_OpenBid_diff",
            num_bins=price_bin_num,
            return_bins=True,
            return_distribution=True,
        )
        qty_bins, _, qty_dist = build_bin_converter_samp(
            fit_df,
            "OrderQty",
            num_bins=qty_bin_num,
            return_bins=True,
            return_distribution=True,
        )
        interval_bins, _, interval_dist = build_bin_converter_samp(
            fit_df,
            "interval_ms",
            num_bins=interval_bin_num,
            return_bins=True,
            return_distribution=True,
        )

        final_result_for_merge["price_bin"] = apply_existing_bins(final_result_for_merge, "Price_OpenBid_diff", price_bins)
        final_result_for_merge["qty_bin"] = apply_existing_bins(final_result_for_merge, "OrderQty", qty_bins)
        final_result_for_merge["interval_bin"] = apply_existing_bins(final_result_for_merge, "interval_ms", interval_bins)

        anchor_metadata = {
            stock: {
                "anchor_time": meta["anchor_time"],
                "anchor_bid_price": float(meta["anchor_bid_price"]),
                "anchor_bid_tick": int(meta["anchor_bid_tick"]),
                "anchor_ask_price": float(meta["anchor_ask_price"]),
                "anchor_update_time": meta["anchor_update_time"],
            }
            for stock, meta in anchor_map.items()
        }

        bin_record = {
            "tick_size": 0.01,
            "price_anchor_mode": "market_open_best_bid_fixed",
            "price_anchor_time": anchor_time,
            "price_anchor_by_stock": anchor_metadata,
            "price_mid_diff": {
                "field": "Price_OpenBid_diff",
                "num_bins": int(price_bin_num),
                "bins": [float(x) for x in np.asarray(price_bins).tolist()],
                "unit": "full_ticks_from_open_bid_anchor",
                "notes": "Price_OpenBid_diff is rounded to integer full ticks from the fixed 09:31:00 L1 bid anchor.",
                "anchor_mode": "market_open_best_bid_fixed",
                "bin_value_distributions": price_dist,
            },
            "order_qty": {
                "field": "OrderQty",
                "num_bins": int(qty_bin_num),
                "bins": [float(x) for x in np.asarray(qty_bins).tolist()],
                "unit": "shares/contracts(?)",
                "notes": "Equal-frequency-ish binning with single-item heavy hitters isolated.",
                "bin_value_distributions": qty_dist,
            },
            "interval_ms": {
                "field": "interval_ms",
                "num_bins": int(interval_bin_num),
                "bins": [float(x) for x in np.asarray(interval_bins).tolist()],
                "unit": "milliseconds",
                "notes": "Computed on the unified real-flow stream within each SecurityID.",
                "bin_value_distributions": interval_dist,
            },
            "realflow_binning_mode": "fit_on_realflow_open_anchor",
            "realflow_fit_side_values": sorted(fit_sides),
        }

        final_result_for_merge = make_order_token_id(
            final_result_for_merge,
            price_col="price_bin",
            qty_col="qty_bin",
            interval_col="interval_bin",
            side_col="Side",
            n_price=int(price_bin_num),
            n_qty=int(qty_bin_num),
            n_interval=int(interval_bin_num),
            n_side=int(n_side),
            side_to_bin=side_to_bin,
            out_col="order_token",
            side_bin_col="side_bin",
            valid_col="tokenizable_event",
        )

    print("Real-flow open-anchor processing complete!")
    if return_bin_record:
        return final_result_for_merge, bin_record
    return final_result_for_merge


def process_lob_data_real_flow_open_anchor_txn_complete(
    order_post_dir,
    lob_snap_dir,
    order_transac_dir,
    liquidity_mask_dir,
    selected_stocks,
    filter_bo,
    date_num_str,
    *,
    anchor_time: str = "09:31:00",
    price_bin_num: Optional[int] = None,
    qty_bin_num: Optional[int] = None,
    interval_bin_num: Optional[int] = None,
    n_side: int = 5,
    side_to_bin: Optional[dict] = None,
    return_bin_record: bool = False,
    split_cancel_sides: bool = False,
    external_fit_dataframe=None,
):
    import gc
    import numpy as np
    import pandas as pd

    print("Reading Order Post data...")
    col_names = [
        "ChannelNo", "ApplSeqNum", "MDStreamID", "SecurityID", "SecurityIDSource",
        "Price", "OrderQty", "Side", "TransactTime", "OrdType", "LocalTime", "SeqNo", "MISC"
    ]
    order_post = pd.read_csv(order_post_dir, header=None, names=col_names)
    print("Cleaning Order Post data...")
    order_post = order_post[1:]

    order_post["ApplSeqNum"] = pd.to_numeric(order_post["ApplSeqNum"], errors="coerce").astype(int)
    order_post["SecurityID"] = order_post["SecurityID"].astype(str).str.zfill(6) + "_XSHE"

    if len(selected_stocks) > 0:
        order_post = order_post[order_post["SecurityID"].isin(selected_stocks)]

    print("Reading Liquidity Mask data...")
    liquidity_mask = pd.read_csv(liquidity_mask_dir)
    liquidity_mask_slice = liquidity_mask[liquidity_mask["Unnamed: 0"] == int(date_num_str)].reset_index()

    print("Unpivoting Liquidity Mask data...")
    melted = liquidity_mask_slice.melt(
        id_vars=["index"],
        var_name="SecurityID",
        value_name="is_high_liquidity",
    )
    result_df = melted[["SecurityID", "is_high_liquidity"]][1:]

    print("Merging DataFrames...")
    merged_order_post = order_post.merge(result_df, how="left", on="SecurityID")
    print("Merge complete!")

    merged_order_post["TransactDT"] = pd.to_datetime(
        merged_order_post["TransactTime"].str.slice(0, 8), format="%H:%M:%S"
    )
    merged_order_post["TransactDT_MS"] = pd.to_datetime(
        merged_order_post["TransactTime"], format="%H:%M:%S.%f"
    )

    print("Filtering merged data for high liquidity...")
    merged_order_post_filtered = merged_order_post[merged_order_post["is_high_liquidity"] == 1]
    merged_order_post_filtered = merged_order_post_filtered.sort_values(
        ["SecurityID", "TransactDT_MS"], kind="mergesort"
    )

    merged_order_post_filtered["TransactDT_SEC"] = merged_order_post_filtered["TransactDT_MS"].dt.floor("S")

    if filter_bo:
        print("Filtering out pre-market data (before 09:30:00)...")
        merged_order_post_filtered = merged_order_post_filtered[
            merged_order_post_filtered["TransactDT_SEC"].dt.time > pd.to_datetime("09:30:00").time()
        ]

    int_cols = ["ChannelNo", "ApplSeqNum", "Side", "OrderQty", "OrdType"]
    for col in int_cols:
        merged_order_post_filtered[col] = pd.to_numeric(merged_order_post_filtered[col], errors="coerce")

    print("Filtering for specific OrdType...")
    merged_order_post_filtered = merged_order_post_filtered[merged_order_post_filtered["OrdType"] == 50].copy()
    merged_order_post_filtered["Price"] = pd.to_numeric(merged_order_post_filtered["Price"], errors="coerce")
    merged_order_post_filtered = merged_order_post_filtered.dropna(subset=["Price"]).copy()

    print(f"Looking up fixed L1 bid anchors at {anchor_time}...")
    anchor_map = get_l1_bid_anchor_by_time(lob_snap_dir, anchor_time, selected_stocks=selected_stocks)

    final_result = merged_order_post_filtered.copy()
    final_result["OpenAnchorBidPrice"] = final_result["SecurityID"].map(
        {k: v["anchor_bid_price"] for k, v in anchor_map.items()}
    )
    final_result["OpenAnchorBidTick"] = final_result["SecurityID"].map(
        {k: v["anchor_bid_tick"] for k, v in anchor_map.items()}
    )
    final_result = final_result.dropna(subset=["OpenAnchorBidPrice", "OpenAnchorBidTick"]).copy()
    final_result["Price_OpenBid_diff"] = np.rint(
        (final_result["Price"].astype(float) - final_result["OpenAnchorBidPrice"].astype(float)) / 0.01
    ).astype(np.int32)

    del merged_order_post, merged_order_post_filtered, order_post
    gc.collect()

    order_transac_col_names = [
        "ChannelNo", "ApplSeqNum", "MDStreamID", "BidApplSeqNum", "OfferApplSeqNum",
        "SecurityID", "SecurityIDSource", "LastPx", "LastQty", "ExecType",
        "TransactTime", "LocalTime", "SeqNo", "MISC"
    ]
    print("Reading Order Transaction data...")
    order_transac = pd.read_csv(order_transac_dir, header=None, names=order_transac_col_names)
    order_transac = order_transac[1:]
    order_transac["SecurityID"] = order_transac["SecurityID"].astype(str).str.zfill(6) + "_XSHE"

    if len(selected_stocks) > 0:
        order_transac = order_transac[order_transac["SecurityID"].isin(selected_stocks)]

    for col in ["ExecType", "BidApplSeqNum", "OfferApplSeqNum", "ChannelNo"]:
        order_transac[col] = pd.to_numeric(order_transac[col], errors="coerce")

    order_transac = order_transac[order_transac["ExecType"].isin([52, 70])].copy()
    gc.collect()

    print("Expanding transaction rows for bid/offer order matches...")
    bid_transac = order_transac.loc[
        order_transac["BidApplSeqNum"].notna() & (order_transac["BidApplSeqNum"] != 0),
        ["ChannelNo", "SecurityID", "LastQty", "TransactTime", "ExecType", "BidApplSeqNum"],
    ].rename(columns={"BidApplSeqNum": "ApplSeqNum"})
    offer_transac = order_transac.loc[
        order_transac["OfferApplSeqNum"].notna() & (order_transac["OfferApplSeqNum"] != 0),
        ["ChannelNo", "SecurityID", "LastQty", "TransactTime", "ExecType", "OfferApplSeqNum"],
    ].rename(columns={"OfferApplSeqNum": "ApplSeqNum"})

    order_transac_expanded = pd.concat([bid_transac, offer_transac], ignore_index=True)
    order_transac_expanded["ApplSeqNum"] = pd.to_numeric(order_transac_expanded["ApplSeqNum"], errors="coerce")

    print("Matching transaction rows back to posted orders...")
    price_match_df = final_result[
        ["ChannelNo", "ApplSeqNum", "SecurityID", "Price", "Price_OpenBid_diff", "OpenAnchorBidPrice", "OpenAnchorBidTick", "Side"]
    ].copy()
    price_match_df = price_match_df.rename(columns={"Side": "OrigSide"})
    for col in ["ChannelNo", "ApplSeqNum", "OrigSide"]:
        price_match_df[col] = pd.to_numeric(price_match_df[col], errors="coerce")
    price_match_df["Price"] = pd.to_numeric(price_match_df["Price"], errors="coerce")

    order_transac_expanded = order_transac_expanded.merge(
        price_match_df,
        how="left",
        on=["ChannelNo", "ApplSeqNum", "SecurityID"],
    )
    order_transac_expanded.dropna(subset=["Price"], inplace=True)

    order_transac_expanded["TransactDT"] = pd.to_datetime(
        order_transac_expanded["TransactTime"].str.slice(0, 8), format="%H:%M:%S"
    )
    order_transac_expanded["TransactDT_MS"] = pd.to_datetime(
        order_transac_expanded["TransactTime"], format="%H:%M:%S.%f"
    )
    order_transac_expanded["TransactDT_SEC"] = order_transac_expanded["TransactDT_MS"].dt.floor("S")
    order_transac_expanded = order_transac_expanded.rename(columns={"LastQty": "OrderQty"})
    order_transac_expanded = order_transac_expanded.drop(columns=["TransactTime"])

    order_transac_expanded["EventSemantic"] = np.where(
        order_transac_expanded["ExecType"] == 52,
        "cancel",
        np.where(order_transac_expanded["ExecType"] == 70, "transaction_complete", None),
    )
    order_transac_expanded["Side"] = np.where(
        order_transac_expanded["ExecType"] == 52,
        99,
        np.where(
            order_transac_expanded["OrigSide"] == 49,
            129,
            np.where(order_transac_expanded["OrigSide"] == 50, 130, np.nan),
        ),
    )

    immediate_exec = order_transac_expanded[
        order_transac_expanded["ExecType"] == 70
    ][["ChannelNo", "ApplSeqNum", "SecurityID", "TransactDT_MS", "OrderQty"]].copy()
    immediate_exec = (
        immediate_exec
        .groupby(["ChannelNo", "ApplSeqNum", "SecurityID", "TransactDT_MS"], as_index=False)["OrderQty"]
        .sum()
        .rename(columns={"OrderQty": "ImmediateExecQty"})
    )

    final_result["PostQtyOriginal"] = final_result["OrderQty"].astype(float)
    final_result = final_result.merge(
        immediate_exec,
        how="left",
        on=["ChannelNo", "ApplSeqNum", "SecurityID", "TransactDT_MS"],
    )
    final_result["ImmediateExecQty"] = final_result["ImmediateExecQty"].fillna(0)
    final_result["ImmediateExecQty"] = np.minimum(
        final_result["ImmediateExecQty"].astype(float),
        final_result["PostQtyOriginal"].astype(float),
    )
    final_result["ResidualPostQty"] = final_result["PostQtyOriginal"] - final_result["ImmediateExecQty"]
    final_result["ImmediateAggressivePost"] = final_result["ImmediateExecQty"] > 0
    final_result["PassiveResidualPosted"] = final_result["ResidualPostQty"] > 0
    final_result.loc[final_result["PassiveResidualPosted"], "OrderQty"] = final_result.loc[
        final_result["PassiveResidualPosted"], "ResidualPostQty"
    ].astype(int)
    final_result["EventSemantic"] = "post"

    post_rows = final_result[final_result["PassiveResidualPosted"]].copy()
    post_rows["ExecType"] = np.nan
    post_rows["OrigSide"] = post_rows["Side"]

    final_result_for_merge = post_rows[
        [
            "ChannelNo", "ApplSeqNum", "SecurityID", "OrderQty", "Side",
            "TransactDT", "TransactDT_MS", "TransactDT_SEC", "Price", "Price_OpenBid_diff",
            "OpenAnchorBidPrice", "OpenAnchorBidTick", "ExecType", "OrigSide",
            "EventSemantic", "PostQtyOriginal", "ImmediateExecQty", "ResidualPostQty",
            "ImmediateAggressivePost", "PassiveResidualPosted"
        ]
    ].copy()

    order_transac_expanded["PostQtyOriginal"] = np.nan
    order_transac_expanded["ImmediateExecQty"] = np.nan
    order_transac_expanded["ResidualPostQty"] = np.nan
    order_transac_expanded["ImmediateAggressivePost"] = False
    order_transac_expanded["PassiveResidualPosted"] = False
    order_transac_expanded = order_transac_expanded[
        [
            "ChannelNo", "ApplSeqNum", "SecurityID", "OrderQty", "Side",
            "TransactDT", "TransactDT_MS", "TransactDT_SEC", "Price", "Price_OpenBid_diff",
            "OpenAnchorBidPrice", "OpenAnchorBidTick", "ExecType", "OrigSide",
            "EventSemantic", "PostQtyOriginal", "ImmediateExecQty", "ResidualPostQty",
            "ImmediateAggressivePost", "PassiveResidualPosted"
        ]
    ].copy()

    final_result_for_merge = pd.concat([final_result_for_merge, order_transac_expanded], ignore_index=True)
    final_result_for_merge = (
        final_result_for_merge
        .sort_values(["SecurityID", "TransactDT_MS", "ChannelNo", "ApplSeqNum"], kind="mergesort")
        .reset_index(drop=True)
    )
    final_result_for_merge = final_result_for_merge[
        (final_result_for_merge["TransactDT"].dt.time >= pd.to_datetime("09:25").time())
        & (final_result_for_merge["TransactDT"].dt.time <= pd.to_datetime("15:00").time())
    ].reset_index(drop=True)

    final_result_for_merge["interval_ms"] = (
        final_result_for_merge.groupby("SecurityID")["TransactDT_MS"]
        .diff()
        .dt.total_seconds() * 1000
    )
    final_result_for_merge["interval_ms"] = final_result_for_merge["interval_ms"].fillna(0).astype(int)
    final_result_for_merge = final_result_for_merge.dropna(subset=["Side"])

    if split_cancel_sides:
        if int(n_side) != 6:
            raise ValueError("split_cancel_sides=True requires n_side=6")
        final_result_for_merge = final_result_for_merge.copy()
        side_series = pd.to_numeric(final_result_for_merge["Side"], errors="coerce")
        orig_series = pd.to_numeric(final_result_for_merge["OrigSide"], errors="coerce")
        m_cancel = side_series == 99
        final_result_for_merge.loc[m_cancel & (orig_series == 49), "Side"] = 97
        final_result_for_merge.loc[m_cancel & (orig_series == 50), "Side"] = 98

    bin_record = None
    if price_bin_num is not None and qty_bin_num is not None and interval_bin_num is not None:
        if side_to_bin is None:
            if split_cancel_sides:
                side_to_bin = {49: 0, 50: 1, 97: 2, 98: 3, 129: 4, 130: 5}
            else:
                side_to_bin = {49: 0, 50: 1, 99: 2, 129: 3, 130: 4}

        fit_sides = set(int(k) for k in side_to_bin.keys())
        if external_fit_dataframe is not None:
            fit_df = external_fit_dataframe[external_fit_dataframe["Side"].isin(fit_sides)].copy()
            _req = {"Price_OpenBid_diff", "OrderQty", "interval_ms", "Side"}
            _miss = _req - set(external_fit_dataframe.columns)
            if _miss:
                raise ValueError(f"external_fit_dataframe missing columns {_miss}")
        else:
            fit_df = final_result_for_merge[final_result_for_merge["Side"].isin(fit_sides)].copy()
        if fit_df.empty:
            raise RuntimeError("No rows available for open-anchor txn-complete bin fitting in real-flow data.")

        price_bins, _, price_dist = build_bin_converter_samp(
            fit_df,
            "Price_OpenBid_diff",
            num_bins=price_bin_num,
            return_bins=True,
            return_distribution=True,
        )
        qty_bins, _, qty_dist = build_bin_converter_samp(
            fit_df,
            "OrderQty",
            num_bins=qty_bin_num,
            return_bins=True,
            return_distribution=True,
        )
        interval_bins, _, interval_dist = build_bin_converter_samp(
            fit_df,
            "interval_ms",
            num_bins=interval_bin_num,
            return_bins=True,
            return_distribution=True,
        )

        final_result_for_merge["price_bin"] = apply_existing_bins(final_result_for_merge, "Price_OpenBid_diff", price_bins)
        final_result_for_merge["qty_bin"] = apply_existing_bins(final_result_for_merge, "OrderQty", qty_bins)
        final_result_for_merge["interval_bin"] = apply_existing_bins(final_result_for_merge, "interval_ms", interval_bins)

        anchor_metadata = {
            stock: {
                "anchor_time": meta["anchor_time"],
                "anchor_bid_price": float(meta["anchor_bid_price"]),
                "anchor_bid_tick": int(meta["anchor_bid_tick"]),
                "anchor_ask_price": float(meta["anchor_ask_price"]),
                "anchor_update_time": meta["anchor_update_time"],
            }
            for stock, meta in anchor_map.items()
        }

        immediate_aggressive_stats = (
            final_result[["SecurityID", "ImmediateAggressivePost", "PassiveResidualPosted"]]
            .groupby("SecurityID")
            .agg(
                immediate_aggressive_posts=("ImmediateAggressivePost", "sum"),
                residual_post_rows=("PassiveResidualPosted", "sum"),
            )
            .reset_index()
        )

        bin_record = {
            "tick_size": 0.01,
            "price_anchor_mode": "market_open_best_bid_fixed",
            "price_anchor_time": anchor_time,
            "price_anchor_by_stock": anchor_metadata,
            "price_mid_diff": {
                "field": "Price_OpenBid_diff",
                "num_bins": int(price_bin_num),
                "bins": [float(x) for x in np.asarray(price_bins).tolist()],
                "unit": "full_ticks_from_open_bid_anchor",
                "notes": "Price_OpenBid_diff is rounded to integer full ticks from the fixed 09:31:00 L1 bid anchor.",
                "anchor_mode": "market_open_best_bid_fixed",
                "bin_value_distributions": price_dist,
            },
            "order_qty": {
                "field": "OrderQty",
                "num_bins": int(qty_bin_num),
                "bins": [float(x) for x in np.asarray(qty_bins).tolist()],
                "unit": "shares/contracts(?)",
                "notes": "Equal-frequency-ish binning with single-item heavy hitters isolated.",
                "bin_value_distributions": qty_dist,
            },
            "interval_ms": {
                "field": "interval_ms",
                "num_bins": int(interval_bin_num),
                "bins": [float(x) for x in np.asarray(interval_bins).tolist()],
                "unit": "milliseconds",
                "notes": "Computed on the unified real-flow stream within each SecurityID.",
                "bin_value_distributions": interval_dist,
            },
            "realflow_binning_mode": "fit_on_realflow_open_anchor_txn_complete",
            "realflow_fit_side_values": sorted(fit_sides),
            "split_cancel_sides": bool(split_cancel_sides),
            "token_side_mapping": (
                {
                    "49": "bid_post",
                    "50": "ask_post",
                    "97": "cancel_resting_bid",
                    "98": "cancel_resting_ask",
                    "129": "transaction_complete_resting_bid",
                    "130": "transaction_complete_resting_ask",
                }
                if split_cancel_sides
                else {
                    "49": "bid_post",
                    "50": "ask_post",
                    "99": "cancel",
                    "129": "transaction_complete_resting_bid",
                    "130": "transaction_complete_resting_ask",
                }
            ),
            "transaction_complete_mode": "split_by_resting_side",
            "immediate_aggressive_order_rule": (
                "Orders with execution rows sharing the same ChannelNo/ApplSeqNum/SecurityID/TransactDT_MS have "
                "their immediate execution quantity removed from the standalone post token. Only residual passive "
                "quantity remains as a post token; the removed quantity is represented by transaction_complete rows."
            ),
            "immediate_aggressive_stats_by_stock": immediate_aggressive_stats.to_dict(orient="records"),
            "pooled_bin_fit_from_external_frame": bool(external_fit_dataframe is not None),
            "pooled_bin_fit_row_count": int(len(fit_df)),
        }

        final_result_for_merge = make_order_token_id(
            final_result_for_merge,
            price_col="price_bin",
            qty_col="qty_bin",
            interval_col="interval_bin",
            side_col="Side",
            n_price=int(price_bin_num),
            n_qty=int(qty_bin_num),
            n_interval=int(interval_bin_num),
            n_side=int(n_side),
            side_to_bin=side_to_bin,
            out_col="order_token",
            side_bin_col="side_bin",
            valid_col="tokenizable_event",
        )

    print("Real-flow open-anchor txn-complete processing complete!")
    if return_bin_record:
        return final_result_for_merge, bin_record
    return final_result_for_merge

# ============================================================
# VERBOSE DEFAULTS
# ============================================================

VERBOSE_STEPS_DEFAULT = 100
VERBOSE_DEPTH_DEFAULT = 10
PRINT_WARMSTART_DEFAULT = True


# ============================================================
# TOKEN / BIN HELPERS
# ============================================================

def decode_order_token(token: int, n_price: int, n_qty: int, n_interval: int, n_side: int):
    token = int(token)
    side = token % n_side
    token //= n_side

    interval = token % n_interval
    token //= n_interval

    qty = token % n_qty
    token //= n_qty

    price = token
    return int(price), int(qty), int(interval), int(side)


def load_bin_record(path: str):
    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)

    for k in ["price_mid_diff", "order_qty", "interval_ms"]:
        if k not in rec or "bins" not in rec[k]:
            raise KeyError(f"bin_record missing '{k}.bins'")
        if "bin_value_distributions" not in rec[k]:
            raise KeyError(f"bin_record missing '{k}.bin_value_distributions'")

    return {
        "raw": rec,
        "half_tick": float(rec.get("half_tick", 0.005)),
        "price_edges": np.asarray(rec["price_mid_diff"]["bins"], dtype=float),
        "qty_edges": np.asarray(rec["order_qty"]["bins"], dtype=float),
        "interval_edges": np.asarray(rec["interval_ms"]["bins"], dtype=float),
        "price_dist": rec["price_mid_diff"]["bin_value_distributions"],
        "qty_dist": rec["order_qty"]["bin_value_distributions"],
        "interval_dist": rec["interval_ms"]["bin_value_distributions"],
        "digitize_right": True,
    }


def _decode_bin_value_fallback(edges: np.ndarray, b: int) -> float:
    lo = float(edges[b])
    hi = float(edges[b + 1])

    k_min = int(math.floor(lo)) + 1
    k_max = int(math.floor(hi))
    if k_min <= k_max and (k_max - k_min + 1) == 1:
        return float(k_min)

    return float(np.rint(0.5 * (lo + hi)))


def _decode_bin_value(
    dist_map: dict,
    edges: np.ndarray,
    b: int,
    method: str = "sample",
    rng=None,
    min_val: Optional[int] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng()

    try:
        entry = dist_map[str(int(b))]
        values = np.asarray(entry["unique_values"], dtype=float)
        probs = np.asarray(entry["probs"], dtype=float)
        counts = np.asarray(entry["counts"], dtype=int)

        if len(values) == 0:
            raise ValueError(f"Bin {b} has empty distribution.")

        if method == "sample":
            v = float(rng.choice(values, p=probs))
        elif method == "mode":
            v = float(values[counts.argmax()])
        else:
            raise ValueError(f"Unknown decode method: {method}")

    except Exception:
        v = _decode_bin_value_fallback(edges, b)

    if min_val is not None:
        v = max(float(min_val), float(v))

    return v


# ============================================================
# LOB DIAGNOSTIC HELPERS
# ============================================================

def _qsum(q: deque) -> int:
    return int(sum(q)) if q is not None else 0


def _qrepr(q: deque, max_items: int = 6) -> str:
    if q is None:
        return "[]"
    items = list(q)[:max_items]
    s = ",".join(str(int(x)) for x in items)
    if len(q) > max_items:
        s += ",..."
    return "[" + s + "]"


def book_snapshot(book, depth=3):
    bids = sorted(book.bids.keys(), reverse=True)[:depth]
    asks = sorted(book.asks.keys())[:depth]

    bid_lvls = []
    for p in bids:
        q = book.bids[p]
        bid_lvls.append({"price": float(p), "total_qty": _qsum(q), "queue": _qrepr(q)})

    ask_lvls = []
    for p in asks:
        q = book.asks[p]
        ask_lvls.append({"price": float(p), "total_qty": _qsum(q), "queue": _qrepr(q)})

    bb = book.best_bid()
    ba = book.best_ask()
    mid = book.midpoint()

    return {
        "best_bid": None if bb is None else float(bb),
        "best_ask": None if ba is None else float(ba),
        "mid": None if mid is None else float(mid),
        "bids": bid_lvls,
        "asks": ask_lvls,
    }


def print_snapshot(snap, prefix=""):
    print(prefix + f"BEST_BID={snap['best_bid']}  BEST_ASK={snap['best_ask']}  MID={snap['mid']}")
    print(prefix + "BIDS(top):")
    if not snap["bids"]:
        print(prefix + "  (empty)")
    for i, lv in enumerate(snap["bids"], 1):
        print(prefix + f"  L{i}: p={lv['price']:.3f} qty={lv['total_qty']} q={lv['queue']}")
    print(prefix + "ASKS(top):")
    if not snap["asks"]:
        print(prefix + "  (empty)")
    for i, lv in enumerate(snap["asks"], 1):
        print(prefix + f"  L{i}: p={lv['price']:.3f} qty={lv['total_qty']} q={lv['queue']}")


@dataclass
class EventDecoded:
    t_ms: int
    side_bin: int
    ticks_from_mid: int
    qty: int
    abs_price: float


class OrderBook:
    def __init__(self):
        self.bids = defaultdict(deque)
        self.asks = defaultdict(deque)

    def best_bid(self):
        return max(self.bids.keys()) if self.bids else None

    def best_ask(self):
        return min(self.asks.keys()) if self.asks else None

    def midpoint(self):
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def _clean_level(self, book_dict, price):
        q = book_dict.get(price, None)
        if q is not None and len(q) == 0:
            del book_dict[price]

    def _match_bid(self, price: float, qty: int):
        fills = []
        while qty > 0 and self.asks:
            ba = self.best_ask()
            if ba is None or price < ba:
                break

            level_q = self.asks[ba]
            filled_here = 0
            while qty > 0 and level_q:
                head_qty = level_q[0]
                take = min(qty, head_qty)
                qty -= take
                head_qty -= take
                filled_here += take
                if head_qty == 0:
                    level_q.popleft()
                else:
                    level_q[0] = head_qty

            if filled_here > 0:
                fills.append({"side": "ask", "price": float(ba), "filled_qty": int(filled_here)})

            self._clean_level(self.asks, ba)

        return qty, fills

    def _match_ask(self, price: float, qty: int):
        fills = []
        while qty > 0 and self.bids:
            bb = self.best_bid()
            if bb is None or price > bb:
                break

            level_q = self.bids[bb]
            filled_here = 0
            while qty > 0 and level_q:
                head_qty = level_q[0]
                take = min(qty, head_qty)
                qty -= take
                head_qty -= take
                filled_here += take
                if head_qty == 0:
                    level_q.popleft()
                else:
                    level_q[0] = head_qty

            if filled_here > 0:
                fills.append({"side": "bid", "price": float(bb), "filled_qty": int(filled_here)})

            self._clean_level(self.bids, bb)

        return qty, fills

    def post_limit(self, side_bin: int, price: float, qty: int):
        if qty <= 0:
            return []

        if side_bin == 0:
            rem, fills = self._match_bid(price, qty)
            if rem > 0:
                self.bids[price].append(rem)
            return fills

        if side_bin == 1:
            rem, fills = self._match_ask(price, qty)
            if rem > 0:
                self.asks[price].append(rem)
            return fills

        raise ValueError("post_limit called with cancel side")

    def cancel(self, side: str, price: float, qty: int):
        return self.remove_passive(side, price, qty, reason="cancel")

    def remove_passive(self, side: str, price: float, qty: int, *, reason: str):
        if qty <= 0:
            return 0

        book = self.bids if side == "bid" else self.asks
        if price not in book:
            return 0

        removed = 0
        level_q = book[price]
        while qty > 0 and level_q:
            head_qty = level_q[0]
            take = min(qty, head_qty)
            qty -= take
            head_qty -= take
            removed += take
            if head_qty == 0:
                level_q.popleft()
            else:
                level_q[0] = head_qty

        self._clean_level(book, price)
        return int(removed)

    def cancel_passive(self, side: str, price: float, qty: int):
        return self.remove_passive(side, price, qty, reason="cancel")

    def execute_against_resting(self, side: str, price: float, qty: int):
        return self.remove_passive(side, price, qty, reason="execute")


# ============================================================
# MODELS
# ============================================================

def _load_gpt2_backbone(gpt2_name: str = "gpt2"):
    cache_dir = os.environ.get("HF_HOME", None)
    try:
        return GPT2Model.from_pretrained(
            gpt2_name,
            local_files_only=True,
            cache_dir=cache_dir,
        )
    except Exception as local_exc:
        try:
            return GPT2Model.from_pretrained(
                gpt2_name,
                cache_dir=cache_dir,
            )
        except Exception as remote_exc:
            raise RuntimeError(
                "Failed to load GPT-2 backbone. Tried local Hugging Face cache first, "
                "then online download. If running on a cluster without stable HTTPS/certificates, "
                "make sure 'gpt2' is cached locally and use the local-only path."
            ) from remote_exc

class OrderGPT2NoAnchor(nn.Module):
    def __init__(self, vocab_size: int, gpt2_name: str = "gpt2"):
        super().__init__()
        self.gpt2 = _load_gpt2_backbone(gpt2_name)
        hidden_size = self.gpt2.config.hidden_size
        self.order_embedding = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb = self.order_embedding(x)
        out = self.gpt2(inputs_embeds=emb)
        h = out.last_hidden_state
        logits = self.head(h[:, -1, :])
        return logits


class OrderGPT2Anchor(nn.Module):
    def __init__(self, vocab_size: int, anchor_count: int, gpt2_name: str = "gpt2"):
        super().__init__()
        self.gpt2 = _load_gpt2_backbone(gpt2_name)
        hidden_size = self.gpt2.config.hidden_size
        self.order_embedding = nn.Embedding(vocab_size, hidden_size)
        self.dynamic_anchors = nn.Parameter(torch.randn(anchor_count, hidden_size) * 0.02)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb = self.order_embedding(x)
        query = emb.mean(dim=1)
        scores = torch.matmul(query, self.dynamic_anchors.t())
        anchor_probs = torch.softmax(scores, dim=1)
        weighted_anchor = torch.matmul(anchor_probs, self.dynamic_anchors)

        anchor_token = weighted_anchor.unsqueeze(1)
        gpt_input = torch.cat([anchor_token, emb], dim=1)

        out = self.gpt2(inputs_embeds=gpt_input)
        h = out.last_hidden_state
        logits = self.head(h[:, -1, :])
        return logits, anchor_probs, scores


def build_model(model_variant: str, vocab_size: int, gpt2_name: str = "gpt2", anchor_count: Optional[int] = None):
    if model_variant == "no_anchor":
        return OrderGPT2NoAnchor(vocab_size=vocab_size, gpt2_name=gpt2_name)

    if model_variant == "anchor":
        if anchor_count is None:
            raise ValueError("anchor_count must be provided when model_variant='anchor'")
        return OrderGPT2Anchor(vocab_size=vocab_size, anchor_count=anchor_count, gpt2_name=gpt2_name)

    raise ValueError(f"Unknown model_variant: {model_variant}. Expected 'anchor' or 'no_anchor'.")


def load_checkpoint(model: nn.Module, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", None)
    if state is None:
        state = ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return ckpt


# ============================================================
# SAMPLING
# ============================================================

def top_p_filtering(probs: torch.Tensor, top_p: float):
    if top_p >= 1.0:
        return probs

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=0)
    mask = cdf <= top_p
    if mask.sum() == 0:
        mask[0] = True
    kept_idx = sorted_idx[mask]
    kept_probs = probs[kept_idx]
    kept_probs = kept_probs / kept_probs.sum().clamp_min(1e-12)
    out = torch.zeros_like(probs)
    out[kept_idx] = kept_probs
    return out


def _extract_logits(model_out: torch.Tensor):
    # Anchor model returns tuple; no-anchor returns logits tensor.
    if isinstance(model_out, tuple):
        return model_out[0]
    return model_out


@torch.no_grad()
def sample_next_token(
    model: nn.Module,
    context_tokens: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    use_sampling: bool,
    sample_gen: Optional[torch.Generator] = None,
):
    logits = _extract_logits(model(context_tokens)).squeeze(0)

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits / float(temperature)
    probs = F.softmax(logits, dim=0)

    if not use_sampling:
        return int(torch.argmax(probs).item())

    probs = top_p_filtering(probs, top_p)
    token = torch.multinomial(probs.cpu(), 1, generator=sample_gen).item()
    return int(token)


# ============================================================
# DECODE BINS -> EVENT
# ============================================================

def _ceil_to_tick_rmb(x_rmb: float, tick: float = 0.01) -> float:
    if not np.isfinite(x_rmb):
        return float("nan")
    if x_rmb == 0.0:
        return 0.0
    sgn = 1.0 if x_rmb > 0 else -1.0
    return sgn * (math.ceil(abs(x_rmb) / tick) * tick)


def decode_event_from_token(
    token_id: int,
    binpack: dict,
    midpoint: float,
    cur_t_ms: int,
    *,
    price_bin_num: int,
    qty_bin_num: int,
    interval_bin_num: int,
    n_side: int,
    decode_method: str = "sample",
    rng=None,
):
    pbin, qbin, ibin, sbin = decode_order_token(
        token_id, price_bin_num, qty_bin_num, interval_bin_num, n_side
    )

    price_edges = binpack["price_edges"]
    qty_edges = binpack["qty_edges"]
    interval_edges = binpack["interval_edges"]

    price_dist = binpack["price_dist"]
    qty_dist = binpack["qty_dist"]
    interval_dist = binpack["interval_dist"]

    ticks_from_mid = _decode_bin_value(price_dist, price_edges, pbin, method=decode_method, rng=rng, min_val=None)
    qty = _decode_bin_value(qty_dist, qty_edges, qbin, method=decode_method, rng=rng, min_val=1)
    dt_ms = _decode_bin_value(interval_dist, interval_edges, ibin, method=decode_method, rng=rng, min_val=0)

    ticks_from_mid = int(np.rint(ticks_from_mid))
    qty = int(np.rint(qty))
    dt_ms = int(np.rint(dt_ms))

    if sbin == 0 and ticks_from_mid == 0:
        ticks_from_mid = -1
    if sbin == 1 and ticks_from_mid == 0:
        ticks_from_mid = +1

    half_tick = float(binpack.get("half_tick", 0.005))
    tick_size = 0.01

    diff_rmb = float(ticks_from_mid) * half_tick
    diff_rmb = _ceil_to_tick_rmb(diff_rmb, tick=tick_size)

    abs_price = float(midpoint + diff_rmb)
    abs_price = float(np.round(abs_price / tick_size) * tick_size)

    ev = EventDecoded(
        t_ms=int(cur_t_ms + dt_ms),
        side_bin=int(sbin),
        ticks_from_mid=int(ticks_from_mid),
        qty=int(qty),
        abs_price=float(abs_price),
    )
    return ev, int(dt_ms)


def decode_event_from_token_open_anchor(
    token_id: int,
    binpack: dict,
    anchor_price: float,
    cur_t_ms: int,
    *,
    price_bin_num: int,
    qty_bin_num: int,
    interval_bin_num: int,
    n_side: int,
    decode_method: str = "sample",
    rng=None,
):
    pbin, qbin, ibin, sbin = decode_order_token(
        token_id, price_bin_num, qty_bin_num, interval_bin_num, n_side
    )

    price_edges = binpack["price_edges"]
    qty_edges = binpack["qty_edges"]
    interval_edges = binpack["interval_edges"]

    price_dist = binpack["price_dist"]
    qty_dist = binpack["qty_dist"]
    interval_dist = binpack["interval_dist"]

    price_ticks_from_anchor = _decode_bin_value(
        price_dist, price_edges, pbin, method=decode_method, rng=rng, min_val=None
    )
    qty = _decode_bin_value(qty_dist, qty_edges, qbin, method=decode_method, rng=rng, min_val=1)
    dt_ms = _decode_bin_value(interval_dist, interval_edges, ibin, method=decode_method, rng=rng, min_val=0)

    price_ticks_from_anchor = int(np.rint(price_ticks_from_anchor))
    qty = int(np.rint(qty))
    dt_ms = int(np.rint(dt_ms))

    tick_size = float(binpack.get("raw", {}).get("tick_size", 0.01))
    abs_price = float(anchor_price + price_ticks_from_anchor * tick_size)
    abs_price = float(np.round(abs_price / tick_size) * tick_size)

    ev = EventDecoded(
        t_ms=int(cur_t_ms + dt_ms),
        side_bin=int(sbin),
        ticks_from_mid=int(price_ticks_from_anchor),
        qty=int(qty),
        abs_price=float(abs_price),
    )
    return ev, int(dt_ms)


# ============================================================
# BOOK APPLY
# ============================================================

def apply_event_to_book(book: OrderBook, ev: EventDecoded):
    out = {
        "event_kind": None,
        "action": None,
        "fills": [],
        "removed": 0,
        "resting_side": None,
        "price": float(ev.abs_price),
        "requested_qty": int(ev.qty),
    }

    if ev.side_bin in (0, 1):
        fills = book.post_limit(ev.side_bin, ev.abs_price, ev.qty)
        out["event_kind"] = "post"
        out["action"] = "POST_BID" if ev.side_bin == 0 else "POST_ASK"
        out["fills"] = fills
        return out

    if ev.ticks_from_mid > 0:
        removed = book.cancel_passive("ask", ev.abs_price, ev.qty)
        out["event_kind"] = "cancel"
        out["action"] = "CANCEL_ASK"
        out["removed"] = removed
        out["resting_side"] = "ask"
        return out

    if ev.ticks_from_mid < 0:
        removed = book.cancel_passive("bid", ev.abs_price, ev.qty)
        out["event_kind"] = "cancel"
        out["action"] = "CANCEL_BID"
        out["removed"] = removed
        out["resting_side"] = "bid"
        return out

    if book.asks:
        removed = book.cancel_passive("ask", ev.abs_price, ev.qty)
        out["event_kind"] = "cancel"
        out["action"] = "CANCEL0_ASK"
        out["removed"] = removed
        out["resting_side"] = "ask"
    elif book.bids:
        removed = book.cancel_passive("bid", ev.abs_price, ev.qty)
        out["event_kind"] = "cancel"
        out["action"] = "CANCEL0_BID"
        out["removed"] = removed
        out["resting_side"] = "bid"
    else:
        out["event_kind"] = "cancel"
        out["action"] = "CANCEL0_EMPTY"
        out["removed"] = 0
    return out


def apply_event_to_book_open_anchor(book: OrderBook, ev: EventDecoded):
    out = {
        "event_kind": None,
        "action": None,
        "fills": [],
        "removed": 0,
        "resting_side": None,
        "price": float(ev.abs_price),
        "requested_qty": int(ev.qty),
    }

    if ev.side_bin in (0, 1):
        fills = book.post_limit(ev.side_bin, ev.abs_price, ev.qty)
        out["event_kind"] = "post"
        out["action"] = "POST_BID" if ev.side_bin == 0 else "POST_ASK"
        out["fills"] = fills
        return out

    ask_has = ev.abs_price in book.asks and _qsum(book.asks[ev.abs_price]) > 0
    bid_has = ev.abs_price in book.bids and _qsum(book.bids[ev.abs_price]) > 0

    if ask_has and not bid_has:
        removed = book.cancel_passive("ask", ev.abs_price, ev.qty)
        out["event_kind"] = "cancel"
        out["action"] = "CANCEL_ASK"
        out["removed"] = removed
        out["resting_side"] = "ask"
        return out

    if bid_has and not ask_has:
        removed = book.cancel_passive("bid", ev.abs_price, ev.qty)
        out["event_kind"] = "cancel"
        out["action"] = "CANCEL_BID"
        out["removed"] = removed
        out["resting_side"] = "bid"
        return out

    if ask_has and bid_has:
        removed_ask = book.cancel_passive("ask", ev.abs_price, ev.qty)
        out["event_kind"] = "cancel"
        out["action"] = "CANCEL_BOTH_PREF_ASK"
        out["removed"] = removed_ask
        out["resting_side"] = "ask"
        return out

    midpoint = book.midpoint()
    if midpoint is not None:
        preferred_side = "ask" if ev.abs_price >= midpoint else "bid"
        removed = book.cancel_passive(preferred_side, ev.abs_price, ev.qty)
        out["event_kind"] = "cancel"
        out["action"] = f"CANCEL_MISS_{preferred_side.upper()}"
        out["removed"] = removed
        out["resting_side"] = preferred_side
        return out

    out["event_kind"] = "cancel"
    out["action"] = "CANCEL_MISS_EMPTY"
    out["removed"] = 0
    return out


def apply_event_to_book_open_anchor_txn_complete(
    book: OrderBook, ev: EventDecoded, *, split_cancel_sides: bool = False
):
    """
    Txn-complete open-anchor book update.

    When split_cancel_sides=False (5-way side bins): posts (0,1), ambiguous cancel (2),
    transaction-complete against resting bid/ask (3,4).

    When split_cancel_sides=True (6-way side bins): posts (0,1), explicit cancel resting
    bid (2) / ask (3), transaction-complete against resting bid/ask (4,5).
    """
    out = {
        "event_kind": None,
        "action": None,
        "fills": [],
        "removed": 0,
        "resting_side": None,
        "price": float(ev.abs_price),
        "requested_qty": int(ev.qty),
        "rejected": False,
    }

    if ev.side_bin == 0:
        best_ask = book.best_ask()
        if best_ask is not None and ev.abs_price >= best_ask:
            out["event_kind"] = "post"
            out["action"] = "REJECT_CROSS_BID"
            out["rejected"] = True
            return out

        fills = book.post_limit(ev.side_bin, ev.abs_price, ev.qty)
        out["event_kind"] = "post"
        out["action"] = "POST_BID"
        out["fills"] = fills
        return out

    if ev.side_bin == 1:
        best_bid = book.best_bid()
        if best_bid is not None and ev.abs_price <= best_bid:
            out["event_kind"] = "post"
            out["action"] = "REJECT_CROSS_ASK"
            out["rejected"] = True
            return out

        fills = book.post_limit(ev.side_bin, ev.abs_price, ev.qty)
        out["event_kind"] = "post"
        out["action"] = "POST_ASK"
        out["fills"] = fills
        return out

    if split_cancel_sides:
        if ev.side_bin == 2:
            removed = book.cancel_passive("bid", ev.abs_price, ev.qty)
            out["event_kind"] = "cancel"
            out["action"] = "CANCEL_BID"
            out["removed"] = removed
            out["resting_side"] = "bid"
            return out

        if ev.side_bin == 3:
            removed = book.cancel_passive("ask", ev.abs_price, ev.qty)
            out["event_kind"] = "cancel"
            out["action"] = "CANCEL_ASK"
            out["removed"] = removed
            out["resting_side"] = "ask"
            return out

        if ev.side_bin == 4:
            removed = book.execute_against_resting("bid", ev.abs_price, ev.qty)
            out["event_kind"] = "transaction_complete"
            out["action"] = "EXEC_BID"
            out["removed"] = removed
            out["resting_side"] = "bid"
            if removed > 0:
                out["fills"] = [{"side": "bid", "price": float(ev.abs_price), "filled_qty": int(removed)}]
            return out

        if ev.side_bin == 5:
            removed = book.execute_against_resting("ask", ev.abs_price, ev.qty)
            out["event_kind"] = "transaction_complete"
            out["action"] = "EXEC_ASK"
            out["removed"] = removed
            out["resting_side"] = "ask"
            if removed > 0:
                out["fills"] = [{"side": "ask", "price": float(ev.abs_price), "filled_qty": int(removed)}]
            return out

        raise ValueError(f"Unsupported txn-complete side_bin={ev.side_bin} (split_cancel_sides=True)")

    if ev.side_bin == 2:
        ask_has = ev.abs_price in book.asks and _qsum(book.asks[ev.abs_price]) > 0
        bid_has = ev.abs_price in book.bids and _qsum(book.bids[ev.abs_price]) > 0

        if ask_has and not bid_has:
            removed = book.cancel_passive("ask", ev.abs_price, ev.qty)
            out["event_kind"] = "cancel"
            out["action"] = "CANCEL_ASK"
            out["removed"] = removed
            out["resting_side"] = "ask"
            return out

        if bid_has and not ask_has:
            removed = book.cancel_passive("bid", ev.abs_price, ev.qty)
            out["event_kind"] = "cancel"
            out["action"] = "CANCEL_BID"
            out["removed"] = removed
            out["resting_side"] = "bid"
            return out

        if ask_has and bid_has:
            removed_ask = book.cancel_passive("ask", ev.abs_price, ev.qty)
            out["event_kind"] = "cancel"
            out["action"] = "CANCEL_BOTH_PREF_ASK"
            out["removed"] = removed_ask
            out["resting_side"] = "ask"
            return out

        midpoint = book.midpoint()
        if midpoint is not None:
            preferred_side = "ask" if ev.abs_price >= midpoint else "bid"
            removed = book.cancel_passive(preferred_side, ev.abs_price, ev.qty)
            out["event_kind"] = "cancel"
            out["action"] = f"CANCEL_MISS_{preferred_side.upper()}"
            out["removed"] = removed
            out["resting_side"] = preferred_side
            return out

        if book.asks:
            removed = book.cancel_passive("ask", ev.abs_price, ev.qty)
            out["event_kind"] = "cancel"
            out["action"] = "CANCEL_MISS_ASK_FALLBACK"
            out["removed"] = removed
            out["resting_side"] = "ask"
            return out

        if book.bids:
            removed = book.cancel_passive("bid", ev.abs_price, ev.qty)
            out["event_kind"] = "cancel"
            out["action"] = "CANCEL_MISS_BID_FALLBACK"
            out["removed"] = removed
            out["resting_side"] = "bid"
            return out

        out["event_kind"] = "cancel"
        out["action"] = "CANCEL_MISS_EMPTY"
        return out

    if ev.side_bin == 3:
        removed = book.execute_against_resting("bid", ev.abs_price, ev.qty)
        out["event_kind"] = "transaction_complete"
        out["action"] = "EXEC_BID"
        out["removed"] = removed
        out["resting_side"] = "bid"
        if removed > 0:
            out["fills"] = [{"side": "bid", "price": float(ev.abs_price), "filled_qty": int(removed)}]
        return out

    if ev.side_bin == 4:
        removed = book.execute_against_resting("ask", ev.abs_price, ev.qty)
        out["event_kind"] = "transaction_complete"
        out["action"] = "EXEC_ASK"
        out["removed"] = removed
        out["resting_side"] = "ask"
        if removed > 0:
            out["fills"] = [{"side": "ask", "price": float(ev.abs_price), "filled_qty": int(removed)}]
        return out

    raise ValueError(f"Unsupported txn-complete side_bin={ev.side_bin}")


def _notna_scalar(x):
    if x is None:
        return False
    try:
        return not bool(np.isnan(x))
    except Exception:
        return True


def _get_snapshot_value(row, col, default=None):
    if hasattr(row, "__contains__") and col in row:
        v = row[col]
        return default if not _notna_scalar(v) else v
    return default


def init_book_from_snapshot(snapshot_row, max_level=10):
    book = OrderBook()

    for lvl in range(1, max_level + 1):
        bp = _get_snapshot_value(snapshot_row, f"BidPrice{lvl}", None)
        bv = _get_snapshot_value(snapshot_row, f"BidVolume{lvl}", 0)
        if bp is not None and bv is not None:
            bp = float(bp)
            bv = int(round(float(bv)))
            if bv > 0:
                book.bids[bp].append(bv)

    for lvl in range(1, max_level + 1):
        ap = _get_snapshot_value(snapshot_row, f"AskPrice{lvl}", None)
        av = _get_snapshot_value(snapshot_row, f"AskVolume{lvl}", 0)
        if ap is not None and av is not None:
            ap = float(ap)
            av = int(round(float(av)))
            if av > 0:
                book.asks[ap].append(av)

    return book


# ============================================================
# MAIN SIMULATION
# ============================================================

def simulate_from_snapshot_and_context(
    window_df,
    snapshot_row,
    lookahead_ms,
    *,
    ckpt_path: str,
    bin_record_path: str,
    device,
    model_variant: str,
    vocab_size: int,
    window_len: int,
    price_bin_num: int,
    qty_bin_num: int,
    interval_bin_num: int,
    n_side: int,
    use_sampling: bool,
    temperature: float,
    top_p: float,
    anchor_count: Optional[int] = None,
    sample_gen: Optional[torch.Generator] = None,
    verbose_steps: int = VERBOSE_STEPS_DEFAULT,
    verbose_depth: int = VERBOSE_DEPTH_DEFAULT,
    max_snapshot_level: int = 10,
    print_context_preview: bool = True,
    fallback_initial_mid: float = 100.0,
    gpt2_name: str = "gpt2",
):
    preview_rng = np.random.default_rng(42)
    gen_rng = np.random.default_rng(43)

    if window_df is None or len(window_df) == 0:
        raise RuntimeError("window_df is empty")
    if "order_token" not in window_df.columns:
        raise RuntimeError("window_df must contain column: order_token")
    if "SecurityID" not in window_df.columns:
        raise RuntimeError("window_df must contain column: SecurityID")

    sort_cols = ["SecurityID", "TransactDT_MS", "ChannelNo", "ApplSeqNum"]
    usable_sort_cols = [c for c in sort_cols if c in window_df.columns]
    if usable_sort_cols:
        window_df = window_df.sort_values(usable_sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        window_df = window_df.reset_index(drop=True)

    sid0 = str(window_df["SecurityID"].iloc[0])

    print("Loading bin_record:", bin_record_path)
    binpack = load_bin_record(bin_record_path)

    print("Loading checkpoint:", ckpt_path)
    model = build_model(
        model_variant=model_variant,
        vocab_size=vocab_size,
        gpt2_name=gpt2_name,
        anchor_count=anchor_count,
    )
    ckpt_meta = load_checkpoint(model, ckpt_path, device)
    print("Checkpoint keys:", list(ckpt_meta.keys()) if isinstance(ckpt_meta, dict) else type(ckpt_meta))

    book = init_book_from_snapshot(snapshot_row, max_level=max_snapshot_level)

    snap_mid = _get_snapshot_value(snapshot_row, "MidPrice", None)
    if snap_mid is not None:
        cur_mid = float(snap_mid)
        print("Initial midpoint from snapshot MidPrice:", cur_mid)
    else:
        m = book.midpoint()
        if m is not None:
            cur_mid = float(m)
            print("Initial midpoint from reconstructed snapshot book:", cur_mid)
        else:
            cur_mid = float(fallback_initial_mid)
            print("Initial midpoint FALLBACK:", cur_mid)

    cur_t_ms = 0
    horizon_t_ms = int(lookahead_ms)

    tokens = window_df["order_token"].astype(int).to_numpy()
    if len(tokens) < window_len:
        raise RuntimeError(f"Need at least {window_len} context tokens, got {len(tokens)}")
    context = tokens[-window_len:].tolist()

    print(f"Simulating SecurityID: {sid0}")
    print(f"Model variant: {model_variant}")
    print(f"Internal start time_ms: {cur_t_ms}")
    print(f"Lookahead horizon_ms: {horizon_t_ms}")
    print(f"Context token count used: {len(context)}")

    print("\nInitial snapshot-loaded book:")
    init_snap = book_snapshot(book, depth=verbose_depth)
    print_snapshot(init_snap, prefix="  ")

    context_log = []
    if print_context_preview:
        print("\nContext preview (used for LLM conditioning only; NOT applied to book):")

    for i, tok in enumerate(context):
        ev_preview, dt_preview = decode_event_from_token(
            tok,
            binpack,
            cur_mid,
            cur_t_ms,
            price_bin_num=price_bin_num,
            qty_bin_num=qty_bin_num,
            interval_bin_num=interval_bin_num,
            n_side=n_side,
            decode_method="sample",
            rng=preview_rng,
        )

        pbin, qbin, ibin, sbin = decode_order_token(tok, price_bin_num, qty_bin_num, interval_bin_num, n_side)
        side_name = {0: "BID_POST", 1: "ASK_POST", 2: "CANCEL"}.get(sbin, str(sbin))

        context_log.append(
            {
                "ctx_idx": int(i),
                "token": int(tok),
                "pbin": int(pbin),
                "qbin": int(qbin),
                "ibin": int(ibin),
                "side_bin": int(sbin),
                "side_name": side_name,
                "decoded_ticks_from_mid": int(ev_preview.ticks_from_mid),
                "decoded_qty": int(ev_preview.qty),
                "decoded_dt_ms": int(dt_preview),
                "decoded_abs_price_if_applied_now": float(ev_preview.abs_price),
            }
        )

        if print_context_preview and i < verbose_steps:
            print(f"\n[CTX {i:03d}] token={tok}")
            print(f"  decode: pbin={pbin} qbin={qbin} ibin={ibin} side={sbin}({side_name})")
            print(
                f"  preview(empirical-sample decode): ticks_from_mid={ev_preview.ticks_from_mid} "
                f"qty={ev_preview.qty} dt_ms={dt_preview} abs_price={ev_preview.abs_price:.5f}"
            )

    print("\nContext loaded. Book unchanged. Generation starts now from snapshot state.")

    generation_log = []
    step = 0

    print(f"\nGenerating forward until simulated t_ms >= {horizon_t_ms} ...")

    while cur_t_ms < horizon_t_ms:
        ctx = torch.tensor(context[-window_len:], dtype=torch.long, device=device).unsqueeze(0)

        tok_next = sample_next_token(
            model,
            ctx,
            temperature=temperature,
            top_p=top_p,
            use_sampling=use_sampling,
            sample_gen=sample_gen,
        )

        ev, dt_ms = decode_event_from_token(
            tok_next,
            binpack,
            cur_mid,
            cur_t_ms,
            price_bin_num=price_bin_num,
            qty_bin_num=qty_bin_num,
            interval_bin_num=interval_bin_num,
            n_side=n_side,
            decode_method="sample",
            rng=gen_rng,
        )
        cur_t_ms = int(ev.t_ms)

        action = apply_event_to_book(book, ev)

        m = book.midpoint()
        if m is not None:
            cur_mid = float(m)

        context.append(int(tok_next))

        bb = book.best_bid()
        ba = book.best_ask()

        generation_log.append(
            {
                "gen_idx": int(step),
                "t_ms": int(cur_t_ms),
                "token": int(tok_next),
                "side_bin": int(ev.side_bin),
                "ticks_from_mid": int(ev.ticks_from_mid),
                "qty": int(ev.qty),
                "abs_price": float(ev.abs_price),
                "action": action.get("action", None),
                "fills": action.get("fills", []),
                "removed": action.get("removed", 0),
                "dt_ms": int(dt_ms),
                "mid_after": None if cur_mid is None else float(cur_mid),
                "best_bid_after": None if bb is None else float(bb),
                "best_ask_after": None if ba is None else float(ba),
            }
        )

        if step < verbose_steps:
            pbin, qbin, ibin, sbin = decode_order_token(tok_next, price_bin_num, qty_bin_num, interval_bin_num, n_side)
            side_name = {0: "BID_POST", 1: "ASK_POST", 2: "CANCEL"}.get(sbin, str(sbin))
            print(f"\n[GEN {step:03d}] sampled_token={tok_next}")
            print(f"  decode: pbin={pbin} qbin={qbin} ibin={ibin} side={sbin}({side_name})")
            print(
                f"  decode(empirical-sample): ticks_from_mid={ev.ticks_from_mid} "
                f"qty={ev.qty} dt_ms={dt_ms} abs_price={ev.abs_price:.5f}"
            )
            print(f"  apply: {action['action']} fills={action.get('fills', [])} removed={action.get('removed', 0)}")
            snap = book_snapshot(book, depth=verbose_depth)
            print_snapshot(snap, prefix="  ")

        if (step + 1) % 500 == 0:
            print(
                f"[SIM] step={step+1} t_ms={cur_t_ms} "
                f"mid={'None' if cur_mid is None else f'{cur_mid:.3f}'} "
                f"bb={'None' if bb is None else f'{bb:.3f}'} "
                f"ba={'None' if ba is None else f'{ba:.3f}'} dt_ms={dt_ms}"
            )

        step += 1

    final_snap = book_snapshot(book, depth=verbose_depth)

    return {
        "security_id": sid0,
        "lookahead_ms": int(horizon_t_ms),
        "reached_t_ms": int(cur_t_ms),
        "simulated_mid_at_horizon": None if cur_mid is None else float(cur_mid),
        "final_best_bid": None if book.best_bid() is None else float(book.best_bid()),
        "final_best_ask": None if book.best_ask() is None else float(book.best_ask()),
        "generated_event_count": int(step),
        "context_log": context_log,
        "generation_log": generation_log,
        "final_snapshot": final_snap,
        "checkpoint_meta": ckpt_meta,
    }


# ============================================================
# EVAL HELPERS
# ============================================================

def _to_time_str(x):
    ts = pd.Timestamp(x)
    return ts.strftime("%H:%M:%S")


def _get_stock_snapshot_times(LOB_info, stock):
    df_stock = LOB_info[LOB_info["SecurityID"] == stock].copy()
    if df_stock.empty:
        raise RuntimeError(f"No snapshot rows found for stock={stock}")

    if "TransactDT_SEC" not in df_stock.columns:
        raise RuntimeError("LOB_info must contain column: TransactDT_SEC")

    df_stock["TransactDT_SEC"] = pd.to_datetime(df_stock["TransactDT_SEC"])
    df_stock = df_stock.sort_values("TransactDT_SEC").reset_index(drop=True)

    return df_stock["TransactDT_SEC"].drop_duplicates().tolist()


def _build_valid_time_pairs(LOB_info, stock, lookahead_minutes=2):
    times = _get_stock_snapshot_times(LOB_info, stock)
    time_set = set(times)

    delta = pd.Timedelta(minutes=lookahead_minutes)
    valid_pairs = []
    for t0 in times:
        t1 = t0 + delta
        if t1 in time_set:
            valid_pairs.append((_to_time_str(t0), _to_time_str(t1)))

    if len(valid_pairs) == 0:
        raise RuntimeError(
            f"No valid start/horizon pairs found for stock={stock} with {lookahead_minutes}-minute lookahead."
        )

    return valid_pairs


def _direction_sign(x, eps=1e-12):
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def run_random_midprice_eval_resample_on_error(
    LOB_info,
    processed_LOB_data,
    *,
    stock: str,
    n_samples: int,
    lookahead_minutes: int,
    order_num: int,
    ckpt_path: str,
    bin_record_path: str,
    device,
    model_variant: str,
    vocab_size: int,
    window_len: int,
    price_bin_num: int,
    qty_bin_num: int,
    interval_bin_num: int,
    n_side: int,
    use_sampling: bool,
    temperature: float,
    top_p: float,
    anchor_count: Optional[int] = None,
    sample_gen: Optional[torch.Generator] = None,
    max_attempts_multiplier: int = 20,
    verbose_steps: int = VERBOSE_STEPS_DEFAULT,
    verbose_depth: int = VERBOSE_DEPTH_DEFAULT,
    max_snapshot_level: int = 10,
    print_context_preview: bool = True,
    fallback_initial_mid: float = 100.0,
    gpt2_name: str = "gpt2",
    random_seed: Optional[int] = None,
):
    rng_choice = random.Random(random_seed) if random_seed is not None else random

    valid_pairs = _build_valid_time_pairs(LOB_info, stock, lookahead_minutes=lookahead_minutes)
    if len(valid_pairs) == 0:
        raise RuntimeError("No valid time pairs available.")

    max_attempts = max(n_samples * max_attempts_multiplier, len(valid_pairs))
    tried_pairs = set()
    results = []

    attempt = 0
    success_count = 0

    while success_count < n_samples and attempt < max_attempts:
        attempt += 1

        remaining_pairs = [p for p in valid_pairs if p not in tried_pairs]
        if len(remaining_pairs) == 0:
            break

        starting_time, horizon_time = rng_choice.choice(remaining_pairs)
        tried_pairs.add((starting_time, horizon_time))

        try:
            snapshot_start = get_lob_snapshot_by_time(LOB_info, starting_time, stock)
            if snapshot_start is None or len(snapshot_start) == 0:
                continue
            snapshot_start_row = snapshot_start.iloc[0]

            snapshot_horizon = get_lob_snapshot_by_time(LOB_info, horizon_time, stock)
            if snapshot_horizon is None or len(snapshot_horizon) == 0:
                continue
            snapshot_horizon_row = snapshot_horizon.iloc[0]

            window_df = get_order_window_ending_at_second(
                processed_LOB_data=processed_LOB_data,
                target_time=starting_time,
                stock=stock,
                order_num=order_num,
            )
            if window_df is None or len(window_df) < order_num:
                continue

            start_mid = float(snapshot_start_row["MidPrice"])
            true_mid = float(snapshot_horizon_row["MidPrice"])

            sim_result = simulate_from_snapshot_and_context(
                window_df=window_df,
                snapshot_row=snapshot_start_row,
                lookahead_ms=lookahead_minutes * 60 * 1000,
                ckpt_path=ckpt_path,
                bin_record_path=bin_record_path,
                device=device,
                model_variant=model_variant,
                vocab_size=vocab_size,
                window_len=window_len,
                price_bin_num=price_bin_num,
                qty_bin_num=qty_bin_num,
                interval_bin_num=interval_bin_num,
                n_side=n_side,
                use_sampling=use_sampling,
                temperature=temperature,
                top_p=top_p,
                anchor_count=anchor_count,
                sample_gen=sample_gen,
                verbose_steps=verbose_steps,
                verbose_depth=verbose_depth,
                max_snapshot_level=max_snapshot_level,
                print_context_preview=print_context_preview,
                fallback_initial_mid=fallback_initial_mid,
                gpt2_name=gpt2_name,
            )

            pred_mid = sim_result["simulated_mid_at_horizon"]
            if pred_mid is None:
                continue

            pred_ret = float(pred_mid) - start_mid
            true_ret = true_mid - start_mid

            pred_dir = _direction_sign(pred_ret)
            true_dir = _direction_sign(true_ret)
            dir_correct = int(pred_dir == true_dir)

            results.append(
                {
                    "stock": stock,
                    "model_variant": model_variant,
                    "starting_time": starting_time,
                    "horizon_time": horizon_time,
                    "start_mid": start_mid,
                    "pred_mid": float(pred_mid),
                    "true_mid": true_mid,
                    "pred_ret": pred_ret,
                    "true_ret": true_ret,
                    "pred_dir": pred_dir,
                    "true_dir": true_dir,
                    "dir_correct": dir_correct,
                    "generated_event_count": int(sim_result["generated_event_count"]),
                    "reached_t_ms": int(sim_result["reached_t_ms"]),
                }
            )
            success_count += 1

        except Exception:
            continue

    if len(results) == 0:
        raise RuntimeError("No valid evaluation results were produced.")

    results_df = pd.DataFrame(results)
    corr = results_df["pred_mid"].corr(results_df["true_mid"])
    direction_hits = int(results_df["dir_correct"].sum())
    n_done = int(len(results_df))
    direction_acc = direction_hits / n_done

    return {
        "results_df": results_df,
        "corr_pred_true_mid": corr,
        "direction_correct_count": direction_hits,
        "direction_accuracy": direction_acc,
        "attempts_used": attempt,
    }
