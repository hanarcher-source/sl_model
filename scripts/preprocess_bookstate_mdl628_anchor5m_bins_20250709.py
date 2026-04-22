#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess per-second 10-level LOB snapshots from `LOB_data/<DAY>/mdl_6_28_0.csv`
into a compact per-second tokenized "book state" dataset.

Design:
  - Snapshot time grid: per-second (floor UpdateTime to seconds; keep last row per second)
  - Anchor: fixed 5-minute buckets; anchor snapshot = first snapshot in each bucket
  - Encoding per level-slot (20 slots = 10 ask + 10 bid):
      price_delta_ticks in [-max_tick, +max_tick]  -> P = 2*max_tick + 1 bins
      vol_delta_log signed, delta = log1p(v) - log1p(v_anchor) -> V bins (symmetric)
      joint token: token = price_bin * V + vol_bin   (K = P*V)

Output:
  joblib file with columns:
    - SecurityID (e.g. 000617_XSHE)
    - TradeDate (YYYYMMDD)
    - TransactDT_SEC (timestamp, date=1900-01-01)
    - BucketStart (timestamp floored to 5min)
    - book_token_00 ... book_token_19 (int32)
  plus a JSON meta file containing binning parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Iterable

import joblib
import numpy as np
import pandas as pd


def _read_header_fields(path: str) -> list[str]:
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
    return [str(x).strip() for x in header]


def _stock_to_raw_id(stock: str) -> str:
    # stock like 000617_XSHE -> 000617
    s = str(stock).strip()
    if "_" in s:
        s = s.split("_", 1)[0]
    return str(s).zfill(6)


def _raw_id_to_stock(raw_id: str) -> str:
    return str(raw_id).zfill(6) + "_XSHE"


def _lob_cols(levels: int) -> list[str]:
    cols: list[str] = []
    for i in range(1, levels + 1):
        cols.append(f"BidPrice{i}")
    for i in range(1, levels + 1):
        cols.append(f"AskPrice{i}")
    for i in range(1, levels + 1):
        cols.append(f"BidVolume{i}")
    for i in range(1, levels + 1):
        cols.append(f"AskVolume{i}")
    return cols


def _safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _fit_symmetric_linear_edges(x: np.ndarray, bins: int, clip_quantile: float) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise RuntimeError("No finite volume deltas found to fit bins.")
    max_abs = float(np.quantile(np.abs(x), float(clip_quantile)))
    max_abs = max(max_abs, 1e-6)
    edges = np.linspace(-max_abs, +max_abs, int(bins) + 1, dtype=np.float64)
    return edges, max_abs


def _digitize_clip(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # returns int bin in [0, len(edges)-2]
    x = np.asarray(x, dtype=np.float64)
    lo = float(edges[0])
    hi = float(edges[-1])
    x = np.clip(x, lo, hi)
    b = np.digitize(x, edges, right=True) - 1
    b = np.clip(b, 0, len(edges) - 2)
    return b.astype(np.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--day", default="20250709")
    p.add_argument(
        "--lob-path",
        default="",
        help="Optional override path to mdl_6_28_0.csv. If empty, uses /finance_ML/.../LOB_data/<day>/mdl_6_28_0.csv",
    )
    p.add_argument(
        "--stocks",
        default="000617_XSHE,002263_XSHE,002721_XSHE",
        help="Comma-separated SecurityIDs to keep.",
    )
    p.add_argument("--levels", type=int, default=10)
    p.add_argument("--tick-size", type=float, default=0.01)
    p.add_argument("--max-tick", type=int, default=20)
    p.add_argument("--vol-bins", type=int, default=31)
    p.add_argument("--anchor-minutes", type=int, default=5)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--vol-clip-quantile", type=float, default=0.995)
    p.add_argument(
        "--vol-edges-from-meta",
        default="",
        help="If set, load `vol_edges` from a previously written meta.json (e.g. 0709) and reuse them (no refit).",
    )
    p.add_argument(
        "--keep-raw",
        action="store_true",
        help="If set, include absolute per-second LOB columns and per-bucket anchor columns in the output joblib (for snapshot-metric evaluation).",
    )
    p.add_argument(
        "--output-dir",
        default="/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/processed_book_state/pool_0709_bookstate_anchor5m_mdl628",
    )
    args = p.parse_args()

    day = str(args.day).strip()
    stocks = [s.strip() for s in str(args.stocks).split(",") if s.strip()]
    raw_ids = {_stock_to_raw_id(s) for s in stocks}
    levels = int(args.levels)
    tick_size = float(args.tick_size)
    max_tick = int(args.max_tick)
    P = 2 * max_tick + 1
    V = int(args.vol_bins)
    K = P * V

    lob_path = str(args.lob_path).strip()
    if not lob_path:
        lob_path = f"/finance_ML/zhanghaohan/LOB_data/{day}/mdl_6_28_0.csv"
    if not os.path.exists(lob_path):
        raise FileNotFoundError(lob_path)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_job = os.path.join(args.output_dir, f"bookstate_{day}_mdl628_anchor{args.anchor_minutes}m_P{P}_V{V}_{ts}.joblib")
    out_meta = os.path.join(args.output_dir, f"bookstate_{day}_mdl628_anchor{args.anchor_minutes}m_P{P}_V{V}_{ts}_meta.json")

    edges_from = str(args.vol_edges_from_meta).strip()
    reuse_edges = None
    if edges_from:
        if not os.path.exists(edges_from):
            raise FileNotFoundError(edges_from)
        with open(edges_from, "r", encoding="utf-8") as f:
            prior = json.load(f)
        prior_edges = np.asarray(prior.get("vol_edges", []), dtype=np.float64)
        if prior_edges.size != (V + 1):
            raise ValueError(f"vol_edges_from_meta has {prior_edges.size} edges; expected V+1={V+1}")
        reuse_edges = prior_edges

    header = _read_header_fields(lob_path)
    # Header row is 89 fields; data rows are 90. Pad one extra column name.
    if len(header) < 90:
        header = header + ["EXTRA_COL"]
    else:
        header = header[:90]

    need_cols = ["UpdateTime", "SecurityID"] + _lob_cols(levels)
    missing = [c for c in need_cols if c not in header]
    if missing:
        raise KeyError(f"Missing required columns in header: {missing[:10]} ...")

    usecols = need_cols
    df_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        lob_path,
        header=None,
        names=header,
        usecols=usecols,
        skiprows=1,
        chunksize=int(args.chunksize),
        low_memory=True,
        dtype={"SecurityID": str, "UpdateTime": str},
        on_bad_lines="skip",
    ):
        # raw SecurityID is 6-digit string (no suffix)
        chunk["SecurityID"] = chunk["SecurityID"].astype(str).str.strip().str.zfill(6)
        chunk = chunk[chunk["SecurityID"].isin(raw_ids)].copy()
        if chunk.empty:
            continue
        df_parts.append(chunk)

    if not df_parts:
        raise RuntimeError(f"No rows found for stocks={sorted(stocks)} in {lob_path}")

    df = pd.concat(df_parts, ignore_index=True)
    df["SecurityID"] = df["SecurityID"].map(_raw_id_to_stock)

    # Time parsing: UpdateTime like HH:MM:SS.mmm -> floor to seconds
    ut = df["UpdateTime"].astype(str).str.slice(0, 8)
    df["TransactDT_SEC"] = pd.to_datetime(ut, format="%H:%M:%S", errors="coerce").dt.floor("s")
    df = df.dropna(subset=["TransactDT_SEC"]).copy()

    # Numeric conversion for prices/volumes
    lob_cols = _lob_cols(levels)
    _safe_numeric(df, lob_cols)
    df = df.dropna(subset=["BidPrice1", "AskPrice1"]).copy()

    # Keep last snapshot per second
    df = (
        df.sort_values(["SecurityID", "TransactDT_SEC"], kind="mergesort")
        .groupby(["SecurityID", "TransactDT_SEC"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    # Reindex to a full per-second grid per stock and forward-fill.
    # Raw snapshot feeds can skip seconds with no updates; for a fixed 1Hz model we need a dense grid.
    dense_parts: list[pd.DataFrame] = []
    for sid, g in df.groupby("SecurityID", sort=False):
        g = g.sort_values("TransactDT_SEC", kind="mergesort").set_index("TransactDT_SEC")
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="1s")
        g2 = g.reindex(full_idx)
        g2["SecurityID"] = sid
        g2["UpdateTime"] = g2["UpdateTime"].ffill()
        for c in lob_cols:
            g2[c] = g2[c].ffill()
        g2 = g2.dropna(subset=["BidPrice1", "AskPrice1"]).reset_index(names="TransactDT_SEC")
        dense_parts.append(g2)
    df = pd.concat(dense_parts, ignore_index=True)

    # Anchor buckets
    df["BucketStart"] = df["TransactDT_SEC"].dt.floor(f"{int(args.anchor_minutes)}min")
    anchors = (
        df.sort_values(["SecurityID", "BucketStart", "TransactDT_SEC"], kind="mergesort")
        .groupby(["SecurityID", "BucketStart"], as_index=False)
        .head(1)
        .rename(columns={c: f"{c}_A" for c in lob_cols})
    )
    df = df.merge(anchors[["SecurityID", "BucketStart"] + [f"{c}_A" for c in lob_cols]], on=["SecurityID", "BucketStart"], how="left")
    df = df.dropna(subset=[f"BidPrice1_A", f"AskPrice1_A"]).copy()

    # Build deltas (20 slots: ask1..10 then bid1..10)
    ask_p = np.stack([df[f"AskPrice{i}"].to_numpy(np.float64) for i in range(1, levels + 1)], axis=1)
    bid_p = np.stack([df[f"BidPrice{i}"].to_numpy(np.float64) for i in range(1, levels + 1)], axis=1)
    ask_pA = np.stack([df[f"AskPrice{i}_A"].to_numpy(np.float64) for i in range(1, levels + 1)], axis=1)
    bid_pA = np.stack([df[f"BidPrice{i}_A"].to_numpy(np.float64) for i in range(1, levels + 1)], axis=1)

    ask_v = np.stack([df[f"AskVolume{i}"].to_numpy(np.float64) for i in range(1, levels + 1)], axis=1)
    bid_v = np.stack([df[f"BidVolume{i}"].to_numpy(np.float64) for i in range(1, levels + 1)], axis=1)
    ask_vA = np.stack([df[f"AskVolume{i}_A"].to_numpy(np.float64) for i in range(1, levels + 1)], axis=1)
    bid_vA = np.stack([df[f"BidVolume{i}_A"].to_numpy(np.float64) for i in range(1, levels + 1)], axis=1)

    dp_ask = np.rint((ask_p - ask_pA) / tick_size).astype(np.int32)
    dp_bid = np.rint((bid_p - bid_pA) / tick_size).astype(np.int32)
    dp = np.concatenate([dp_ask, dp_bid], axis=1)
    dp = np.clip(dp, -max_tick, +max_tick)
    price_bin = (dp + max_tick).astype(np.int32)  # [N,20] in [0,P-1]

    dv_ask = np.log1p(np.clip(ask_v, 0.0, None)) - np.log1p(np.clip(ask_vA, 0.0, None))
    dv_bid = np.log1p(np.clip(bid_v, 0.0, None)) - np.log1p(np.clip(bid_vA, 0.0, None))
    dv = np.concatenate([dv_ask, dv_bid], axis=1)  # [N,20]

    if reuse_edges is not None:
        edges = reuse_edges
        max_abs = float(max(abs(float(edges[0])), abs(float(edges[-1]))))
    else:
        edges, max_abs = _fit_symmetric_linear_edges(dv.reshape(-1), bins=V, clip_quantile=float(args.vol_clip_quantile))
    vol_bin = _digitize_clip(dv, edges)  # [N,20] in [0,V-1]

    token = price_bin * V + vol_bin
    token = token.astype(np.int32)
    if token.min() < 0 or token.max() >= K:
        raise AssertionError(f"Token out of range: min={int(token.min())} max={int(token.max())} K={K}")

    out_cols = ["SecurityID", "TransactDT_SEC", "BucketStart"]
    if bool(args.keep_raw):
        out_cols += lob_cols
        out_cols += [f"{c}_A" for c in lob_cols]
    out = df[out_cols].copy()
    out["TradeDate"] = day
    for j in range(20):
        out[f"book_token_{j:02d}"] = token[:, j]

    joblib.dump(out, out_job, compress=3)
    meta = {
        "day": day,
        "lob_path": lob_path,
        "stocks": stocks,
        "levels": levels,
        "anchor_minutes": int(args.anchor_minutes),
        "tick_size": tick_size,
        "max_tick": max_tick,
        "P_price_bins": int(P),
        "V_vol_bins": int(V),
        "K_joint_vocab": int(K),
        "vol_edges": edges.tolist(),
        "vol_clip_quantile": float(args.vol_clip_quantile),
        "vol_max_abs_used": float(max_abs),
        "vol_edges_from_meta": edges_from if edges_from else None,
        "output_joblib": out_job,
        "created_at": ts,
        "n_rows": int(len(out)),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[ok] wrote", out_job)
    print("[ok] meta ", out_meta)
    print("[ok] rows", len(out), "K", K)


if __name__ == "__main__":
    main()

