#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a single LOBSTER-style stream from real processed order flow
(no model sampling, no warmup context).

Input semantics expected in processed real-flow dataframe:
- Side == 49: bid post
- Side == 50: ask post
- Side == 99: cancel
- Side == 129: execution
- OrigSide == 49/50 identifies the resting side for cancel/execution rows.
"""

import csv
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

# ============================================================
# PATH SETUP
# ============================================================

UTILITY_DIR = "/finance_ML/zhanghaohan/stock_language_model/utility"
if UTILITY_DIR not in sys.path:
    sys.path.append(UTILITY_DIR)

from sim_helper_unified import (  # noqa: E402
    get_lob_snapshot_by_time,
    init_book_from_snapshot,
)


# ============================================================
# CONFIG
# ============================================================

EXP_NAME = "fixed_start_realflow_generate_lobster"

SELECTED_STOCKS = ["000617_XSHE"]
STOCK = SELECTED_STOCKS[0]

# Processed real-flow orders (canonical runtime path on slurm machine)
DATA_PATH = "/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/processed_real_flow/final_result_for_merge_realflow_20250710_20260402_2219.joblib"

# Snapshot source used to initialize the order book at fixed start time
LOB_SNAP_PATH = "/finance_ML/zhanghaohan/LOB_data/20250710/mdl_6_28_0.csv"

START_TIME_STR = "10:00:00"
SIM_LOOKAHEAD_MINUTES = 10

LOB_LEVELS = 10
BASE_OUT_DIR = "/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream"
TRADE_DATE_STR = "2025-07-10"

# Real-flow encoding constants
SIDE_BID_POST = 49
SIDE_ASK_POST = 50
SIDE_CANCEL = 99
SIDE_EXEC = 129

ORIG_BID = 49
ORIG_ASK = 50


# ============================================================
# HELPERS
# ============================================================

def _price_to_lobster_int(price_float: float) -> int:
    return int(round(float(price_float) * 10000.0))


def _sum_queue(q) -> int:
    if q is None:
        return 0
    return int(sum(q))


def _book_to_lobster_row(book, levels: int = 10):
    asks = sorted(book.asks.keys())
    bids = sorted(book.bids.keys(), reverse=True)

    row = []
    for i in range(levels):
        if i < len(asks):
            ap = asks[i]
            av = _sum_queue(book.asks[ap])
            ap_int = _price_to_lobster_int(ap)
        else:
            ap_int = 9999999999
            av = 0

        if i < len(bids):
            bp = bids[i]
            bv = _sum_queue(book.bids[bp])
            bp_int = _price_to_lobster_int(bp)
        else:
            bp_int = -9999999999
            bv = 0

        row.extend([ap_int, int(av), bp_int, int(bv)])

    return row


def _sec_after_midnight(ts: pd.Timestamp) -> float:
    sod = ts.replace(hour=0, minute=0, second=0, microsecond=0, nanosecond=0)
    return (ts - sod).total_seconds()


def _start_end_ms_from_time_str(time_str: str, lookahead_ms: int):
    t = pd.to_datetime(time_str, format="%H:%M:%S", errors="raise")
    start_ms = int((t.hour * 3600 + t.minute * 60 + t.second) * 1000)
    end_ms = int(start_ms + lookahead_ms)
    return start_ms, end_ms


def _safe_ticker_for_filename(stock: str) -> str:
    return stock.replace("_", "")


def _make_logger(log_path: str):
    def _log(msg: str):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    return _log


def _normalize_snapshot_table(lob_snap_path: str) -> pd.DataFrame:
    lob_snap = pd.read_csv(lob_snap_path)

    columns = list(lob_snap.columns) + ["MISC"]
    lob_info = pd.read_csv(lob_snap_path, header=None, names=columns)
    lob_info = lob_info[1:].copy()

    lob_info["AskPrice1"] = pd.to_numeric(lob_info["AskPrice1"], errors="coerce")
    lob_info["BidPrice1"] = pd.to_numeric(lob_info["BidPrice1"], errors="coerce")
    lob_info = lob_info.dropna(subset=["AskPrice1", "BidPrice1"]).copy()

    lob_info["MidPrice"] = (lob_info["AskPrice1"] + lob_info["BidPrice1"]) / 2
    lob_info["SecurityID"] = lob_info["SecurityID"].astype(str).str.zfill(6) + "_XSHE"

    lob_info["UpdateTime"] = lob_info["UpdateTime"].astype(str)
    lob_info = lob_info[lob_info["UpdateTime"].str.len() >= 8].copy()

    lob_info["TransactDT_SEC"] = pd.to_datetime(
        lob_info["UpdateTime"].str.slice(0, 8),
        format="%H:%M:%S",
        errors="coerce",
    ).dt.floor("s")
    lob_info = lob_info.dropna(subset=["TransactDT_SEC"]).copy()

    return lob_info


def _side_to_resting_book_side(orig_side_val: int):
    if int(orig_side_val) == ORIG_BID:
        return "bid"
    if int(orig_side_val) == ORIG_ASK:
        return "ask"
    return None


def _msg_direction_from_resting(resting_side: str) -> int:
    if resting_side == "ask":
        return -1
    if resting_side == "bid":
        return 1
    return -1


def _exec_direction_from_resting(resting_side: str) -> int:
    # resting ask consumed -> buyer-initiated trade (+1)
    # resting bid consumed -> seller-initiated trade (-1)
    if resting_side == "ask":
        return 1
    if resting_side == "bid":
        return -1
    return -1


def _post_passive(book, side: int, price: float, qty: int):
    if qty <= 0:
        return
    if side == SIDE_BID_POST:
        book.bids[price].append(int(qty))
        return
    if side == SIDE_ASK_POST:
        book.asks[price].append(int(qty))
        return
    raise ValueError(f"_post_passive got unknown side={side}")


def _remove_liquidity(book, event_kind: str, resting_side: str, price: float, qty: int) -> int:
    if event_kind == "cancel":
        return int(book.cancel_passive(resting_side, price, qty))
    if event_kind == "execute":
        return int(book.execute_against_resting(resting_side, price, qty))
    raise ValueError(f"Unknown event_kind={event_kind}")


# ============================================================
# MAIN
# ============================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(BASE_OUT_DIR, f"{EXP_NAME}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    run_log_path = os.path.join(exp_dir, "run.log")
    log = _make_logger(run_log_path)
    log(f"[INFO] Experiment folder: {exp_dir}")

    lookahead_ms = int(SIM_LOOKAHEAD_MINUTES * 60 * 1000)
    start_ts = pd.Timestamp(f"{TRADE_DATE_STR} {START_TIME_STR}")
    end_ts = start_ts + pd.Timedelta(milliseconds=lookahead_ms)

    log("[INFO] Loading and normalizing snapshot table...")
    lob_info = _normalize_snapshot_table(LOB_SNAP_PATH)

    snapshot_df = get_lob_snapshot_by_time(lob_info, START_TIME_STR, STOCK)
    if snapshot_df is None or len(snapshot_df) == 0:
        raise RuntimeError(f"No snapshot found for stock={STOCK} at {START_TIME_STR}")
    snapshot_row = snapshot_df.iloc[0]

    book = init_book_from_snapshot(snapshot_row, max_level=LOB_LEVELS)
    snap_mid = snapshot_row.get("MidPrice", None)
    if snap_mid is None or pd.isna(snap_mid):
        m = book.midpoint()
        cur_mid = 100.0 if m is None else float(m)
    else:
        cur_mid = float(snap_mid)

    log("[INFO] Loading processed real-flow orders...")
    df = joblib.load(DATA_PATH)
    if "SecurityID" not in df.columns:
        raise RuntimeError("Input dataframe missing SecurityID")

    df = df[df["SecurityID"] == STOCK].copy()
    if len(df) == 0:
        raise RuntimeError(f"No rows found for stock={STOCK} in {DATA_PATH}")

    if "TransactDT_MS" not in df.columns:
        raise RuntimeError("Input dataframe missing TransactDT_MS")

    df["TransactDT_MS"] = pd.to_datetime(df["TransactDT_MS"], errors="coerce")
    df = df.dropna(subset=["TransactDT_MS"]).copy()

    # Re-anchor to configured trade date if source timestamps carry 1900-01-01.
    df["event_dt"] = pd.to_datetime(
        TRADE_DATE_STR + " " + df["TransactDT_MS"].dt.strftime("%H:%M:%S.%f"),
        errors="coerce",
    )
    df = df.dropna(subset=["event_dt"]).copy()

    df = df[(df["event_dt"] >= start_ts) & (df["event_dt"] <= end_ts)].copy()
    if len(df) == 0:
        raise RuntimeError(
            f"No events in [{start_ts}, {end_ts}] for stock={STOCK}. "
            "Adjust START_TIME_STR or SIM_LOOKAHEAD_MINUTES."
        )

    sort_cols = [c for c in ["event_dt", "ChannelNo", "ApplSeqNum"] if c in df.columns]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # Robust numeric conversion for event columns used below.
    for col in ["Side", "OrigSide", "OrderQty", "Price"]:
        if col not in df.columns:
            raise RuntimeError(f"Input dataframe missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Side", "OrderQty", "Price"]).copy()
    df["Side"] = df["Side"].astype(int)
    df["OrderQty"] = df["OrderQty"].astype(int)
    df["Price"] = df["Price"].astype(float)
    if "OrigSide" in df.columns:
        df["OrigSide"] = pd.to_numeric(df["OrigSide"], errors="coerce")

    raw_generation_rows = []
    lobster_message_rows = []
    lobster_book_rows = []
    synthetic_order_id = 20_000_000
    unmatched_cancel_events = 0
    unmatched_exec_events = 0

    side_counts = {k: int(v) for k, v in df["Side"].value_counts().to_dict().items()}
    log(f"[INFO] Real events in window: {len(df)} side_counts={side_counts}")

    for idx, row in enumerate(df.itertuples(index=False), start=0):
        event_dt = pd.Timestamp(getattr(row, "event_dt"))
        t_ms = int((event_dt - start_ts).total_seconds() * 1000)

        qty = int(max(0, int(getattr(row, "OrderQty"))))
        side = int(getattr(row, "Side"))
        abs_price = float(np.round(float(getattr(row, "Price")) / 0.01) * 0.01)
        event_kind = "unknown"
        action = "UNKNOWN"
        fills = []
        removed = 0
        resolved_resting_side = None

        # 1) Update book according to real-flow event semantics.
        if side == SIDE_BID_POST and qty > 0:
            event_kind = "post"
            _post_passive(book, SIDE_BID_POST, abs_price, qty)
            action = "POST_BID"

        elif side == SIDE_ASK_POST and qty > 0:
            event_kind = "post"
            _post_passive(book, SIDE_ASK_POST, abs_price, qty)
            action = "POST_ASK"

        elif side in (SIDE_CANCEL, SIDE_EXEC) and qty > 0:
            event_kind = "cancel" if side == SIDE_CANCEL else "execute"
            orig_side_val = getattr(row, "OrigSide", np.nan)
            resting_side = None
            if pd.notna(orig_side_val):
                resting_side = _side_to_resting_book_side(int(orig_side_val))

            if resting_side is None:
                removed = _remove_liquidity(book, event_kind, "ask", abs_price, qty)
                if removed > 0:
                    resting_side = "ask"
                else:
                    removed = _remove_liquidity(book, event_kind, "bid", abs_price, qty)
                    resting_side = "bid"
            else:
                removed = _remove_liquidity(book, event_kind, resting_side, abs_price, qty)

            resolved_resting_side = resting_side

            action = ("CANCEL_" if event_kind == "cancel" else "EXEC_") + str(resting_side).upper()
            if removed <= 0:
                if side == SIDE_CANCEL:
                    unmatched_cancel_events += 1
                else:
                    unmatched_exec_events += 1

        bb = book.best_bid()
        ba = book.best_ask()
        mid_now = book.midpoint()
        if mid_now is not None:
            cur_mid = float(mid_now)

        raw_generation_rows.append(
            {
                "gen_idx": int(idx),
                "t_ms": int(t_ms),
                "event_dt": event_dt.isoformat(),
                "side": int(side),
                "event_kind": event_kind,
                "orig_side": None if pd.isna(getattr(row, "OrigSide", np.nan)) else int(getattr(row, "OrigSide")),
                "resting_side": resolved_resting_side,
                "price": float(abs_price),
                "qty": int(qty),
                "abs_price": float(abs_price),
                "action": action,
                "fills": json.dumps(fills, ensure_ascii=True),
                "removed": int(removed),
                "canceled_qty": int(removed if event_kind == "cancel" else 0),
                "executed_qty": int(removed if event_kind == "execute" else 0),
                "mid_after": None if mid_now is None else float(mid_now),
                "best_bid_after": None if bb is None else float(bb),
                "best_ask_after": None if ba is None else float(ba),
            }
        )

        time_sec = _sec_after_midnight(event_dt)
        price_int = _price_to_lobster_int(abs_price)

        # 2) Emit message rows (LOBSTER schema).
        if side == SIDE_BID_POST and qty > 0:
            synthetic_order_id += 1
            lobster_message_rows.append([float(time_sec), 1, int(synthetic_order_id), int(qty), int(price_int), 1])
            lobster_book_rows.append(_book_to_lobster_row(book, levels=LOB_LEVELS))

        elif side == SIDE_ASK_POST and qty > 0:
            synthetic_order_id += 1
            lobster_message_rows.append([float(time_sec), 1, int(synthetic_order_id), int(qty), int(price_int), -1])
            lobster_book_rows.append(_book_to_lobster_row(book, levels=LOB_LEVELS))

        elif side in (SIDE_CANCEL, SIDE_EXEC) and qty > 0:
            resting_side = resolved_resting_side
            if resting_side is None:
                orig_side_val = getattr(row, "OrigSide", np.nan)
                if pd.notna(orig_side_val):
                    resting_side = _side_to_resting_book_side(int(orig_side_val))
            if resting_side is None:
                resting_side = "ask"

            if event_kind == "cancel":
                # Type 3: delete when full size removed; type 2: partial cancel.
                msg_type = 3 if removed >= qty and removed > 0 else 2
                msg_size = int(removed)
                msg_dir = _msg_direction_from_resting(resting_side)
            else:
                # Type 4 execution; direction follows aggressor implied by resting side.
                msg_type = 4
                msg_size = int(removed)
                msg_dir = _exec_direction_from_resting(resting_side)

            if msg_size > 0:
                synthetic_order_id += 1
                lobster_message_rows.append(
                    [float(time_sec), int(msg_type), int(synthetic_order_id), int(msg_size), int(price_int), int(msg_dir)]
                )
                lobster_book_rows.append(_book_to_lobster_row(book, levels=LOB_LEVELS))

    log(f"[INFO] Exported message rows: {len(lobster_message_rows)}")
    log(f"[INFO] Exported orderbook rows: {len(lobster_book_rows)}")
    log(
        "[INFO] Unmatched remove events "
        f"cancel={unmatched_cancel_events} exec={unmatched_exec_events}"
    )

    raw_csv = os.path.join(exp_dir, "raw_realflow_log.csv")
    pd.DataFrame(raw_generation_rows).to_csv(raw_csv, index=False)

    ticker_for_file = _safe_ticker_for_filename(STOCK)
    start_ms, end_ms = _start_end_ms_from_time_str(START_TIME_STR, lookahead_ms)

    msg_file = os.path.join(
        exp_dir,
        f"{ticker_for_file}_{TRADE_DATE_STR}_{start_ms}_{end_ms}_message_{LOB_LEVELS}.csv",
    )
    ob_file = os.path.join(
        exp_dir,
        f"{ticker_for_file}_{TRADE_DATE_STR}_{start_ms}_{end_ms}_orderbook_{LOB_LEVELS}.csv",
    )

    with open(msg_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(lobster_message_rows)

    with open(ob_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(lobster_book_rows)

    notes = {
        "exp_name": EXP_NAME,
        "timestamp": timestamp,
        "stock": STOCK,
        "start_time": START_TIME_STR,
        "sim_lookahead_minutes": SIM_LOOKAHEAD_MINUTES,
        "sim_lookahead_ms": lookahead_ms,
        "input_event_rows": int(len(df)),
        "lobster_message_rows": int(len(lobster_message_rows)),
        "lobster_orderbook_rows": int(len(lobster_book_rows)),
        "raw_realflow_csv": raw_csv,
        "lobster_message_csv": msg_file,
        "lobster_orderbook_csv": ob_file,
        "paths": {
            "data_path": DATA_PATH,
            "lob_snap_path": LOB_SNAP_PATH,
        },
        "side_mapping": {
            "49": "bid post",
            "50": "ask post",
            "99": "cancel",
            "129": "execution",
            "orig_side_49": "cancel/execution against bid resting order",
            "orig_side_50": "cancel/execution against ask resting order",
        },
        "conversion_caveats": [
            "Order IDs are synthetic for LOBSTER export format compatibility.",
            "Cancel/execution rows with zero matched removal are skipped in LOBSTER message output.",
        ],
        "unmatched_remove_events": {
            "cancel": int(unmatched_cancel_events),
            "exec": int(unmatched_exec_events),
        },
        "run_log": run_log_path,
    }

    notes_json = os.path.join(exp_dir, "generation_notes.json")
    with open(notes_json, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)

    log("[DONE] Fixed-start real-flow export complete.")
    log(f"[DONE] Raw real-flow log: {raw_csv}")
    log(f"[DONE] LOBSTER message CSV: {msg_file}")
    log(f"[DONE] LOBSTER orderbook CSV: {ob_file}")
    log(f"[DONE] Notes JSON: {notes_json}")


if __name__ == "__main__":
    main()
