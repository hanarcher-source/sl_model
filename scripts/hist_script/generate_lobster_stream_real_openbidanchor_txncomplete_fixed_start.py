#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = "/finance_ML/zhanghaohan/stock_language_model"
SCRIPT_DIR = os.path.join(PROJECT_ROOT, "scripts")
HIST_SCRIPT_DIR = os.path.join(SCRIPT_DIR, "hist_script")
UTILITY_DIR = os.path.join(PROJECT_ROOT, "utility")
LOB_BENCH_DIRS = [
    os.path.join(PROJECT_ROOT, "LOB_bench"),
    "/finance_ML/zhanghaohan/lob_bench-main",
]
for path in [PROJECT_ROOT, SCRIPT_DIR, HIST_SCRIPT_DIR, UTILITY_DIR] + LOB_BENCH_DIRS:
    if path not in sys.path:
        sys.path.append(path)

from sim_helper_unified import (  # noqa: E402
    EventDecoded,
    apply_event_to_book_open_anchor_txn_complete,
    get_lob_snapshot_by_time,
    init_book_from_snapshot,
)


LOB_LEVELS = 10
TRADE_DATE_STR = "2025-07-10"
START_TIME_STR = "10:00:00"
SIM_LOOKAHEAD_MINUTES = 10
LOB_SNAP_PATH = "/finance_ML/zhanghaohan/LOB_data/20250710/mdl_6_28_0.csv"
BASE_OUT_DIR = "/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream"


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
    for index in range(levels):
        if index < len(asks):
            ask_price = asks[index]
            ask_volume = _sum_queue(book.asks[ask_price])
            ask_price_int = _price_to_lobster_int(ask_price)
        else:
            ask_price_int = 9999999999
            ask_volume = 0

        if index < len(bids):
            bid_price = bids[index]
            bid_volume = _sum_queue(book.bids[bid_price])
            bid_price_int = _price_to_lobster_int(bid_price)
        else:
            bid_price_int = -9999999999
            bid_volume = 0

        row.extend([ask_price_int, int(ask_volume), bid_price_int, int(bid_volume)])

    return row


def _sec_after_midnight(ts: pd.Timestamp) -> float:
    sod = ts.replace(hour=0, minute=0, second=0, microsecond=0, nanosecond=0)
    return (ts - sod).total_seconds()


def _start_end_ms_from_time_str(time_str: str, end_ms: int):
    t = pd.to_datetime(time_str, format="%H:%M:%S", errors="raise")
    start_ms = int((t.hour * 3600 + t.minute * 60 + t.second) * 1000)
    return start_ms, int(start_ms + end_ms)


def _safe_ticker_for_filename(stock: str) -> str:
    return stock.replace("_", "")


def _make_logger(log_path: str):
    def _log(msg: str):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    return _log


def _normalize_snapshot_table(lob_snap_path: str) -> pd.DataFrame:
    lob_snap = pd.read_csv(lob_snap_path)
    columns = list(lob_snap.columns) + ["MISC"]
    lob_info = pd.read_csv(lob_snap_path, header=None, names=columns)
    lob_info = lob_info[1:].copy()

    numeric_cols = [
        *[f"BidPrice{i}" for i in range(1, 11)],
        *[f"AskPrice{i}" for i in range(1, 11)],
        *[f"BidVolume{i}" for i in range(1, 11)],
        *[f"AskVolume{i}" for i in range(1, 11)],
    ]
    for col in numeric_cols:
        if col in lob_info.columns:
            lob_info[col] = pd.to_numeric(lob_info[col], errors="coerce")

    lob_info = lob_info.dropna(subset=["AskPrice1", "BidPrice1"]).copy()
    lob_info["MidPrice"] = (lob_info["AskPrice1"] + lob_info["BidPrice1"]) / 2
    lob_info["SecurityID"] = lob_info["SecurityID"].astype(str).str.zfill(6) + "_XSHE"
    lob_info["UpdateTime"] = lob_info["UpdateTime"].astype(str)
    lob_info = lob_info[lob_info["UpdateTime"].str.len() >= 8].copy()
    lob_info["TransactDT_SEC"] = pd.to_datetime(
        lob_info["UpdateTime"].str.slice(0, 8), format="%H:%M:%S", errors="coerce"
    ).dt.floor("s")
    lob_info = lob_info.dropna(subset=["TransactDT_SEC"]).copy()
    return lob_info


def _write_outputs(exp_dir, stock, trade_date_str, start_time_str, end_ms, raw_rows, lobster_message_rows, lobster_book_rows):
    raw_csv = os.path.join(exp_dir, "raw_realflow_log.csv")
    pd.DataFrame(raw_rows).to_csv(raw_csv, index=False)

    ticker_for_file = _safe_ticker_for_filename(stock)
    start_ms, end_ts_ms = _start_end_ms_from_time_str(start_time_str, end_ms)
    msg_file = os.path.join(exp_dir, f"{ticker_for_file}_{trade_date_str}_{start_ms}_{end_ts_ms}_message_{LOB_LEVELS}.csv")
    ob_file = os.path.join(exp_dir, f"{ticker_for_file}_{trade_date_str}_{start_ms}_{end_ts_ms}_orderbook_{LOB_LEVELS}.csv")

    with open(msg_file, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(lobster_message_rows)
    with open(ob_file, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(lobster_book_rows)

    return raw_csv, msg_file, ob_file


def main():
    parser = argparse.ArgumentParser(description="Generate a clean fixed-start LOBSTER stream from txn-complete processed real flow.")
    parser.add_argument("--stock", required=True)
    parser.add_argument("--processed-real-flow-path", required=True)
    parser.add_argument("--lob-snap-path", default=LOB_SNAP_PATH)
    parser.add_argument("--trade-date-str", default=TRADE_DATE_STR)
    parser.add_argument("--start-time", default=START_TIME_STR)
    parser.add_argument("--sim-lookahead-minutes", type=int, default=SIM_LOOKAHEAD_MINUTES)
    parser.add_argument("--base-out-dir", default=BASE_OUT_DIR)
    parser.add_argument(
        "--split-cancel-sides",
        action="store_true",
        help="Expect processed realflow with split bid/ask cancel sides (97/98) and 6-way side bins.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stock_tag = args.stock.replace("_", "")
    exp_name_base = (
        "fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_splitcancel"
        if args.split_cancel_sides
        else "fixed_start_realflow_generate_lobster_openbidanchor_txncomplete"
    )
    side_to_bin = (
        {49: 0, 50: 1, 97: 2, 98: 3, 129: 4, 130: 5}
        if args.split_cancel_sides
        else {49: 0, 50: 1, 99: 2, 129: 3, 130: 4}
    )
    exp_dir = os.path.join(args.base_out_dir, f"{exp_name_base}_{stock_tag}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    run_log_path = os.path.join(exp_dir, "run.log")
    log = _make_logger(run_log_path)
    log(f"[INFO] Experiment folder: {exp_dir}")

    lookahead_ms = int(args.sim_lookahead_minutes * 60 * 1000)
    start_ts = pd.Timestamp(f"{args.trade_date_str} {args.start_time}")
    end_ts = start_ts + pd.Timedelta(milliseconds=lookahead_ms)

    log("[INFO] Loading and normalizing snapshot table...")
    lob_info = _normalize_snapshot_table(args.lob_snap_path)
    snapshot_df = get_lob_snapshot_by_time(lob_info, args.start_time, args.stock)
    if snapshot_df is None or len(snapshot_df) == 0:
        raise RuntimeError(f"No snapshot found for stock={args.stock} at {args.start_time}")
    book = init_book_from_snapshot(snapshot_df.iloc[0], max_level=LOB_LEVELS)

    log("[INFO] Loading processed real-flow orders...")
    df = joblib.load(args.processed_real_flow_path)
    df = df[df["SecurityID"] == args.stock].copy()
    if len(df) == 0:
        raise RuntimeError(f"No rows found for stock={args.stock} in {args.processed_real_flow_path}")

    df["TransactDT_MS"] = pd.to_datetime(df["TransactDT_MS"], errors="coerce")
    df = df.dropna(subset=["TransactDT_MS"]).copy()
    df["event_dt"] = pd.to_datetime(
        args.trade_date_str + " " + df["TransactDT_MS"].dt.strftime("%H:%M:%S.%f"),
        errors="coerce",
    )
    df = df.dropna(subset=["event_dt"]).copy()
    df = df[(df["event_dt"] >= start_ts) & (df["event_dt"] <= end_ts)].copy()
    if len(df) == 0:
        raise RuntimeError(f"No events in [{start_ts}, {end_ts}] for stock={args.stock}")

    sort_cols = [c for c in ["event_dt", "ChannelNo", "ApplSeqNum"] if c in df.columns]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    token_df = df[df["tokenizable_event"].fillna(False).astype(bool)].copy()
    if len(token_df) == 0:
        raise RuntimeError("No tokenizable rows found in selected window.")

    log(f"[INFO] Window rows={len(df)} tokenizable_rows={len(token_df)}")
    log(f"[INFO] Tokenizable side counts={token_df['Side'].value_counts().to_dict()}")

    raw_generation_rows = []
    lobster_message_rows = []
    lobster_book_rows = []
    synthetic_order_id = 50_000_000
    rejected_crossing_posts = 0
    zero_remove_events = 0

    for idx, row in enumerate(token_df.itertuples(index=False), start=0):
        side = int(getattr(row, "Side"))
        if side not in side_to_bin:
            continue

        event_dt = pd.Timestamp(getattr(row, "event_dt"))
        t_ms = int((event_dt - start_ts).total_seconds() * 1000)
        abs_price = float(np.round(float(getattr(row, "Price")) / 0.01) * 0.01)
        qty = int(max(0, int(getattr(row, "OrderQty"))))
        ticks_from_anchor = None if pd.isna(getattr(row, "Price_OpenBid_diff", np.nan)) else int(getattr(row, "Price_OpenBid_diff"))
        if ticks_from_anchor is None:
            ticks_from_anchor = 0

        ev = EventDecoded(
            t_ms=int(t_ms),
            side_bin=int(side_to_bin[side]),
            ticks_from_mid=int(ticks_from_anchor),
            qty=int(qty),
            abs_price=float(abs_price),
        )

        action = apply_event_to_book_open_anchor_txn_complete(
            book, ev, split_cancel_sides=bool(args.split_cancel_sides)
        )
        if action.get("rejected", False):
            rejected_crossing_posts += 1
        if int(action.get("removed", 0)) <= 0 and action.get("event_kind") in ("cancel", "transaction_complete"):
            zero_remove_events += 1

        raw_generation_rows.append(
            {
                "gen_idx": int(idx),
                "event_dt": event_dt.isoformat(),
                "t_ms": int(t_ms),
                "side": int(side),
                "event_semantic": getattr(row, "EventSemantic", None),
                "orig_side": None if pd.isna(getattr(row, "OrigSide", np.nan)) else int(getattr(row, "OrigSide")),
                "price": float(abs_price),
                "qty": int(qty),
                "price_openbid_diff": int(ticks_from_anchor),
                "action": action.get("action"),
                "event_kind": action.get("event_kind"),
                "resting_side": action.get("resting_side"),
                "removed": int(action.get("removed", 0)),
                "rejected": bool(action.get("rejected", False)),
                "fill_qty": int(sum(int(fill.get("filled_qty", 0)) for fill in action.get("fills", []))),
                "fills": json.dumps(action.get("fills", []), ensure_ascii=True),
                "best_bid_after": None if book.best_bid() is None else float(book.best_bid()),
                "best_ask_after": None if book.best_ask() is None else float(book.best_ask()),
            }
        )

        time_sec = _sec_after_midnight(event_dt)
        event_kind = action.get("event_kind", "")
        action_name = action.get("action", "")
        price_int = _price_to_lobster_int(abs_price)

        msg_type = None
        msg_size = None
        msg_direction = None

        if event_kind == "post" and not action.get("rejected", False) and action_name == "POST_BID":
            msg_type = 1
            msg_size = int(qty)
            msg_direction = 1
        elif event_kind == "post" and not action.get("rejected", False) and action_name == "POST_ASK":
            msg_type = 1
            msg_size = int(qty)
            msg_direction = -1
        elif event_kind == "cancel":
            removed = int(action.get("removed", 0))
            if removed > 0:
                msg_type = 3 if removed >= int(qty) else 2
                msg_size = removed
                msg_direction = -1 if action.get("resting_side") == "ask" else 1

        if msg_type is not None and msg_size is not None and msg_size > 0 and msg_direction is not None:
            synthetic_order_id += 1
            lobster_message_rows.append([
                float(time_sec), int(msg_type), int(synthetic_order_id), int(msg_size), int(price_int), int(msg_direction)
            ])
            lobster_book_rows.append(_book_to_lobster_row(book, levels=LOB_LEVELS))

        for fill in action.get("fills", []):
            fill_qty = int(fill.get("filled_qty", 0))
            fill_price = float(fill.get("price", abs_price))
            if fill_qty <= 0:
                continue
            resting_side = fill.get("side", "")
            fill_dir = 1 if resting_side == "ask" else -1
            synthetic_order_id += 1
            lobster_message_rows.append([
                float(time_sec), 4, int(synthetic_order_id), int(fill_qty), int(_price_to_lobster_int(fill_price)), int(fill_dir)
            ])
            lobster_book_rows.append(_book_to_lobster_row(book, levels=LOB_LEVELS))

    raw_csv, msg_file, ob_file = _write_outputs(
        exp_dir,
        args.stock,
        args.trade_date_str,
        args.start_time,
        lookahead_ms,
        raw_generation_rows,
        lobster_message_rows,
        lobster_book_rows,
    )

    side_mapping_notes = (
        {
            "49": "passive bid post",
            "50": "passive ask post",
            "97": "cancel resting bid",
            "98": "cancel resting ask",
            "129": "transaction complete against resting bid",
            "130": "transaction complete against resting ask",
        }
        if args.split_cancel_sides
        else {
            "49": "passive bid post",
            "50": "passive ask post",
            "99": "cancel",
            "129": "transaction complete against resting bid",
            "130": "transaction complete against resting ask",
        }
    )

    notes = {
        "exp_name": exp_name_base,
        "split_cancel_sides": bool(args.split_cancel_sides),
        "timestamp": timestamp,
        "stock": args.stock,
        "start_time": args.start_time,
        "trade_date_str": args.trade_date_str,
        "sim_lookahead_minutes": int(args.sim_lookahead_minutes),
        "window_input_rows": int(len(df)),
        "tokenizable_rows": int(len(token_df)),
        "rejected_crossing_posts": int(rejected_crossing_posts),
        "zero_remove_events": int(zero_remove_events),
        "lobster_message_rows": int(len(lobster_message_rows)),
        "lobster_orderbook_rows": int(len(lobster_book_rows)),
        "raw_realflow_csv": raw_csv,
        "lobster_message_csv": msg_file,
        "lobster_orderbook_csv": ob_file,
        "paths": {
            "processed_real_flow_path": args.processed_real_flow_path,
            "lob_snap_path": args.lob_snap_path,
        },
        "side_mapping": side_mapping_notes,
        "conversion_caveats": (
            [
                "Uses the txn-complete processed real-flow schema directly.",
                "Passive crossing posts are rejected instead of exported as intermediate crossed book states.",
                "Transaction-complete rows remove liquidity directly from the specified resting side.",
                "Price, quantity, and timestamps are taken exactly from the processed real-flow rows.",
                "Order IDs are synthetic for LOBSTER export compatibility.",
            ]
            + (
                [
                    "Split-cancel mode: cancels use distinct resting sides (97=bid, 98=ask) and vocab factor n_side=6.",
                ]
                if args.split_cancel_sides
                else []
            )
        ),
        "run_log": run_log_path,
    }

    notes_json = os.path.join(exp_dir, "generation_notes.json")
    with open(notes_json, "w", encoding="utf-8") as fh:
        json.dump(notes, fh, indent=2)

    log(f"[INFO] Exported message rows: {len(lobster_message_rows)}")
    log(f"[INFO] Exported orderbook rows: {len(lobster_book_rows)}")
    log(f"[INFO] Rejected crossing posts: {rejected_crossing_posts}")
    log(f"[INFO] Zero-remove cancel/exec events: {zero_remove_events}")
    log("[DONE] Clean fixed-start real-flow export complete.")
    log(f"[DONE] Raw real-flow log: {raw_csv}")
    log(f"[DONE] LOBSTER message CSV: {msg_file}")
    log(f"[DONE] LOBSTER orderbook CSV: {ob_file}")
    log(f"[DONE] Notes JSON: {notes_json}")


if __name__ == "__main__":
    main()