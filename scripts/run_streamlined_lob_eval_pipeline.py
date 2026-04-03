#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run the full fixed-start LOB evaluation pipeline with one command.

Required user input:
    1) checkpoint path for the generation model

Optional user input:
    2) processed real-flow dataframe path, to skip rebuilding it

Hardcoded runtime configuration for the current workflow:
    - day: 20250710
    - stock: 000617_XSHE
    - start time: 10:00:00
    - warmup order count: 50
    - lookahead window: 10 minutes

Pipeline behavior:
    - Reuse cached processed real-flow data when available.
    - Reuse an existing fixed-start real-flow LOBSTER stream when available.
    - Reuse an existing fixed-start generated LOBSTER stream for the given
      checkpoint when available.
    - Run evaluation every time so the latest metrics logs are produced.
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = "/finance_ML/zhanghaohan/stock_language_model"
SCRIPT_DIR = os.path.join(PROJECT_ROOT, "scripts")
HIST_SCRIPT_DIR = os.path.join(SCRIPT_DIR, "hist_script")
UTILITY_DIR = os.path.join(PROJECT_ROOT, "utility")
for path in [PROJECT_ROOT, SCRIPT_DIR, HIST_SCRIPT_DIR, UTILITY_DIR]:
    if path not in sys.path:
        sys.path.append(path)

from eval_generated_stream import (  # noqa: E402
    _log_all_metrics_summary,
    _setup_logger,
    _to_serializable,
    compute_metrics,
    load_lobster_pair,
    make_plots,
)
from sim_helper_unified import (  # noqa: E402
    apply_event_to_book,
    build_model,
    decode_event_from_token,
    get_lob_snapshot_by_time,
    get_order_window_ending_at_second,
    init_book_from_snapshot,
    load_bin_record,
    load_checkpoint,
    process_lob_data_real_flow,
    sample_next_token,
)


# ============================================================
# HARD-CODED WORKFLOW CONFIG
# ============================================================

SEED = 42
SAMPLE_SEED = 1234
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DAY = "20250710"
TRADE_DATE_STR = "2025-07-10"
STOCK = "000617_XSHE"
SELECTED_STOCKS = [STOCK]

START_TIME_STR = "10:00:00"
WARMUP_ORDER_NUM = 50
WINDOW_LEN = 50
SIM_LOOKAHEAD_MINUTES = 10
SIM_LOOKAHEAD_MS = SIM_LOOKAHEAD_MINUTES * 60 * 1000
MAX_GENERATED_STEPS = 20000
LOG_EVERY_STEPS = 200
LOB_LEVELS = 10

MODEL_VARIANT = "no_anchor"
ANCHOR_COUNT = 128
PRICE_BIN_NUM = 26
QTY_BIN_NUM = 26
INTERVAL_BIN_NUM = 12
N_SIDE = 3
VOCAB_SIZE = 24336
USE_SAMPLING = True
TEMPERATURE = 1.3
TOP_P = 0.98

BASE_OUT_DIR = os.path.join(PROJECT_ROOT, "saved_LOB_stream")
PROCESSED_REALFLOW_DIR = os.path.join(BASE_OUT_DIR, "processed_real_flow")

LOB_DATA_ROOT = "/finance_ML/zhanghaohan/LOB_data"
LOB_DAY_DIR = os.path.join(LOB_DATA_ROOT, DAY)
MODEL_DATA_PATH = os.path.join(
    LOB_DAY_DIR,
    "final_result_for_merge_vocab24336_multidaypool(03_10)_samp_20260318_1743.joblib",
)
BIN_RECORD_PATH = os.path.join(
    LOB_DAY_DIR,
    "bin_record_vocab24336_multidaypool(03_10)_samp_20260318_1743.json",
)
LOB_SNAP_PATH = os.path.join(LOB_DAY_DIR, "mdl_6_28_0.csv")
ORDER_POST_PATH = os.path.join(LOB_DAY_DIR, "mdl_6_33_0.csv")
ORDER_TRANSAC_PATH = os.path.join(LOB_DAY_DIR, "mdl_6_36_0.csv")
LIQUIDITY_MASK_PATH = os.path.join(
    LOB_DATA_ROOT,
    "misc_data",
    "AVG_AMT_3M_7_1D_8390e8742c5e.csv",
)

REALFLOW_EXP_NAME = "fixed_start_realflow_generate_lobster"

SIDE_BID_POST = 49
SIDE_ASK_POST = 50
SIDE_CANCEL = 99
SIDE_EXEC = 129
ORIG_BID = 49
ORIG_ASK = 50


# ============================================================
# REPRO
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# SMALL HELPERS
# ============================================================

def _make_line_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def _log(msg: str):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    return _log


def _safe_ticker_for_filename(stock: str) -> str:
    return stock.replace("_", "")


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


def _sanitize_checkpoint_stem(ckpt_path: str) -> str:
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return stem[:80] if stem else "model"


def _read_notes_if_valid(exp_dir: str):
    notes_path = os.path.join(exp_dir, "generation_notes.json")
    if not os.path.isfile(notes_path):
        return None
    try:
        with open(notes_path, "r", encoding="utf-8") as fh:
            notes = json.load(fh)
    except Exception:
        return None

    msg_path = notes.get("lobster_message_csv")
    ob_path = notes.get("lobster_orderbook_csv")
    if not msg_path or not ob_path:
        return None
    if not (os.path.isfile(msg_path) and os.path.isfile(ob_path)):
        return None
    return notes


def _same_path(a, b) -> bool:
    if a is None or b is None:
        return False
    return os.path.abspath(a) == os.path.abspath(b)


def _iter_generation_dirs(pattern: str):
    for exp_dir in sorted(glob.glob(os.path.join(BASE_OUT_DIR, pattern))):
        if os.path.isdir(exp_dir):
            yield exp_dir


def _find_matching_processed_realflow(day: str):
    pattern = os.path.join(PROCESSED_REALFLOW_DIR, f"final_result_for_merge_realflow_{day}_*.joblib")
    candidates = sorted(glob.glob(pattern))
    return candidates[-1] if candidates else None


def _find_matching_realflow_exp(processed_real_flow_path: str):
    for exp_dir in reversed(list(_iter_generation_dirs(f"{REALFLOW_EXP_NAME}_*"))):
        notes = _read_notes_if_valid(exp_dir)
        if notes is None:
            continue
        if notes.get("stock") != STOCK:
            continue
        if notes.get("start_time") != START_TIME_STR:
            continue
        if int(notes.get("sim_lookahead_minutes", -1)) != SIM_LOOKAHEAD_MINUTES:
            continue
        note_paths = notes.get("paths", {})
        if not _same_path(note_paths.get("data_path"), processed_real_flow_path):
            continue
        if not _same_path(note_paths.get("lob_snap_path"), LOB_SNAP_PATH):
            continue
        return exp_dir
    return None


def _find_matching_generated_exp(ckpt_path: str):
    for exp_dir in reversed(list(_iter_generation_dirs("fixed_start_*_generate_lobster_*"))):
        notes = _read_notes_if_valid(exp_dir)
        if notes is None:
            continue
        note_paths = notes.get("paths", {})
        if not note_paths.get("ckpt_path"):
            continue
        if notes.get("stock") != STOCK:
            continue
        if notes.get("start_time") != START_TIME_STR:
            continue
        if int(notes.get("sim_lookahead_minutes", -1)) != SIM_LOOKAHEAD_MINUTES:
            continue
        if not _same_path(note_paths.get("ckpt_path"), ckpt_path):
            continue
        if not _same_path(note_paths.get("data_path"), MODEL_DATA_PATH):
            continue
        if not _same_path(note_paths.get("bin_record_path"), BIN_RECORD_PATH):
            continue
        if not _same_path(note_paths.get("lob_snap_path"), LOB_SNAP_PATH):
            continue
        if int(notes.get("warmup_order_num", WARMUP_ORDER_NUM)) != WARMUP_ORDER_NUM:
            continue
        if int(notes.get("window_len", WINDOW_LEN)) != WINDOW_LEN:
            continue
        return exp_dir
    return None


def _write_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_to_serializable(payload), fh, indent=2)


# ============================================================
# REAL-FLOW PREPROCESS
# ============================================================

def _build_processed_realflow_cache(output_dir: str, log):
    ts = time.strftime("%Y%m%d_%H%M")
    os.makedirs(output_dir, exist_ok=True)

    log(f"[INFO] Building processed real-flow cache for day={DAY}")
    df_day = process_lob_data_real_flow(
        order_post_dir=ORDER_POST_PATH,
        lob_snap_dir=LOB_SNAP_PATH,
        order_transac_dir=ORDER_TRANSAC_PATH,
        liquidity_mask_dir=LIQUIDITY_MASK_PATH,
        selected_stocks=SELECTED_STOCKS,
        filter_bo=True,
        date_num_str=DAY,
    )
    df_day = df_day.copy()
    df_day["TradeDate"] = DAY

    cache_path = os.path.join(output_dir, f"final_result_for_merge_realflow_{DAY}_{ts}.joblib")
    summary_path = os.path.join(output_dir, f"final_result_for_merge_realflow_{DAY}_{ts}.json")

    joblib.dump(df_day, cache_path, compress=3)
    summary = {
        "day": DAY,
        "created_at": ts,
        "rows": int(len(df_day)),
        "stocks": int(df_day["SecurityID"].nunique()) if len(df_day) > 0 else 0,
        "columns": list(df_day.columns),
        "source": {
            "order_post": ORDER_POST_PATH,
            "lob_snap": LOB_SNAP_PATH,
            "order_transac": ORDER_TRANSAC_PATH,
            "liquidity_mask": LIQUIDITY_MASK_PATH,
        },
        "output": {
            "joblib": cache_path,
        },
    }
    _write_json(summary_path, summary)
    log(f"[DONE] Processed real-flow cache -> {cache_path}")
    return cache_path


# ============================================================
# REAL-FLOW STREAM GENERATION
# ============================================================

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
    raise ValueError(f"Unknown post side={side}")


def _remove_liquidity(book, event_kind: str, resting_side: str, price: float, qty: int) -> int:
    if event_kind == "cancel":
        return int(book.cancel_passive(resting_side, price, qty))
    if event_kind == "execute":
        return int(book.execute_against_resting(resting_side, price, qty))
    raise ValueError(f"Unknown event_kind={event_kind}")


def _generate_realflow_stream(processed_real_flow_path: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(BASE_OUT_DIR, f"{REALFLOW_EXP_NAME}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    run_log_path = os.path.join(exp_dir, "run.log")
    log = _make_line_logger(run_log_path)
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

    log("[INFO] Loading processed real-flow orders...")
    df = joblib.load(processed_real_flow_path)
    if "SecurityID" not in df.columns:
        raise RuntimeError("Input dataframe missing SecurityID")
    if "TransactDT_MS" not in df.columns:
        raise RuntimeError("Input dataframe missing TransactDT_MS")

    df = df[df["SecurityID"] == STOCK].copy()
    if len(df) == 0:
        raise RuntimeError(f"No rows found for stock={STOCK} in {processed_real_flow_path}")

    df["TransactDT_MS"] = pd.to_datetime(df["TransactDT_MS"], errors="coerce")
    df = df.dropna(subset=["TransactDT_MS"]).copy()
    df["event_dt"] = pd.to_datetime(
        TRADE_DATE_STR + " " + df["TransactDT_MS"].dt.strftime("%H:%M:%S.%f"),
        errors="coerce",
    )
    df = df.dropna(subset=["event_dt"]).copy()
    df = df[(df["event_dt"] >= start_ts) & (df["event_dt"] <= end_ts)].copy()
    if len(df) == 0:
        raise RuntimeError(
            f"No events in [{start_ts}, {end_ts}] for stock={STOCK}. "
            "Adjust the fixed pipeline window."
        )

    sort_cols = [c for c in ["event_dt", "ChannelNo", "ApplSeqNum"] if c in df.columns]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

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

    raw_rows = []
    lobster_message_rows = []
    lobster_book_rows = []
    synthetic_order_id = 20_000_000
    unmatched_cancel_events = 0
    unmatched_exec_events = 0

    side_counts = {str(k): int(v) for k, v in df["Side"].value_counts().to_dict().items()}
    log(f"[INFO] Real events in window: {len(df)} side_counts={side_counts}")

    for idx, row in enumerate(df.itertuples(index=False), start=0):
        event_dt = pd.Timestamp(getattr(row, "event_dt"))
        t_ms = int((event_dt - start_ts).total_seconds() * 1000)

        qty = int(max(0, int(getattr(row, "OrderQty"))))
        side = int(getattr(row, "Side"))
        abs_price = float(np.round(float(getattr(row, "Price")) / 0.01) * 0.01)
        event_kind = "unknown"
        action = "UNKNOWN"
        removed = 0
        resolved_resting_side = None

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
        raw_rows.append(
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
                "fills": json.dumps([], ensure_ascii=True),
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
                msg_type = 3 if removed >= qty and removed > 0 else 2
                msg_size = int(removed)
                msg_dir = _msg_direction_from_resting(resting_side)
            else:
                msg_type = 4
                msg_size = int(removed)
                msg_dir = _exec_direction_from_resting(resting_side)

            if msg_size > 0:
                synthetic_order_id += 1
                lobster_message_rows.append(
                    [float(time_sec), int(msg_type), int(synthetic_order_id), int(msg_size), int(price_int), int(msg_dir)]
                )
                lobster_book_rows.append(_book_to_lobster_row(book, levels=LOB_LEVELS))

    raw_csv = os.path.join(exp_dir, "raw_realflow_log.csv")
    pd.DataFrame(raw_rows).to_csv(raw_csv, index=False)

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

    with open(msg_file, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(lobster_message_rows)
    with open(ob_file, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(lobster_book_rows)

    notes = {
        "exp_name": REALFLOW_EXP_NAME,
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
            "data_path": processed_real_flow_path,
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
    _write_json(os.path.join(exp_dir, "generation_notes.json"), notes)

    log("[DONE] Fixed-start real-flow export complete.")
    log(f"[DONE] Raw real-flow log: {raw_csv}")
    log(f"[DONE] LOBSTER message CSV: {msg_file}")
    log(f"[DONE] LOBSTER orderbook CSV: {ob_file}")
    return exp_dir


# ============================================================
# GENERATED STREAM
# ============================================================

def _generate_model_stream(ckpt_path: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_tag = _sanitize_checkpoint_stem(ckpt_path)
    exp_name = f"fixed_start_617_{ckpt_tag}_generate_lobster"
    exp_dir = os.path.join(BASE_OUT_DIR, f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    run_log_path = os.path.join(exp_dir, "run.log")
    log = _make_line_logger(run_log_path)
    log(f"[INFO] Experiment folder: {exp_dir}")

    log("[INFO] Loading snapshot table...")
    lob_info = _normalize_snapshot_table(LOB_SNAP_PATH)

    log("[INFO] Loading processed model input data...")
    processed_lob_data = joblib.load(MODEL_DATA_PATH)
    processed_lob_data = processed_lob_data[
        processed_lob_data["SecurityID"].isin(SELECTED_STOCKS)
    ].copy()
    log(f"[INFO] processed_lob_data filtered rows: {len(processed_lob_data)}")

    snapshot_df = get_lob_snapshot_by_time(lob_info, START_TIME_STR, STOCK)
    if snapshot_df is None or len(snapshot_df) == 0:
        raise RuntimeError(f"No snapshot found for stock={STOCK} at {START_TIME_STR}")
    snapshot_row = snapshot_df.iloc[0]

    window_df = get_order_window_ending_at_second(
        processed_LOB_data=processed_lob_data,
        target_time=START_TIME_STR,
        stock=STOCK,
        order_num=WARMUP_ORDER_NUM,
    )
    if window_df is None or len(window_df) < WARMUP_ORDER_NUM:
        raise RuntimeError(
            f"Warmup window insufficient. Need {WARMUP_ORDER_NUM}, got {0 if window_df is None else len(window_df)}"
        )

    log("[INFO] Loading bin record...")
    binpack = load_bin_record(BIN_RECORD_PATH)

    log("[INFO] Building model...")
    model = build_model(
        model_variant=MODEL_VARIANT,
        vocab_size=VOCAB_SIZE,
        gpt2_name="gpt2",
        anchor_count=ANCHOR_COUNT,
    )
    _ = load_checkpoint(model, ckpt_path, DEVICE)

    book = init_book_from_snapshot(snapshot_row, max_level=LOB_LEVELS)
    snap_mid = snapshot_row.get("MidPrice", None)
    if snap_mid is None or pd.isna(snap_mid):
        midpoint = book.midpoint()
        cur_mid = 100.0 if midpoint is None else float(midpoint)
    else:
        cur_mid = float(snap_mid)

    tokens = window_df["order_token"].astype(int).to_numpy()
    if len(tokens) < WINDOW_LEN:
        raise RuntimeError(f"Need at least {WINDOW_LEN} context tokens, got {len(tokens)}")
    context = tokens[-WINDOW_LEN:].tolist()

    sample_gen = torch.Generator()
    sample_gen.manual_seed(SAMPLE_SEED)
    decode_rng = np.random.default_rng(43)
    stream_start_dt = pd.Timestamp(f"{TRADE_DATE_STR} {START_TIME_STR}")

    log("[INFO] Starting fixed-start generation...")
    log(f"[INFO] stock={STOCK} start_time={START_TIME_STR} lookahead_min={SIM_LOOKAHEAD_MINUTES}")

    cur_t_ms = 0
    step = 0
    raw_rows = []
    lobster_message_rows = []
    lobster_book_rows = []
    synthetic_order_id = 10_000_000

    while cur_t_ms < SIM_LOOKAHEAD_MS and step < MAX_GENERATED_STEPS:
        ctx = torch.tensor(context[-WINDOW_LEN:], dtype=torch.long, device=DEVICE).unsqueeze(0)

        tok_next = sample_next_token(
            model,
            ctx,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            use_sampling=USE_SAMPLING,
            sample_gen=sample_gen,
        )

        ev, dt_ms = decode_event_from_token(
            tok_next,
            binpack,
            cur_mid,
            cur_t_ms,
            price_bin_num=PRICE_BIN_NUM,
            qty_bin_num=QTY_BIN_NUM,
            interval_bin_num=INTERVAL_BIN_NUM,
            n_side=N_SIDE,
            decode_method="sample",
            rng=decode_rng,
        )

        cur_t_ms = int(ev.t_ms)
        action = apply_event_to_book(book, ev)
        midpoint = book.midpoint()
        if midpoint is not None:
            cur_mid = float(midpoint)

        context.append(int(tok_next))
        fills = action.get("fills", [])

        raw_rows.append(
            {
                "gen_idx": int(step),
                "t_ms": int(cur_t_ms),
                "token": int(tok_next),
                "side_bin": int(ev.side_bin),
                "event_kind": action.get("event_kind", None),
                "resting_side": action.get("resting_side", None),
                "ticks_from_mid": int(ev.ticks_from_mid),
                "qty": int(ev.qty),
                "abs_price": float(ev.abs_price),
                "action": action.get("action", None),
                "fills": json.dumps(fills, ensure_ascii=True),
                "removed": int(action.get("removed", 0)),
                "fill_qty": int(sum(int(fill.get("filled_qty", 0)) for fill in fills)),
                "dt_ms": int(dt_ms),
                "mid_after": None if cur_mid is None else float(cur_mid),
                "best_bid_after": None if book.best_bid() is None else float(book.best_bid()),
                "best_ask_after": None if book.best_ask() is None else float(book.best_ask()),
            }
        )

        event_ts = stream_start_dt + pd.Timedelta(milliseconds=cur_t_ms)
        time_sec = _sec_after_midnight(event_ts)
        action_name = action.get("action", "")
        event_kind = action.get("event_kind", "")
        price_int = _price_to_lobster_int(ev.abs_price)

        msg_type = None
        msg_size = None
        msg_direction = None
        if event_kind == "post" and action_name == "POST_BID":
            msg_type = 1
            msg_size = int(ev.qty)
            msg_direction = 1
        elif event_kind == "post" and action_name == "POST_ASK":
            msg_type = 1
            msg_size = int(ev.qty)
            msg_direction = -1
        elif event_kind == "cancel":
            removed = int(action.get("removed", 0))
            if removed > 0:
                msg_type = 3 if removed >= int(ev.qty) else 2
                msg_size = removed
                if "ASK" in action_name:
                    msg_direction = -1
                elif "BID" in action_name:
                    msg_direction = 1
                else:
                    msg_direction = -1

        if msg_type is not None and msg_size is not None and msg_size > 0 and msg_direction is not None:
            synthetic_order_id += 1
            lobster_message_rows.append(
                [
                    float(time_sec),
                    int(msg_type),
                    int(synthetic_order_id),
                    int(msg_size),
                    int(price_int),
                    int(msg_direction),
                ]
            )
            lobster_book_rows.append(_book_to_lobster_row(book, levels=LOB_LEVELS))

        for fill in fills:
            fill_qty = int(fill.get("filled_qty", 0))
            fill_price = float(fill.get("price", ev.abs_price))
            if fill_qty <= 0:
                continue
            resting_side = fill.get("side", "")
            if resting_side == "ask":
                fill_dir = 1
            elif resting_side == "bid":
                fill_dir = -1
            else:
                fill_dir = -1

            synthetic_order_id += 1
            lobster_message_rows.append(
                [
                    float(time_sec),
                    4,
                    int(synthetic_order_id),
                    int(fill_qty),
                    int(_price_to_lobster_int(fill_price)),
                    int(fill_dir),
                ]
            )
            lobster_book_rows.append(_book_to_lobster_row(book, levels=LOB_LEVELS))

        step += 1
        if step % LOG_EVERY_STEPS == 0:
            progress = 100.0 * float(cur_t_ms) / float(max(SIM_LOOKAHEAD_MS, 1))
            log(
                "[PROGRESS] "
                f"step={step} "
                f"sim_t_ms={cur_t_ms}/{SIM_LOOKAHEAD_MS} "
                f"progress={progress:.2f}% "
                f"messages={len(lobster_message_rows)}"
            )

    raw_csv = os.path.join(exp_dir, "raw_generation_log.csv")
    pd.DataFrame(raw_rows).to_csv(raw_csv, index=False)

    ticker_for_file = _safe_ticker_for_filename(STOCK)
    start_ms, end_ms = _start_end_ms_from_time_str(START_TIME_STR, SIM_LOOKAHEAD_MS)
    msg_file = os.path.join(
        exp_dir,
        f"{ticker_for_file}_{TRADE_DATE_STR}_{start_ms}_{end_ms}_message_{LOB_LEVELS}.csv",
    )
    ob_file = os.path.join(
        exp_dir,
        f"{ticker_for_file}_{TRADE_DATE_STR}_{start_ms}_{end_ms}_orderbook_{LOB_LEVELS}.csv",
    )

    with open(msg_file, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(lobster_message_rows)
    with open(ob_file, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(lobster_book_rows)

    notes = {
        "exp_name": exp_name,
        "timestamp": timestamp,
        "stock": STOCK,
        "model_variant": MODEL_VARIANT,
        "start_time": START_TIME_STR,
        "warmup_order_num": WARMUP_ORDER_NUM,
        "window_len": WINDOW_LEN,
        "sim_lookahead_minutes": SIM_LOOKAHEAD_MINUTES,
        "sim_lookahead_ms": SIM_LOOKAHEAD_MS,
        "generated_steps": int(step),
        "lobster_message_rows": int(len(lobster_message_rows)),
        "lobster_orderbook_rows": int(len(lobster_book_rows)),
        "raw_generation_csv": raw_csv,
        "lobster_message_csv": msg_file,
        "lobster_orderbook_csv": ob_file,
        "paths": {
            "data_path": MODEL_DATA_PATH,
            "bin_record_path": BIN_RECORD_PATH,
            "lob_snap_path": LOB_SNAP_PATH,
            "ckpt_path": os.path.abspath(ckpt_path),
        },
        "conversion_caveats": [
            "Order IDs are synthetic; true exchange-level ID linkage is not preserved in simulator state.",
            "Execution rows are emitted from simulator fill logs; multiple messages can share the same timestamp.",
            "Book rows are emitted per output message for LOBSTER shape consistency.",
        ],
        "run_log": run_log_path,
    }
    _write_json(os.path.join(exp_dir, "generation_notes.json"), notes)

    log("[DONE] Fixed-start generation complete.")
    log(f"[DONE] Raw generation log: {raw_csv}")
    log(f"[DONE] LOBSTER message CSV: {msg_file}")
    log(f"[DONE] LOBSTER orderbook CSV: {ob_file}")
    return exp_dir


# ============================================================
# EVALUATION
# ============================================================

def _evaluate_stream(exp_dir: str, label: str, real_ref_dir: str = None):
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    log_file = os.path.join(exp_dir, "eval.log")
    logger = _setup_logger(log_file)

    logger.info("=" * 70)
    logger.info("%s — stylized fact evaluation", label)
    logger.info("=" * 70)
    logger.info("Evaluating: %s", exp_dir)

    logger.info("\n── loading data ───────────────────────────────────────")
    messages, book, notes = load_lobster_pair(exp_dir, logger)

    ref_messages = None
    ref_book = None
    if real_ref_dir:
        logger.info("Loading real reference LOBSTER data from %s", real_ref_dir)
        ref_messages, ref_book, _ = load_lobster_pair(real_ref_dir, logger)

    logger.info("\n── computing metrics ──────────────────────────────────")
    metrics = compute_metrics(messages, book, logger, ref_messages=ref_messages, ref_book=ref_book)

    logger.info("\n── capability split (strict) ─────────────────────────")
    for name in metrics["capability_report"]["computed_generated_only_metrics"]:
        logger.info("  [computed] %s", name)
    for name in metrics["capability_report"]["computed_reference_comparison_metrics"]:
        logger.info("  [computed: real-vs-generated] %s", name)
    for name in metrics["capability_report"]["not_attempted_requires_real_reference_lob_data"]:
        logger.info("  [skipped: needs real reference LOB data] %s", name)
    for name, reason in metrics["capability_report"].get("failed_reference_comparison_metrics", {}).items():
        logger.info("  [failed: real-vs-generated] %s | %s", name, reason)
    for name in metrics["capability_report"]["not_attempted_requires_true_order_id_linkage"]:
        logger.info("  [skipped: needs true order-ID linkage] %s", name)

    logger.info("\n── generating plots ───────────────────────────────────")
    make_plots(messages, book, plots_dir, notes, logger)

    summary_path = os.path.join(exp_dir, "metrics_summary.json")
    _write_json(summary_path, metrics)
    logger.info("Wrote metrics summary: %s", summary_path)

    logger.info("\n── key metrics ────────────────────────────────────────")
    if metrics.get("spread_ticks"):
        s = metrics["spread_ticks"]
        logger.info("  spread        mean=%.2f ticks(0.01)  median=%.2f ticks(0.01)", s["mean"], s["median"])
    if metrics.get("volatility_1min") is not None:
        logger.info("  vol (1min)    %.6f", metrics["volatility_1min"])
    if metrics.get("returns_1min"):
        r = metrics["returns_1min"]
        logger.info("  returns       skew=%.3f  kurt=%.3f", r["skew"], r["kurtosis"])
    if metrics.get("iat_ms"):
        i = metrics["iat_ms"]
        logger.info("  IAT           mean=%.1fms  median=%.1fms", i["mean"], i["median"])
    if metrics.get("ob_imbalance"):
        o = metrics["ob_imbalance"]
        logger.info("  OB imbalance  mean=%.4f  std=%.4f", o["mean"], o["std"])

    _log_all_metrics_summary(metrics, logger)
    logger.info("\n" + "=" * 70)
    logger.info("Evaluation complete")
    logger.info("=" * 70)

    return {
        "exp_dir": exp_dir,
        "eval_log": log_file,
        "metrics_summary": summary_path,
        "plots_dir": plots_dir,
    }


# ============================================================
# PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the streamlined fixed-start LOB generation + evaluation pipeline."
    )
    parser.add_argument(
        "ckpt_path",
        help="Path to the model checkpoint used for generated LOB messages.",
    )
    parser.add_argument(
        "--processed-real-flow-path",
        default=None,
        help="Optional prebuilt processed real-flow joblib path. If omitted, the latest cached file is reused or rebuilt.",
    )
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_REALFLOW_DIR, exist_ok=True)

    pipeline_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_dir = os.path.join(BASE_OUT_DIR, f"streamlined_eval_pipeline_{pipeline_ts}")
    os.makedirs(pipeline_dir, exist_ok=True)
    pipeline_log_path = os.path.join(pipeline_dir, "run.log")
    log = _make_line_logger(pipeline_log_path)

    log("[INFO] Starting streamlined LOB eval pipeline")
    log(f"[INFO] checkpoint={ckpt_path}")
    log(
        "[INFO] fixed config "
        f"day={DAY} stock={STOCK} start={START_TIME_STR} "
        f"warmup={WARMUP_ORDER_NUM} lookahead_min={SIM_LOOKAHEAD_MINUTES}"
    )

    if args.processed_real_flow_path:
        processed_real_flow_path = os.path.abspath(args.processed_real_flow_path)
        if not os.path.isfile(processed_real_flow_path):
            raise FileNotFoundError(f"Processed real-flow file not found: {processed_real_flow_path}")
        processed_status = "provided"
        log(f"[INFO] Using user-provided processed real-flow file: {processed_real_flow_path}")
    else:
        existing_processed = _find_matching_processed_realflow(DAY)
        if existing_processed:
            processed_real_flow_path = existing_processed
            processed_status = "reused"
            log(f"[INFO] Reusing processed real-flow cache: {processed_real_flow_path}")
        else:
            processed_real_flow_path = _build_processed_realflow_cache(PROCESSED_REALFLOW_DIR, log)
            processed_status = "created"

    existing_realflow_exp = _find_matching_realflow_exp(processed_real_flow_path)
    if existing_realflow_exp:
        realflow_exp_dir = existing_realflow_exp
        realflow_status = "reused"
        log(f"[INFO] Reusing real-flow LOBSTER stream: {realflow_exp_dir}")
    else:
        log("[INFO] No matching real-flow LOBSTER stream found. Generating...")
        realflow_exp_dir = _generate_realflow_stream(processed_real_flow_path)
        realflow_status = "created"

    existing_generated_exp = _find_matching_generated_exp(ckpt_path)
    if existing_generated_exp:
        generated_exp_dir = existing_generated_exp
        generated_status = "reused"
        log(f"[INFO] Reusing generated LOBSTER stream: {generated_exp_dir}")
    else:
        log("[INFO] No matching generated LOBSTER stream found. Generating...")
        generated_exp_dir = _generate_model_stream(ckpt_path)
        generated_status = "created"

    log("[INFO] Running evaluation for real-flow reference stream...")
    realflow_eval = _evaluate_stream(
        exp_dir=realflow_exp_dir,
        label="real-flow reference evaluation",
        real_ref_dir=None,
    )

    log("[INFO] Running evaluation for generated stream with real-flow comparison metrics...")
    generated_eval = _evaluate_stream(
        exp_dir=generated_exp_dir,
        label="generated stream evaluation",
        real_ref_dir=realflow_exp_dir,
    )

    summary = {
        "created_at": pipeline_ts,
        "pipeline_dir": pipeline_dir,
        "pipeline_log": pipeline_log_path,
        "config": {
            "day": DAY,
            "trade_date": TRADE_DATE_STR,
            "stock": STOCK,
            "start_time": START_TIME_STR,
            "warmup_order_num": WARMUP_ORDER_NUM,
            "window_len": WINDOW_LEN,
            "sim_lookahead_minutes": SIM_LOOKAHEAD_MINUTES,
            "sim_lookahead_ms": SIM_LOOKAHEAD_MS,
        },
        "inputs": {
            "ckpt_path": ckpt_path,
            "processed_real_flow_path": processed_real_flow_path,
        },
        "artifacts": {
            "processed_real_flow": {
                "status": processed_status,
                "path": processed_real_flow_path,
            },
            "realflow_stream": {
                "status": realflow_status,
                "exp_dir": realflow_exp_dir,
            },
            "generated_stream": {
                "status": generated_status,
                "exp_dir": generated_exp_dir,
            },
        },
        "evaluations": {
            "realflow": realflow_eval,
            "generated": generated_eval,
        },
    }

    summary_path = os.path.join(pipeline_dir, "pipeline_summary.json")
    _write_json(summary_path, summary)
    log(f"[DONE] Pipeline summary -> {summary_path}")
    log(f"[DONE] Real-flow metrics -> {realflow_eval['metrics_summary']}")
    log(f"[DONE] Generated metrics -> {generated_eval['metrics_summary']}")


if __name__ == "__main__":
    main()
