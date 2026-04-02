#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a single simulated LOB stream from a fixed start time and export:
1) raw simulator logs (with fills/completions)
2) best-effort LOBSTER-style message/orderbook CSV files

Notes:
- This script does NOT run evaluation and does NOT random-sample start times.
- It uses a fixed start time (default 10:00:00) and warmup context length 50.
- It uses the checkpoint copied into stock_language_model/model_cache.

Uncertainty / best-effort conversion caveat:
- The internal simulator uses queue quantities by price level, not persistent exchange order IDs.
- Therefore, LOBSTER message "Order ID" semantics cannot be perfectly reconstructed for
  cancels/executions against historical resting liquidity. We emit synthetic IDs and keep
  detailed raw logs so downstream users can audit conversion assumptions.
"""

import csv
import os
import sys
import json
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch

# ============================================================
# PATH SETUP
# ============================================================

UTILITY_DIR = "/finance_ML/zhanghaohan/stock_language_model/utility"
if UTILITY_DIR not in sys.path:
    sys.path.append(UTILITY_DIR)

from sim_helper_unified import (
    get_lob_snapshot_by_time,
    get_order_window_ending_at_second,
    load_bin_record,
    build_model,
    load_checkpoint,
    sample_next_token,
    decode_event_from_token,
    apply_event_to_book,
    init_book_from_snapshot,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXP_NAME = "fixed_start_617_blank_generate_lobster"

SELECTED_STOCKS = ["000617_XSHE"]
STOCK = SELECTED_STOCKS[0]

# Input data
DATA_PATH = "/finance_ML/zhanghaohan/LOB_data/20250710/final_result_for_merge_vocab24336_multidaypool(03_10)_samp_20260318_1743.joblib"
BIN_RECORD_PATH = "/finance_ML/zhanghaohan/LOB_data/20250710/bin_record_vocab24336_multidaypool(03_10)_samp_20260318_1743.json"
LOB_SNAP_PATH = "/finance_ML/zhanghaohan/LOB_data/20250710/mdl_6_28_0.csv"

# Use the checkpoint copied to the new workspace
CKPT_PATH = "/finance_ML/zhanghaohan/stock_language_model/model_cache/blankGPT2_multiday_continue_617_stock_win50_20260325_202035.pt"

# Fixed-start generation setup
START_TIME_STR = "10:00:00"
WARMUP_ORDER_NUM = 50
WINDOW_LEN = 50

# Simulation length
# This is a tunable choice. Defaulting to 10 minutes for now.
SIM_LOOKAHEAD_MINUTES = 10
SIM_LOOKAHEAD_MS = SIM_LOOKAHEAD_MINUTES * 60 * 1000
MAX_GENERATED_STEPS = 20000
LOG_EVERY_STEPS = 200

# must match model/tokenization
MODEL_VARIANT = "no_anchor"  # "anchor" or "no_anchor"
ANCHOR_COUNT = 128            # used only when MODEL_VARIANT == "anchor"

PRICE_BIN_NUM = 26
QTY_BIN_NUM = 26
INTERVAL_BIN_NUM = 12
N_SIDE = 3
VOCAB_SIZE = 24336

USE_SAMPLING = True
TEMPERATURE = 1.3
TOP_P = 0.98
SAMPLE_SEED = 1234

# Export config
LOB_LEVELS = 10
BASE_OUT_DIR = "/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream"
TRADE_DATE_STR = "2025-07-10"


# ============================================================
# REPRO
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOBSTER EXPORT HELPERS
# ============================================================

def _price_to_lobster_int(price_float: float) -> int:
    return int(round(float(price_float) * 10000.0))


def _sum_queue(q) -> int:
    if q is None:
        return 0
    return int(sum(q))


def _book_to_lobster_row(book, levels: int = 10):
    """
    Build one LOBSTER orderbook row:
    [AskPrice1, AskSize1, BidPrice1, BidSize1, AskPrice2, AskSize2, BidPrice2, BidSize2, ...]
    """
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

    # --------------------
    # Load snapshot table
    # --------------------
    log("[INFO] Loading LOB snapshot file...")
    lob_snap = pd.read_csv(LOB_SNAP_PATH)

    columns = list(lob_snap.columns) + ["MISC"]
    LOB_info = pd.read_csv(LOB_SNAP_PATH, header=None, names=columns)
    LOB_info = LOB_info[1:].copy()

    LOB_info["AskPrice1"] = pd.to_numeric(LOB_info["AskPrice1"], errors="coerce")
    LOB_info["BidPrice1"] = pd.to_numeric(LOB_info["BidPrice1"], errors="coerce")
    LOB_info = LOB_info.dropna(subset=["AskPrice1", "BidPrice1"]).copy()

    LOB_info["MidPrice"] = (LOB_info["AskPrice1"] + LOB_info["BidPrice1"]) / 2
    LOB_info["SecurityID"] = LOB_info["SecurityID"].astype(str).str.zfill(6) + "_XSHE"

    LOB_info["UpdateTime"] = LOB_info["UpdateTime"].astype(str)
    LOB_info = LOB_info[LOB_info["UpdateTime"].str.len() >= 8].copy()

    LOB_info["TransactDT_SEC"] = pd.to_datetime(
        LOB_info["UpdateTime"].str.slice(0, 8),
        format="%H:%M:%S",
        errors="coerce",
    ).dt.floor("s")
    LOB_info = LOB_info.dropna(subset=["TransactDT_SEC"]).copy()

    log(f"[INFO] LOB_info rows after cleaning: {len(LOB_info)}")

    # --------------------
    # Load processed order data
    # --------------------
    log("[INFO] Loading processed LOB data...")
    processed_LOB_data = joblib.load(DATA_PATH)
    processed_LOB_data = processed_LOB_data[
        processed_LOB_data["SecurityID"].isin(SELECTED_STOCKS)
    ].copy()
    log(f"[INFO] processed_LOB_data filtered rows: {len(processed_LOB_data)}")

    # --------------------
    # Fixed start snapshot + warmup window
    # --------------------
    snapshot_df = get_lob_snapshot_by_time(LOB_info, START_TIME_STR, STOCK)
    if snapshot_df is None or len(snapshot_df) == 0:
        raise RuntimeError(f"No snapshot found for stock={STOCK} at {START_TIME_STR}")
    snapshot_row = snapshot_df.iloc[0]

    window_df = get_order_window_ending_at_second(
        processed_LOB_data=processed_LOB_data,
        target_time=START_TIME_STR,
        stock=STOCK,
        order_num=WARMUP_ORDER_NUM,
    )
    if window_df is None or len(window_df) < WARMUP_ORDER_NUM:
        raise RuntimeError(
            f"Warmup window insufficient. Need {WARMUP_ORDER_NUM}, got {0 if window_df is None else len(window_df)}"
        )

    # --------------------
    # Model + bin record
    # --------------------
    log("[INFO] Loading bin record...")
    binpack = load_bin_record(BIN_RECORD_PATH)

    log("[INFO] Building model...")
    model = build_model(
        model_variant=MODEL_VARIANT,
        vocab_size=VOCAB_SIZE,
        gpt2_name="gpt2",
        anchor_count=ANCHOR_COUNT,
    )
    _ = load_checkpoint(model, CKPT_PATH, DEVICE)

    # --------------------
    # Init simulation state
    # --------------------
    book = init_book_from_snapshot(snapshot_row, max_level=LOB_LEVELS)

    snap_mid = snapshot_row.get("MidPrice", None)
    if snap_mid is None or pd.isna(snap_mid):
        m = book.midpoint()
        if m is None:
            cur_mid = 100.0
        else:
            cur_mid = float(m)
    else:
        cur_mid = float(snap_mid)

    tokens = window_df["order_token"].astype(int).to_numpy()
    if len(tokens) < WINDOW_LEN:
        raise RuntimeError(f"Need at least {WINDOW_LEN} context tokens, got {len(tokens)}")
    context = tokens[-WINDOW_LEN:].tolist()

    sample_gen = torch.Generator()
    sample_gen.manual_seed(SAMPLE_SEED)
    decode_rng = np.random.default_rng(43)

    # Base wall-clock timestamp for exported stream
    stream_start_dt = pd.Timestamp(f"{TRADE_DATE_STR} {START_TIME_STR}")

    # --------------------
    # Generate forward
    # --------------------
    log("[INFO] Starting fixed-start generation...")
    log(f"[INFO] stock={STOCK} start_time={START_TIME_STR} lookahead_min={SIM_LOOKAHEAD_MINUTES}")

    cur_t_ms = 0
    step = 0

    raw_generation_rows = []
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

        m = book.midpoint()
        if m is not None:
            cur_mid = float(m)

        context.append(int(tok_next))

        # Build raw simulator log row (includes completion/fill details).
        raw_generation_rows.append(
            {
                "gen_idx": int(step),
                "t_ms": int(cur_t_ms),
                "token": int(tok_next),
                "side_bin": int(ev.side_bin),
                "ticks_from_mid": int(ev.ticks_from_mid),
                "qty": int(ev.qty),
                "abs_price": float(ev.abs_price),
                "action": action.get("action", None),
                "fills": json.dumps(action.get("fills", []), ensure_ascii=True),
                "removed": int(action.get("removed", 0)),
                "dt_ms": int(dt_ms),
                "mid_after": None if cur_mid is None else float(cur_mid),
                "best_bid_after": None if book.best_bid() is None else float(book.best_bid()),
                "best_ask_after": None if book.best_ask() is None else float(book.best_ask()),
            }
        )

        # Convert current event to best-effort LOBSTER message rows.
        event_ts = stream_start_dt + pd.Timedelta(milliseconds=cur_t_ms)
        time_sec = _sec_after_midnight(event_ts)

        action_name = action.get("action", "")
        price_int = _price_to_lobster_int(ev.abs_price)

        # 1) Submission / cancel message (one per generated event)
        msg_type = None
        msg_size = None
        msg_direction = None

        if action_name == "POST_BID":
            msg_type = 1
            msg_size = int(ev.qty)
            msg_direction = 1
        elif action_name == "POST_ASK":
            msg_type = 1
            msg_size = int(ev.qty)
            msg_direction = -1
        elif action_name.startswith("CANCEL"):
            removed = int(action.get("removed", 0))
            if removed > 0:
                msg_type = 3 if removed >= int(ev.qty) else 2
                msg_size = removed
                # cancel ask => direction -1, cancel bid => direction +1
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

        # 2) Also emit visible executions discovered by simulator matching logic.
        #    Caveat: order IDs are synthetic due to missing persistent per-order tracking.
        fills = action.get("fills", [])
        for fill in fills:
            fill_qty = int(fill.get("filled_qty", 0))
            fill_price = float(fill.get("price", ev.abs_price))
            if fill_qty <= 0:
                continue

            # fill side refers to resting book side consumed.
            # resting ask consumed -> buyer initiated -> direction +1
            # resting bid consumed -> seller initiated -> direction -1
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

    # --------------------
    # Save outputs
    # --------------------
    log("[INFO] Saving outputs...")

    raw_csv = os.path.join(exp_dir, "raw_generation_log.csv")
    pd.DataFrame(raw_generation_rows).to_csv(raw_csv, index=False)

    # LOBSTER-style filenames
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
        "model_variant": MODEL_VARIANT,
        "start_time": START_TIME_STR,
        "sim_lookahead_minutes": SIM_LOOKAHEAD_MINUTES,
        "sim_lookahead_ms": SIM_LOOKAHEAD_MS,
        "generated_steps": int(step),
        "lobster_message_rows": int(len(lobster_message_rows)),
        "lobster_orderbook_rows": int(len(lobster_book_rows)),
        "raw_generation_csv": raw_csv,
        "lobster_message_csv": msg_file,
        "lobster_orderbook_csv": ob_file,
        "paths": {
            "data_path": DATA_PATH,
            "bin_record_path": BIN_RECORD_PATH,
            "lob_snap_path": LOB_SNAP_PATH,
            "ckpt_path": CKPT_PATH,
        },
        "conversion_caveats": [
            "Order IDs are synthetic; true exchange-level ID linkage is not preserved in simulator state.",
            "Execution rows are emitted from simulator fill logs; multiple messages can share same timestamp.",
            "Book rows are emitted per output message for LOBSTER shape consistency.",
        ],
        "run_log": run_log_path,
    }

    notes_json = os.path.join(exp_dir, "generation_notes.json")
    with open(notes_json, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)

    log("[DONE] Fixed-start generation complete.")
    log(f"[DONE] Raw generation log: {raw_csv}")
    log(f"[DONE] LOBSTER message CSV: {msg_file}")
    log(f"[DONE] LOBSTER orderbook CSV: {ob_file}")
    log(f"[DONE] Notes JSON: {notes_json}")


if __name__ == "__main__":
    main()
