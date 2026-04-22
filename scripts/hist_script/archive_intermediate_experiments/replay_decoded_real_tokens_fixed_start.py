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
    decode_event_from_token,
    decode_order_token,
    get_lob_snapshot_by_time,
    init_book_from_snapshot,
    load_bin_record,
)


LOB_LEVELS = 10
PRICE_BIN_NUM = 26
QTY_BIN_NUM = 26
INTERVAL_BIN_NUM = 12
N_SIDE = 3
DECODE_SEED = 43


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


def _write_outputs(exp_dir, stock, trade_date_str, start_time_str, end_ms, raw_rows, lobster_message_rows, lobster_book_rows):
    raw_csv = os.path.join(exp_dir, "raw_generation_log.csv")
    pd.DataFrame(raw_rows).to_csv(raw_csv, index=False)

    ticker_for_file = _safe_ticker_for_filename(stock)
    start_ms, end_ts_ms = _start_end_ms_from_time_str(start_time_str, end_ms)

    msg_file = os.path.join(
        exp_dir,
        f"{ticker_for_file}_{trade_date_str}_{start_ms}_{end_ts_ms}_message_{LOB_LEVELS}.csv",
    )
    ob_file = os.path.join(
        exp_dir,
        f"{ticker_for_file}_{trade_date_str}_{start_ms}_{end_ts_ms}_orderbook_{LOB_LEVELS}.csv",
    )

    with open(msg_file, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(lobster_message_rows)
    with open(ob_file, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(lobster_book_rows)

    return raw_csv, msg_file, ob_file


def _evaluate(exp_dir: str, real_ref_dir: str):
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    log_file = os.path.join(exp_dir, "eval.log")
    logger = _setup_logger(log_file)
    logger.info("=" * 70)
    logger.info("decoded real-token replay evaluation")
    logger.info("=" * 70)
    logger.info(f"Evaluating: {exp_dir}")

    logger.info("\n── loading data ───────────────────────────────────────")
    messages, book, notes = load_lobster_pair(exp_dir, logger)
    logger.info(f"Loading real reference LOBSTER data from {real_ref_dir}")
    ref_messages, ref_book, _ = load_lobster_pair(real_ref_dir, logger)

    logger.info("\n── computing metrics ──────────────────────────────────")
    metrics = compute_metrics(messages, book, logger, ref_messages=ref_messages, ref_book=ref_book)

    logger.info("\n── generating plots ───────────────────────────────────")
    make_plots(messages, book, plots_dir, notes, logger)

    summary_path = os.path.join(exp_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(_to_serializable(metrics), fh, indent=2)
    logger.info(f"Wrote metrics summary: {summary_path}")
    _log_all_metrics_summary(metrics, logger)
    return metrics, summary_path, log_file


def main():
    parser = argparse.ArgumentParser(description="Replay real token stream through decoder/book reconstruction and evaluate it.")
    parser.add_argument("--stock", required=True, help="Stock code in *_XSHE format.")
    parser.add_argument("--processed-real-flow-path", required=True, help="Path to processed real-flow joblib with token columns.")
    parser.add_argument("--bin-record-path", required=True, help="Path to companion real-flow bin record JSON.")
    parser.add_argument("--real-ref-dir", required=True, help="Path to real fixed-start LOBSTER reference directory.")
    parser.add_argument("--lob-snap-path", default="/finance_ML/zhanghaohan/LOB_data/20250710/mdl_6_28_0.csv")
    parser.add_argument("--day", default="20250710")
    parser.add_argument("--trade-date-str", default="2025-07-10")
    parser.add_argument("--start-time", default="10:00:00")
    parser.add_argument("--sim-lookahead-minutes", type=int, default=10)
    parser.add_argument("--base-out-dir", default="/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stock_tag = args.stock.replace("_", "")
    exp_name = f"fixed_start_decoded_real_tokens_{stock_tag}"
    exp_dir = os.path.join(args.base_out_dir, f"{exp_name}_{timestamp}")
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
    snapshot_row = snapshot_df.iloc[0]

    book = init_book_from_snapshot(snapshot_row, max_level=LOB_LEVELS)
    snap_mid = snapshot_row.get("MidPrice", None)
    if snap_mid is None or pd.isna(snap_mid):
        midpoint = book.midpoint()
        cur_mid = 100.0 if midpoint is None else float(midpoint)
    else:
        cur_mid = float(snap_mid)

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

    binpack = load_bin_record(args.bin_record_path)
    decode_rng = np.random.default_rng(DECODE_SEED)
    cur_t_ms = 0
    synthetic_order_id = 30_000_000

    raw_generation_rows = []
    lobster_message_rows = []
    lobster_book_rows = []

    for idx, row in enumerate(token_df.itertuples(index=False), start=0):
        token_id = int(getattr(row, "order_token"))
        pbin, qbin, ibin, sbin = decode_order_token(
            token_id,
            PRICE_BIN_NUM,
            QTY_BIN_NUM,
            INTERVAL_BIN_NUM,
            N_SIDE,
        )

        ev, dt_ms = decode_event_from_token(
            token_id,
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

        event_ts = start_ts + pd.Timedelta(milliseconds=cur_t_ms)
        original_price = float(np.round(float(getattr(row, "Price")) / 0.01) * 0.01)
        original_qty = int(max(0, int(getattr(row, "OrderQty"))))
        original_side = int(getattr(row, "Side"))

        raw_generation_rows.append(
            {
                "gen_idx": int(idx),
                "event_dt_real": pd.Timestamp(getattr(row, "event_dt")).isoformat(),
                "event_dt_decoded": event_ts.isoformat(),
                "real_side": int(original_side),
                "real_price": float(original_price),
                "real_qty": int(original_qty),
                "real_price_mid_diff": None if pd.isna(getattr(row, "Price_Mid_diff", np.nan)) else int(getattr(row, "Price_Mid_diff")),
                "token": int(token_id),
                "price_bin": int(pbin),
                "qty_bin": int(qbin),
                "interval_bin": int(ibin),
                "side_bin": int(sbin),
                "decoded_t_ms": int(cur_t_ms),
                "decoded_dt_ms": int(dt_ms),
                "decoded_ticks_from_mid": int(ev.ticks_from_mid),
                "decoded_qty": int(ev.qty),
                "decoded_abs_price": float(ev.abs_price),
                "action": action.get("action", None),
                "event_kind": action.get("event_kind", None),
                "resting_side": action.get("resting_side", None),
                "removed": int(action.get("removed", 0)),
                "fill_qty": int(sum(int(fill.get("filled_qty", 0)) for fill in action.get("fills", []))),
                "fills": json.dumps(action.get("fills", []), ensure_ascii=True),
                "best_bid_after": None if book.best_bid() is None else float(book.best_bid()),
                "best_ask_after": None if book.best_ask() is None else float(book.best_ask()),
                "mid_after": None if cur_mid is None else float(cur_mid),
            }
        )

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

        for fill in action.get("fills", []):
            fill_qty = int(fill.get("filled_qty", 0))
            fill_price = float(fill.get("price", ev.abs_price))
            if fill_qty <= 0:
                continue

            resting_side = fill.get("side", "")
            fill_dir = 1 if resting_side == "ask" else -1
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

    raw_csv, msg_file, ob_file = _write_outputs(
        exp_dir,
        args.stock,
        args.trade_date_str,
        args.start_time,
        max(cur_t_ms, lookahead_ms),
        raw_generation_rows,
        lobster_message_rows,
        lobster_book_rows,
    )

    notes = {
        "exp_name": exp_name,
        "timestamp": timestamp,
        "stock": args.stock,
        "start_time": args.start_time,
        "trade_date_str": args.trade_date_str,
        "sim_lookahead_minutes": int(args.sim_lookahead_minutes),
        "window_input_rows": int(len(df)),
        "tokenizable_rows": int(len(token_df)),
        "decode_seed": int(DECODE_SEED),
        "decoded_reached_t_ms": int(cur_t_ms),
        "lobster_message_rows": int(len(lobster_message_rows)),
        "lobster_orderbook_rows": int(len(lobster_book_rows)),
        "raw_generation_csv": raw_csv,
        "lobster_message_csv": msg_file,
        "lobster_orderbook_csv": ob_file,
        "paths": {
            "processed_real_flow_path": args.processed_real_flow_path,
            "bin_record_path": args.bin_record_path,
            "lob_snap_path": args.lob_snap_path,
            "real_ref_dir": args.real_ref_dir,
        },
        "conversion_caveats": [
            "Only tokenizable real events are replayed directly (sides 49, 50, 99).",
            "Execution rows are reconstructed indirectly from aggressive crossing fills discovered by the simulator.",
            "Within-bin values are regenerated from the saved empirical per-bin distributions in the bin record.",
            "Order IDs are synthetic for LOBSTER export compatibility.",
        ],
        "run_log": run_log_path,
    }

    notes_json = os.path.join(exp_dir, "generation_notes.json")
    with open(notes_json, "w", encoding="utf-8") as fh:
        json.dump(notes, fh, indent=2)

    log("[DONE] Decoded real-token fixed-start replay export complete.")
    log(f"[DONE] Raw replay log: {raw_csv}")
    log(f"[DONE] LOBSTER message CSV: {msg_file}")
    log(f"[DONE] LOBSTER orderbook CSV: {ob_file}")
    log(f"[DONE] Notes JSON: {notes_json}")

    metrics, summary_path, eval_log_path = _evaluate(exp_dir, args.real_ref_dir)
    notes["metrics_summary_json"] = summary_path
    notes["eval_log"] = eval_log_path
    notes["key_reference_metrics"] = {
        key: metrics.get("reference_comparison", {}).get("metrics", {}).get(key)
        for key in [
            "spread",
            "orderbook_imbalance",
            "log_inter_arrival_time",
            "ask_volume_touch",
            "bid_volume_touch",
            "ofi",
        ]
    }
    with open(notes_json, "w", encoding="utf-8") as fh:
        json.dump(_to_serializable(notes), fh, indent=2)

    log(f"[DONE] Evaluation summary JSON: {summary_path}")
    log(f"[DONE] Evaluation log: {eval_log_path}")


if __name__ == "__main__":
    main()