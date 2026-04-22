#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
After preprocess (split-cancel joblib + bin_record), run in one process:

  1) Clean realflow LOBSTER reference (generate_lobster_stream_real_openbidanchor_txncomplete_fixed_start.py)
  2) Decoded token replay (replay_decoded_real_tokens_openbidanchor_txncomplete_fixed_start.py)
  3) Stylized-fact eval (eval_generated_stream.py)

Use this so you only submit preprocess separately, then one job for the rest.

Example:

  python -u scripts/hist_script/run_txncomplete_splitcancel_postprocess_pipeline.py \\
    --stocks 000617_XSHE,000981_XSHE,002263_XSHE,002366_XSHE --day 20250710
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

PROJECT_ROOT = "/finance_ML/zhanghaohan/stock_language_model"
SCRIPT_DIR = os.path.join(PROJECT_ROOT, "scripts")
HIST_SCRIPT_DIR = os.path.join(SCRIPT_DIR, "hist_script")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "saved_LOB_stream", "processed_real_flow")
BASE_OUT_DIR = os.path.join(PROJECT_ROOT, "saved_LOB_stream")

GENERATE_PY = os.path.join(HIST_SCRIPT_DIR, "generate_lobster_stream_real_openbidanchor_txncomplete_fixed_start.py")
REPLAY_PY = os.path.join(HIST_SCRIPT_DIR, "replay_decoded_real_tokens_openbidanchor_txncomplete_fixed_start.py")
EVAL_PY = os.path.join(HIST_SCRIPT_DIR, "eval_generated_stream.py")

FLOW_TAG = "openbidanchor_txncomplete_splitcancel"
REF_GLOB = "fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_splitcancel_{stock_tag}_*"
REPLAY_GLOB = "fixed_start_decoded_real_tokens_openbidanchor_txncomplete_splitcancel_{stock_tag}_*"


def _newest_dir_since(pattern: str, since_ts: float) -> str:
    best = None
    best_m = -1.0
    for p in glob.glob(pattern):
        if not os.path.isdir(p):
            continue
        m = os.path.getmtime(p)
        if m >= since_ts - 2.0 and m > best_m:
            best, best_m = p, m
    if best is None:
        raise RuntimeError(f"No directory matched {pattern!r} with mtime >= {since_ts}")
    return best


def _latest_preprocess_paths(stock: str, day: str) -> Tuple[str, str]:
    stock_tag = stock.replace("_", "")
    jpat = os.path.join(
        PROCESSED_DIR,
        f"final_result_for_merge_realflow_{FLOW_TAG}_{day}_{stock_tag}_*.joblib",
    )
    jlibs = sorted(glob.glob(jpat), key=os.path.getmtime, reverse=True)
    if not jlibs:
        raise FileNotFoundError(f"No preprocess joblib for {stock} / {day}: {jpat}")
    jlib = jlibs[0]
    bpat = os.path.join(
        PROCESSED_DIR,
        f"bin_record_realflow_{FLOW_TAG}_{day}_{stock_tag}_*.json",
    )
    bins = sorted(glob.glob(bpat), key=os.path.getmtime, reverse=True)
    if not bins:
        raise FileNotFoundError(f"No bin_record for {stock} / {day}: {bpat}")
    return jlib, bins[0]


def _run(cmd: List[str]) -> None:
    print("\n" + "=" * 72, flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 72 + "\n", flush=True)
    subprocess.run(cmd, check=True)


def _run_one_stock(
    stock: str,
    day: str,
    lob_snap_path: str,
    trade_date_str: str,
    start_time: str,
    sim_lookahead_minutes: int,
    *,
    processed_override: Optional[str] = None,
    bin_override: Optional[str] = None,
) -> dict:
    if processed_override and bin_override:
        jlib, bin_json = processed_override, bin_override
    else:
        jlib, bin_json = _latest_preprocess_paths(stock, day)
    stock_tag = stock.replace("_", "")

    print(f"\n[{stock}] preprocess joblib: {jlib}", flush=True)
    print(f"[{stock}] preprocess bin_record: {bin_json}", flush=True)

    t0 = time.time()
    _run(
        [
            sys.executable,
            "-u",
            GENERATE_PY,
            "--stock",
            stock,
            "--split-cancel-sides",
            "--processed-real-flow-path",
            jlib,
            "--lob-snap-path",
            lob_snap_path,
            "--trade-date-str",
            trade_date_str,
            "--start-time",
            start_time,
            "--sim-lookahead-minutes",
            str(sim_lookahead_minutes),
        ]
    )
    ref_pat = os.path.join(BASE_OUT_DIR, REF_GLOB.format(stock_tag=stock_tag))
    ref_dir = _newest_dir_since(ref_pat, t0)
    print(f"[{stock}] clean reference dir: {ref_dir}", flush=True)

    t1 = time.time()
    _run(
        [
            sys.executable,
            "-u",
            REPLAY_PY,
            "--stock",
            stock,
            "--split-cancel-sides",
            "--processed-real-flow-path",
            jlib,
            "--bin-record-path",
            bin_json,
            "--real-ref-dir",
            ref_dir,
            "--lob-snap-path",
            lob_snap_path,
            "--trade-date-str",
            trade_date_str,
            "--start-time",
            start_time,
            "--sim-lookahead-minutes",
            str(sim_lookahead_minutes),
        ]
    )
    replay_pat = os.path.join(BASE_OUT_DIR, REPLAY_GLOB.format(stock_tag=stock_tag))
    replay_dir = _newest_dir_since(replay_pat, t1)
    print(f"[{stock}] replay experiment dir: {replay_dir}", flush=True)

    _run(
        [
            sys.executable,
            "-u",
            EVAL_PY,
            replay_dir,
            "--real_ref_dir",
            ref_dir,
        ]
    )

    metrics_path = os.path.join(replay_dir, "metrics_summary.json")
    metrics = None
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as fh:
            metrics = json.load(fh)

    return {
        "stock": stock,
        "processed_real_flow_path": jlib,
        "bin_record_path": bin_json,
        "clean_reference_dir": ref_dir,
        "replay_experiment_dir": replay_dir,
        "metrics_summary_path": metrics_path if os.path.isfile(metrics_path) else None,
        "metrics_summary": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chain split-cancel clean-ref generation, token replay, and eval (after preprocess)."
    )
    parser.add_argument("--stock", default=None, help="Single stock, e.g. 000617_XSHE")
    parser.add_argument(
        "--stocks",
        default=None,
        help="Comma-separated stocks; if set, runs all in sequence in this job.",
    )
    parser.add_argument("--day", default="20250710", help="Trading day tag used in preprocess filenames.")
    parser.add_argument(
        "--lob-snap-path",
        default="/finance_ML/zhanghaohan/LOB_data/20250710/mdl_6_28_0.csv",
    )
    parser.add_argument("--trade-date-str", default="2025-07-10")
    parser.add_argument("--start-time", default="10:00:00")
    parser.add_argument("--sim-lookahead-minutes", type=int, default=10)
    parser.add_argument(
        "--processed-real-flow-path",
        default=None,
        help="Optional: exact preprocess joblib (single-stock runs only; skips glob discovery).",
    )
    parser.add_argument(
        "--bin-record-path",
        default=None,
        help="Optional: matching bin_record JSON (required if --processed-real-flow-path is set).",
    )
    parser.add_argument(
        "--pipeline-out-dir",
        default=None,
        help="Where to write pipeline_summary.json (default: saved_LOB_stream/pipeline_txncomplete_splitcancel_<ts>).",
    )
    args = parser.parse_args()

    if args.stocks:
        stocks = [s.strip() for s in args.stocks.split(",") if s.strip()]
    elif args.stock:
        stocks = [args.stock.strip()]
    else:
        parser.error("Provide --stock or --stocks")

    if args.processed_real_flow_path or args.bin_record_path:
        if len(stocks) != 1:
            parser.error("Overrides --processed-real-flow-path/--bin-record-path only allowed with a single stock.")
        if bool(args.processed_real_flow_path) != bool(args.bin_record_path):
            parser.error("Provide both --processed-real-flow-path and --bin-record-path, or neither.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_dir = args.pipeline_out_dir or os.path.join(
        BASE_OUT_DIR, f"pipeline_txncomplete_splitcancel_{ts}"
    )
    os.makedirs(pipeline_dir, exist_ok=True)
    log_path = os.path.join(pipeline_dir, "pipeline.log")

    def plog(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    plog(f"Starting pipeline for {len(stocks)} stock(s): {stocks}")
    results = []
    any_failed = False
    for stock in stocks:
        plog(f"--- begin {stock} ---")
        try:
            row = _run_one_stock(
                stock,
                args.day,
                args.lob_snap_path,
                args.trade_date_str,
                args.start_time,
                args.sim_lookahead_minutes,
                processed_override=args.processed_real_flow_path,
                bin_override=args.bin_record_path,
            )
            results.append({"status": "ok", **row})
            plog(f"--- end {stock} OK ---")
        except Exception as exc:
            plog(f"--- end {stock} FAILED: {exc} ---")
            results.append({"status": "failed", "stock": stock, "error": str(exc)})
            any_failed = True

    summary = {
        "created_at": ts,
        "pipeline_dir": pipeline_dir,
        "pipeline_log": log_path,
        "day": args.day,
        "stocks": stocks,
        "steps": ["generate_clean_ref", "replay_tokens", "eval_generated_stream"],
        "results": results,
        "all_ok": not any_failed,
    }
    summary_path = os.path.join(pipeline_dir, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    plog(f"Wrote {summary_path}")
    print(f"\n[DONE] Pipeline summary: {summary_path}", flush=True)
    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
