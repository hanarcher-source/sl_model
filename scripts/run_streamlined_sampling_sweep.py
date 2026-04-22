#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a small targeted sampling sweep for the streamlined LOB pipeline.

The sweep is sequential by design so a single SLURM job can capture all runs
in one stdout log while also writing a consolidated CSV/JSON summary.
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from datetime import datetime


PROJECT_ROOT = "/finance_ML/zhanghaohan/stock_language_model"
BASE_OUT_DIR = os.path.join(PROJECT_ROOT, "saved_LOB_stream")
PIPELINE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run_streamlined_lob_eval_pipeline.py")

DEFAULT_SWEEP = [
    {"name": "baseline_t1p3_p0p98", "temperature": 1.3, "top_p": 0.98},
    {"name": "t1p2_p0p95", "temperature": 1.2, "top_p": 0.95},
    {"name": "t1p0_p0p95", "temperature": 1.0, "top_p": 0.95},
    {"name": "t0p8_p0p95", "temperature": 0.8, "top_p": 0.95},
    {"name": "t0p8_p0p90", "temperature": 0.8, "top_p": 0.90},
]


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _find_new_pipeline_summary(before_paths):
    after_paths = set(glob.glob(os.path.join(BASE_OUT_DIR, "streamlined_eval_pipeline_*", "pipeline_summary.json")))
    new_paths = sorted(after_paths - before_paths)
    if new_paths:
        return new_paths[-1]
    if after_paths:
        return sorted(after_paths)[-1]
    return None


def _extract_metric(metrics: dict, *keys):
    cur = metrics
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _build_summary_row(config, pipeline_summary):
    generated_metrics_path = pipeline_summary["evaluations"]["generated"]["metrics_summary"]
    generated_metrics = _load_json(generated_metrics_path)
    ref_metrics = generated_metrics.get("reference_comparison", {}).get("metrics", {})

    row = {
        "run_name": config["name"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "pipeline_summary": pipeline_summary.get("pipeline_dir"),
        "generated_metrics_summary": generated_metrics_path,
        "generated_stream_exp_dir": pipeline_summary["artifacts"]["generated_stream"]["exp_dir"],
        "generated_stream_status": pipeline_summary["artifacts"]["generated_stream"]["status"],
        "spread_l1": _extract_metric(ref_metrics, "spread", "l1_by_group"),
        "spread_w1": _extract_metric(ref_metrics, "spread", "wasserstein"),
        "orderbook_imbalance_l1": _extract_metric(ref_metrics, "orderbook_imbalance", "l1_by_group"),
        "orderbook_imbalance_w1": _extract_metric(ref_metrics, "orderbook_imbalance", "wasserstein"),
        "ask_volume_touch_l1": _extract_metric(ref_metrics, "ask_volume_touch", "l1_by_group"),
        "ask_volume_touch_w1": _extract_metric(ref_metrics, "ask_volume_touch", "wasserstein"),
        "bid_volume_touch_l1": _extract_metric(ref_metrics, "bid_volume_touch", "l1_by_group"),
        "bid_volume_touch_w1": _extract_metric(ref_metrics, "bid_volume_touch", "wasserstein"),
        "log_inter_arrival_time_l1": _extract_metric(ref_metrics, "log_inter_arrival_time", "l1_by_group"),
        "log_inter_arrival_time_w1": _extract_metric(ref_metrics, "log_inter_arrival_time", "wasserstein"),
        "limit_bid_order_depth_l1": _extract_metric(ref_metrics, "limit_bid_order_depth", "l1_by_group"),
        "limit_bid_order_depth_w1": _extract_metric(ref_metrics, "limit_bid_order_depth", "wasserstein"),
        "ofi_up_l1": _extract_metric(ref_metrics, "ofi_up", "l1_by_group"),
        "ofi_down_l1": _extract_metric(ref_metrics, "ofi_down", "l1_by_group"),
        "generated_iat_mean_ms": _extract_metric(generated_metrics, "iat_ms", "mean"),
        "generated_spread_mean_ticks": _extract_metric(generated_metrics, "spread_ticks", "mean"),
        "generated_ob_imbalance_mean": _extract_metric(generated_metrics, "ob_imbalance", "mean"),
    }
    return row


def _write_csv(path: str, rows):
    fieldnames = list(rows[0].keys()) if rows else [
        "run_name", "temperature", "top_p", "pipeline_summary",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run a five-point streamlined sampling sweep.")
    parser.add_argument("ckpt_path", help="Checkpoint path for generated stream model.")
    parser.add_argument(
        "--processed-real-flow-path",
        default=None,
        help="Optional processed real-flow cache path to reuse.",
    )
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    processed_real_flow_path = None
    if args.processed_real_flow_path:
        processed_real_flow_path = os.path.abspath(args.processed_real_flow_path)
        if not os.path.isfile(processed_real_flow_path):
            raise FileNotFoundError(f"Processed real-flow file not found: {processed_real_flow_path}")

    sweep_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(BASE_OUT_DIR, f"streamlined_sampling_sweep_{sweep_ts}")
    os.makedirs(sweep_dir, exist_ok=True)

    print("=" * 88)
    print("Starting streamlined sampling sweep")
    print(f"sweep_dir={sweep_dir}")
    print(f"checkpoint={ckpt_path}")
    if processed_real_flow_path:
        print(f"processed_real_flow_path={processed_real_flow_path}")
    print(f"num_runs={len(DEFAULT_SWEEP)}")
    print("=" * 88)

    rows = []
    for idx, config in enumerate(DEFAULT_SWEEP, start=1):
        print("")
        print("=" * 88)
        print(
            f"[RUN {idx}/{len(DEFAULT_SWEEP)}] "
            f"name={config['name']} temperature={config['temperature']} top_p={config['top_p']}"
        )
        print("=" * 88)

        before_paths = set(glob.glob(os.path.join(BASE_OUT_DIR, "streamlined_eval_pipeline_*", "pipeline_summary.json")))
        cmd = [
            sys.executable,
            PIPELINE_SCRIPT,
            ckpt_path,
            "--temperature", str(config["temperature"]),
            "--top-p", str(config["top_p"]),
            "--run-tag", config["name"],
        ]
        if processed_real_flow_path:
            cmd.extend(["--processed-real-flow-path", processed_real_flow_path])

        subprocess.run(cmd, check=True)

        pipeline_summary_path = _find_new_pipeline_summary(before_paths)
        if pipeline_summary_path is None:
            raise RuntimeError(f"Could not locate pipeline_summary.json for run {config['name']}")

        pipeline_summary = _load_json(pipeline_summary_path)
        row = _build_summary_row(config, pipeline_summary)
        rows.append(row)

        print(
            "[SUMMARY] "
            f"run_name={row['run_name']} "
            f"spread_l1={row['spread_l1']} "
            f"spread_w1={row['spread_w1']} "
            f"ask_touch_l1={row['ask_volume_touch_l1']} "
            f"obi_l1={row['orderbook_imbalance_l1']} "
            f"iat_mean_ms={row['generated_iat_mean_ms']}"
        )

        summary_json_path = os.path.join(sweep_dir, "sweep_results.json")
        summary_csv_path = os.path.join(sweep_dir, "sweep_results.csv")
        with open(summary_json_path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        _write_csv(summary_csv_path, rows)

    print("")
    print("=" * 88)
    print("Sampling sweep complete")
    print(f"json_summary={os.path.join(sweep_dir, 'sweep_results.json')}")
    print(f"csv_summary={os.path.join(sweep_dir, 'sweep_results.csv')}")
    print("=" * 88)


if __name__ == "__main__":
    main()
