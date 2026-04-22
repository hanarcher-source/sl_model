#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a real-flow-generated LOBSTER stream using the exact same pipeline as
`eval_generated_stream.py`.

Usage
-----
python eval_realflow_stream.py
python eval_realflow_stream.py /path/to/exp_dir

Default behavior
----------------
If exp_dir is not provided, this script picks the latest directory matching:
    saved_LOB_stream/fixed_start_realflow_generate_lobster_*
"""

import argparse
import glob
import json
import os

from eval_generated_stream import (
    _HERE,
    _setup_logger,
    _to_serializable,
    load_lobster_pair,
    compute_metrics,
    make_plots,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a real-flow LOBSTER stream (same format as eval_generated_stream.py)."
    )
    parser.add_argument(
        "exp_dir",
        nargs="?",
        default=None,
        help="Path to experiment directory (default: latest fixed_start_realflow_generate_lobster_* in saved_LOB_stream/)",
    )
    args = parser.parse_args()

    if args.exp_dir:
        exp_dir = os.path.abspath(args.exp_dir)
    else:
        base = os.path.abspath(os.path.join(_HERE, "..", "saved_LOB_stream"))
        pattern = "fixed_start_realflow_generate_lobster_*"
        candidates = sorted(glob.glob(os.path.join(base, pattern)))
        if not candidates:
            raise FileNotFoundError(
                f"No {pattern} experiment directories found under {base}"
            )
        exp_dir = candidates[-1]

    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    log_file = os.path.join(exp_dir, "eval.log")
    logger = _setup_logger(log_file)

    logger.info("=" * 70)
    logger.info("eval_realflow_stream.py — stylized fact evaluation")
    logger.info("=" * 70)
    logger.info(f"Evaluating: {exp_dir}")

    logger.info("\n── loading data ───────────────────────────────────────")
    messages, book, notes = load_lobster_pair(exp_dir, logger)

    logger.info("\n── computing metrics ──────────────────────────────────")
    metrics = compute_metrics(messages, book, logger)

    logger.info("\n── capability split (strict) ─────────────────────────")
    for name in metrics["capability_report"]["computed_generated_only_metrics"]:
        logger.info("  [computed] %s", name)
    for name in metrics["capability_report"]["not_attempted_requires_real_reference_lob_data"]:
        logger.info("  [skipped: needs real reference LOB data] %s", name)
    for name in metrics["capability_report"]["not_attempted_requires_true_order_id_linkage"]:
        logger.info("  [skipped: needs true order-ID linkage] %s", name)

    logger.info("\n── generating plots ───────────────────────────────────")
    make_plots(messages, book, plots_dir, notes, logger)

    summary_path = os.path.join(exp_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(_to_serializable(metrics), fh, indent=2)
    logger.info(f"Wrote metrics summary: {summary_path}")

    logger.info("\n── key metrics ────────────────────────────────────────")
    if metrics.get("spread_ticks"):
        s = metrics["spread_ticks"]
        logger.info(
            f"  spread        mean={s['mean']:.2f} ticks(0.01)  median={s['median']:.2f} ticks(0.01)"
        )
    if metrics.get("volatility_1min") is not None:
        logger.info(f"  vol (1min)    {metrics['volatility_1min']:.6f}")
    if metrics.get("returns_1min"):
        r = metrics["returns_1min"]
        logger.info(f"  returns       skew={r['skew']:.3f}  kurt={r['kurtosis']:.3f}")
    if metrics.get("iat_ms"):
        i = metrics["iat_ms"]
        logger.info(f"  IAT           mean={i['mean']:.1f}ms  median={i['median']:.1f}ms")
    if metrics.get("ob_imbalance"):
        o = metrics["ob_imbalance"]
        logger.info(f"  OB imbalance  mean={o['mean']:.4f}  std={o['std']:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("Evaluation complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
