#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoregressive Blank-GPT2 token inference (dynamic-anchor variant) on txn-complete preprocess,
then fixed-start decode/replay to LOBSTER and eval vs the same clean real reference dir.

This mirrors inference_replay_blankgpt2_openbidanchor_txncomplete_fixed_start.py but loads
the dynamic-anchor model checkpoint saved by train_blankgpt2_dynamic_anchor_txncomplete_single_day.py.

Validation/early-stop during training is CE-only; here generation uses only logits.
"""

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

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
    apply_event_to_book_open_anchor_txn_complete,
    decode_event_from_token_open_anchor,
    decode_order_token,
    get_lob_snapshot_by_time,
    init_book_from_snapshot,
    load_bin_record,
)


def _load_replay_module():
    path = os.path.join(
        HIST_SCRIPT_DIR,
        "replay_decoded_real_tokens_openbidanchor_txncomplete_fixed_start.py",
    )
    spec = importlib.util.spec_from_file_location("replay_txncomplete_fixed", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class OrderGPT2_DynamicAnchor(nn.Module):
    def __init__(self, vocab_size: int, anchor_count: int):
        super().__init__()
        self.gpt2 = GPT2Model(GPT2Config())
        hidden_size = self.gpt2.config.hidden_size
        self.order_embedding = nn.Embedding(int(vocab_size), hidden_size)
        self.dynamic_anchors = nn.Parameter(torch.randn(int(anchor_count), hidden_size) * 0.02)
        self.head = nn.Linear(hidden_size, int(vocab_size))

    def forward(self, x):
        emb = self.order_embedding(x)
        query = emb.mean(dim=1)
        scores = torch.matmul(query, self.dynamic_anchors.t())
        anchor_probs = torch.softmax(scores, dim=1)
        weighted_anchor = torch.matmul(anchor_probs, self.dynamic_anchors)
        anchor_token = weighted_anchor.unsqueeze(1)
        gpt_input = torch.cat([anchor_token, emb], dim=1)
        outputs = self.gpt2(inputs_embeds=gpt_input)
        h = outputs.last_hidden_state
        logits = self.head(h[:, -1, :])
        return logits


def _extract_ws_table(metrics_summary: dict):
    ref = (metrics_summary or {}).get("reference_comparison", {}).get("metrics", {})
    out = {}
    for name, payload in ref.items():
        if isinstance(payload, dict) and "wasserstein" in payload:
            out[name] = float(payload["wasserstein"])
        elif isinstance(payload, dict) and "weighted_wasserstein" in payload:
            out[name] = float(payload["weighted_wasserstein"])
    return out


def _compare_to_baseline(model_summary_path: str, baseline_summary_path: str, out_json: str):
    with open(model_summary_path, encoding="utf-8") as f:
        m = json.load(f)
    with open(baseline_summary_path, encoding="utf-8") as f:
        b = json.load(f)
    ws_m = _extract_ws_table(m)
    ws_b = _extract_ws_table(b)
    keys = sorted(set(ws_m) & set(ws_b))
    ratios = {}
    deltas = {}
    for k in keys:
        vb, vm = ws_b[k], ws_m[k]
        deltas[k] = float(vm - vb)
        ratios[k] = float(vm / vb) if vb > 1e-12 else None
    report = {
        "model_metrics_summary": model_summary_path,
        "direct_token_replay_metrics_summary": baseline_summary_path,
        "interpretation": (
            "Both streams are evaluated vs the same clean real LOBSTER reference. "
            "Direct replay uses ground-truth order_token decode; model replay autoregresses after seed."
        ),
        "wasserstein_vs_reference": {"model": ws_m, "direct_token_replay": ws_b},
        "delta_model_minus_direct": deltas,
        "ratio_model_over_direct": ratios,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    rep = _load_replay_module()

    parser = argparse.ArgumentParser(
        description="Dynamic-anchor model autoregressive tokens -> txn-complete decode/replay -> eval vs clean ref."
    )
    parser.add_argument("--stock", required=True)
    parser.add_argument("--checkpoint", required=True, help="*_best.pt from dynamic-anchor training")
    parser.add_argument("--processed-real-flow-path", required=True)
    parser.add_argument("--bin-record-path", required=True)
    parser.add_argument(
        "--real-ref-dir",
        required=True,
        help="fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_*",
    )
    parser.add_argument(
        "--baseline-metrics-json",
        default="",
        help="metrics_summary.json from direct replay (real tokens). If set, writes comparison JSON.",
    )
    parser.add_argument("--lob-snap-path", default="/finance_ML/zhanghaohan/LOB_data/20250710/mdl_6_28_0.csv")
    parser.add_argument("--trade-date-str", default="2025-07-10")
    parser.add_argument("--start-time", default="10:00:00")
    parser.add_argument("--sim-lookahead-minutes", type=int, default=10)
    parser.add_argument("--base-out-dir", default="/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream")
    parser.add_argument("--window-len", type=int, default=50)
    parser.add_argument("--vocab-size", type=int, default=40560)
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--inference-seed", type=int, default=12345)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Multinomial sample next token; default is greedy argmax.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature when --sample (ignored for greedy).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="If >0 and --sample: restrict sampling support to top-k logits per step (0 = no restriction).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.inference_seed)
    np.random.seed(args.inference_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)

    vocab = int(ckpt.get("vocab_size", args.vocab_size))
    anchor_count = int(ckpt.get("anchor_count", args.anchor_count))
    model = OrderGPT2_DynamicAnchor(vocab_size=vocab, anchor_count=anchor_count).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stock_tag = args.stock.replace("_", "")
    exp_name = f"fixed_start_model_blankgpt2_tokens_dynamic_anchor_txncomplete_{stock_tag}"
    exp_dir = os.path.join(args.base_out_dir, f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    run_log_path = os.path.join(exp_dir, "run.log")

    def _log(msg: str):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(run_log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    _log(f"[INFO] Experiment folder: {exp_dir}")
    _log(f"[INFO] checkpoint: {args.checkpoint}")
    _log(f"[INFO] device: {device} greedy={not args.sample}")
    _log(f"[INFO] vocab={vocab} anchor_count={anchor_count}")

    lookahead_ms = int(args.sim_lookahead_minutes * 60 * 1000)
    start_ts = pd.Timestamp(f"{args.trade_date_str} {args.start_time}")
    end_ts = start_ts + pd.Timedelta(milliseconds=lookahead_ms)

    lob_info = rep._normalize_snapshot_table(args.lob_snap_path)
    snapshot_df = get_lob_snapshot_by_time(lob_info, args.start_time, args.stock)
    if snapshot_df is None or len(snapshot_df) == 0:
        raise RuntimeError(f"No snapshot for stock={args.stock} at {args.start_time}")
    snapshot_row = snapshot_df.iloc[0]
    book = init_book_from_snapshot(snapshot_row, max_level=rep.LOB_LEVELS)

    df = joblib.load(args.processed_real_flow_path)
    df = df[df["SecurityID"] == args.stock].copy()
    df["TransactDT_MS"] = pd.to_datetime(df["TransactDT_MS"], errors="coerce")
    df = df.dropna(subset=["TransactDT_MS"]).copy()
    df["event_dt"] = pd.to_datetime(
        args.trade_date_str + " " + df["TransactDT_MS"].dt.strftime("%H:%M:%S.%f"),
        errors="coerce",
    )
    df = df.dropna(subset=["event_dt"]).copy()
    df = df[(df["event_dt"] >= start_ts) & (df["event_dt"] <= end_ts)].copy()
    sort_cols = [c for c in ["event_dt", "ChannelNo", "ApplSeqNum"] if c in df.columns]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    token_df = df[df["tokenizable_event"].fillna(False).astype(bool)].copy()
    if len(token_df) == 0:
        raise RuntimeError("No tokenizable rows in window.")

    gt = token_df["order_token"].values.astype(np.int64)
    n = len(gt)
    w = int(args.window_len)
    if n <= w:
        raise RuntimeError(f"Need more than window_len tokenizable rows (n={n}, w={w}).")

    pred = np.zeros(n, dtype=np.int64)
    pred[:w] = gt[:w]

    gen_device = "cuda" if device.type == "cuda" else "cpu"
    sample_gen = torch.Generator(device=gen_device)
    sample_gen.manual_seed(args.inference_seed + 7)

    with torch.no_grad():
        for t in range(w, n):
            x = torch.as_tensor(pred[t - w : t], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x) / max(float(args.temperature), 1e-6)
            if args.sample:
                k = int(args.top_k)
                if k > 0 and k < int(logits.shape[-1]):
                    thr = torch.topk(logits, k, dim=-1).values[..., -1, None]
                    logits = logits.masked_fill(logits < thr, float("-inf"))
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                pick = torch.multinomial(probs, 1, generator=sample_gen).item()
            else:
                pick = int(logits.argmax(dim=-1).item())
            pick = int(min(max(pick, 0), vocab - 1))
            pred[t] = pick

    token_acc = float((pred == gt).mean())
    tail_acc = float((pred[w:] == gt[w:]).mean()) if n > w else 0.0
    _log(
        f"[INFO] tokenizable_rows={n} token_exact_match_rate_all={token_acc:.6f} "
        f"tail_after_seed={tail_acc:.6f}"
    )

    binpack = load_bin_record(args.bin_record_path)
    br_split = bool(binpack.get("raw", {}).get("split_cancel_sides", False))
    if br_split:
        raise RuntimeError("This script expects split_cancel_sides=False (5-side) bin_record.")
    anchor_meta = binpack.get("raw", {}).get("price_anchor_by_stock", {}).get(args.stock)
    if not anchor_meta:
        raise RuntimeError(f"Missing anchor metadata for {args.stock} in bin_record")
    anchor_price = float(anchor_meta["anchor_bid_price"])

    decoded_events = []
    for tok in pred:
        parts = decode_order_token(int(tok), anchor_price=anchor_price)
        ev = decode_event_from_token_open_anchor(parts, stock=args.stock)
        decoded_events.append(ev)

    replay_out_dir = os.path.join(exp_dir, "replay_out")
    os.makedirs(replay_out_dir, exist_ok=True)

    sim_state = rep._FixedStartState(
        stock=args.stock,
        book=book,
        decoded_events=decoded_events,
        df_window=df,
        token_df_window=token_df,
        start_ts=start_ts,
        end_ts=end_ts,
        out_dir=replay_out_dir,
        n_side=5,
    )
    rep._run_fixed_start_replay(sim_state, apply_event_to_book_open_anchor_txn_complete)

    metrics_summary = rep._eval_vs_clean_reference(
        replay_out_dir=replay_out_dir,
        real_ref_dir=args.real_ref_dir,
        out_dir=exp_dir,
    )
    metrics_summary_path = os.path.join(exp_dir, "metrics_summary.json")
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    _log(f"[INFO] metrics_summary: {metrics_summary_path}")

    if args.baseline_metrics_json:
        out_cmp = os.path.join(exp_dir, "compare_vs_direct_replay.json")
        _compare_to_baseline(metrics_summary_path, args.baseline_metrics_json, out_cmp)
        _log(f"[INFO] compare_vs_direct_replay: {out_cmp}")

    _log("[DONE]")


if __name__ == "__main__":
    main()

