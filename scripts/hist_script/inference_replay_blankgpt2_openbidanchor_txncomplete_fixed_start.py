#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoregressive Blank-GPT2 token inference on 20250710 (or any) txn-complete preprocess,
then fixed-start decode/replay to LOBSTER and eval vs the same clean real reference dir.

First window_len tokens match ground truth (seed); remaining tokens are greedy (or sampled)
next-token predictions. Same snapshot, time window, and decode path as
replay_decoded_real_tokens_openbidanchor_txncomplete_fixed_start.py.
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
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config

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


class OrderGPT2_NoAnchor(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        gpt2_config = GPT2Config.from_pretrained("gpt2")
        self.gpt2 = GPT2Model(gpt2_config)
        hidden_size = self.gpt2.config.hidden_size
        self.order_embedding = nn.Embedding(int(vocab_size), hidden_size)
        self.head = nn.Linear(hidden_size, int(vocab_size))

    def forward(self, x):
        emb = self.order_embedding(x)
        outputs = self.gpt2(inputs_embeds=emb)
        h = outputs.last_hidden_state
        logits = self.head(h[:, -1, :])
        return logits


class OrderGPT2_DynamicAnchor(nn.Module):
    def __init__(self, vocab_size: int, anchor_count: int):
        super().__init__()
        # Match dynamic-anchor training: blank GPT2 config (random init)
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


class OrderGPT2_DynGateBias(nn.Module):
    def __init__(self, vocab_size: int, anchor_count: int, gate_dimwise: bool = False):
        super().__init__()
        self.gpt2 = GPT2Model(GPT2Config())
        hidden_size = self.gpt2.config.hidden_size
        self.order_embedding = nn.Embedding(int(vocab_size), hidden_size)
        self.dynamic_anchors = nn.Parameter(torch.randn(int(anchor_count), hidden_size) * 0.02)
        out_dim = hidden_size if gate_dimwise else 1
        self.gate = nn.Linear(hidden_size, out_dim)
        self.gate_dimwise = bool(gate_dimwise)
        self.head = nn.Linear(hidden_size, int(vocab_size))

    def forward(self, x):
        emb = self.order_embedding(x)
        query = emb.mean(dim=1)
        scores = torch.matmul(query, self.dynamic_anchors.t())
        probs = torch.softmax(scores, dim=1)
        weighted_anchor = torch.matmul(probs, self.dynamic_anchors)
        g = torch.sigmoid(self.gate(query))
        emb2 = emb + g.unsqueeze(1) * weighted_anchor.unsqueeze(1)
        outputs = self.gpt2(inputs_embeds=emb2)
        h = outputs.last_hidden_state
        logits = self.head(h[:, -1, :])
        return logits


class OrderGPT2_DynAttnPoolTopK(nn.Module):
    def __init__(self, vocab_size: int, anchor_count: int, topk_anchors: int = 4):
        super().__init__()
        self.gpt2 = GPT2Model(GPT2Config())
        hidden_size = self.gpt2.config.hidden_size
        self.order_embedding = nn.Embedding(int(vocab_size), hidden_size)
        self.dynamic_anchors = nn.Parameter(torch.randn(int(anchor_count), hidden_size) * 0.02)
        self.router_q = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.topk_anchors = int(topk_anchors)
        self.head = nn.Linear(hidden_size, int(vocab_size))

    def forward(self, x):
        emb = self.order_embedding(x)  # [B,T,H]
        attn_scores = (emb * self.router_q.view(1, 1, -1)).sum(dim=2)  # [B,T]
        attn = torch.softmax(attn_scores, dim=1)
        query = torch.sum(attn.unsqueeze(2) * emb, dim=1)  # [B,H]
        scores = torch.matmul(query, self.dynamic_anchors.t())  # [B,A]
        k = max(1, min(self.topk_anchors, int(scores.shape[1])))
        topk_vals, topk_idx = torch.topk(scores, k=k, dim=1)
        masked = torch.full_like(scores, float("-inf"))
        masked.scatter_(1, topk_idx, topk_vals)
        probs = torch.softmax(masked, dim=1)
        weighted_anchor = torch.matmul(probs, self.dynamic_anchors)
        anchor_token = weighted_anchor.unsqueeze(1)
        gpt_input = torch.cat([anchor_token, emb], dim=1)
        outputs = self.gpt2(inputs_embeds=gpt_input)
        h = outputs.last_hidden_state
        logits = self.head(h[:, -1, :])
        return logits


class OrderGPT2_SentencePresetS2IP(nn.Module):
    """
    Inference model for sentence-preset S2IP checkpoints:
      - gpt2: pretrained GPT-2 backbone (weights overwritten by checkpoint)
      - anchor_raw: [A,H] frozen buffer loaded from checkpoint (mean last hidden over valid tokens)
      - proj: trainable shared projection (loaded)
      - prefix: hard top-K projected anchors prepended
    """

    def __init__(
        self,
        vocab_size: int,
        anchor_count: int,
        topk_anchors: int = 5,
        separate_proj: bool = False,
        anchor_map: str = "none",
    ):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        hidden_size = self.gpt2.config.hidden_size
        self.order_embedding = nn.Embedding(int(vocab_size), hidden_size)
        self.head = nn.Linear(hidden_size, int(vocab_size))
        self.separate_proj = bool(separate_proj)
        if self.separate_proj:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.a_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Buffer placeholder; will be overwritten by load_state_dict.
        self.register_buffer("anchor_raw", torch.zeros(int(anchor_count), hidden_size))
        # Optional metadata in some checkpoints. Keep it non-persistent so loading works for both
        # older ckpts (missing) and newer ckpts (extra).
        self.register_buffer(
            "anchor_token_ids",
            torch.zeros(int(anchor_count), dtype=torch.long),
            persistent=False,
        )
        self.anchor_count = int(anchor_count)
        self.topk_anchors = int(topk_anchors)
        self.anchor_map_type = str(anchor_map)
        if self.anchor_map_type == "none":
            self.anchor_map = None
        elif self.anchor_map_type == "linear":
            self.anchor_map = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.anchor_map_type == "mlp":
            self.anchor_map = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            raise ValueError(f"Unknown anchor_map: {self.anchor_map_type!r}")

    def forward(self, x):
        emb = self.order_embedding(x)  # [B,T,H]
        query = emb.mean(dim=1)  # [B,H]
        a_raw = self.anchor_raw
        if self.anchor_map is not None:
            a_raw = self.anchor_map(a_raw)
        if self.separate_proj:
            q = self.q_proj(query)  # [B,H]
            a = self.a_proj(a_raw)  # [A,H]
        else:
            q = self.proj(query)  # [B,H]
            a = self.proj(a_raw)  # [A,H]
        qn = F.normalize(q, dim=1)
        an = F.normalize(a, dim=1)
        scores = torch.matmul(qn, an.t())  # [B,A] cosine
        k = max(1, min(int(self.topk_anchors), int(scores.shape[1])))
        topk_idx = torch.topk(scores, k=k, dim=1).indices  # [B,K]
        prompt = a[topk_idx]  # [B,K,H]
        gpt_input = torch.cat([prompt, emb], dim=1)
        outputs = self.gpt2(inputs_embeds=gpt_input)
        h = outputs.last_hidden_state
        logits = self.head(h[:, -1, :])
        return logits


class OrderGPT2_SentencePresetS2IP_GatedCrossAttn(nn.Module):
    """
    Sentence-preset S2IP + gated cross-attention into the window (see gated-cross-attn trainer).
    State dict must include cross_attn.* and gate_proj.*.
    """

    def __init__(
        self,
        vocab_size: int,
        anchor_count: int,
        topk_anchors: int = 5,
        separate_proj: bool = False,
        anchor_map: str = "none",
        cross_attn_heads: int = 12,
    ):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        hidden_size = self.gpt2.config.hidden_size
        if int(hidden_size) % int(cross_attn_heads) != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by cross_attn_heads {cross_attn_heads}")
        self.order_embedding = nn.Embedding(int(vocab_size), hidden_size)
        self.head = nn.Linear(hidden_size, int(vocab_size))
        self.separate_proj = bool(separate_proj)
        if self.separate_proj:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.a_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.register_buffer("anchor_raw", torch.zeros(int(anchor_count), hidden_size))
        self.register_buffer(
            "anchor_token_ids",
            torch.zeros(int(anchor_count), dtype=torch.long),
            persistent=False,
        )
        self.anchor_count = int(anchor_count)
        self.topk_anchors = int(topk_anchors)
        self.anchor_map_type = str(anchor_map)
        if self.anchor_map_type == "none":
            self.anchor_map = None
        elif self.anchor_map_type == "linear":
            self.anchor_map = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.anchor_map_type == "mlp":
            self.anchor_map = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            raise ValueError(f"Unknown anchor_map: {self.anchor_map_type!r}")
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=int(hidden_size),
            num_heads=int(cross_attn_heads),
            dropout=0.1,
            batch_first=True,
        )
        self.gate_proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        emb = self.order_embedding(x)
        query = emb.mean(dim=1)
        a_raw = self.anchor_raw
        if self.anchor_map is not None:
            a_raw = self.anchor_map(a_raw)
        if self.separate_proj:
            q = self.q_proj(query)
            a = self.a_proj(a_raw)
        else:
            q = self.proj(query)
            a = self.proj(a_raw)
        bsz = emb.shape[0]
        attn_kv = a.unsqueeze(0).expand(bsz, -1, -1).contiguous()
        ca_out, _ = self.cross_attn(emb, attn_kv, attn_kv, need_weights=False)
        gate = torch.sigmoid(self.gate_proj(query))
        emb = emb + gate.unsqueeze(1) * ca_out
        qn = F.normalize(q, dim=1)
        an = F.normalize(a, dim=1)
        scores = torch.matmul(qn, an.t())
        k = max(1, min(int(self.topk_anchors), int(scores.shape[1])))
        topk_idx = torch.topk(scores, k=k, dim=1).indices
        prompt = a[topk_idx]
        gpt_input = torch.cat([prompt, emb], dim=1)
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
        description="Model autoregressive tokens -> txn-complete decode/replay -> eval vs clean ref."
    )
    parser.add_argument("--stock", required=True)
    parser.add_argument("--checkpoint", required=True, help="blankGPT2 *_best.pt from training")
    parser.add_argument("--processed-real-flow-path", required=True)
    parser.add_argument("--bin-record-path", required=True)
    parser.add_argument("--real-ref-dir", required=True, help="fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_*")
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
    parser.add_argument(
        "--model-variant",
        choices=(
            "auto",
            "no_anchor",
            "dynamic_anchor",
            "dyn_gatebias",
            "dyn_attnpool_topk",
            "sentence_preset_s2ip",
            "sentence_preset_s2ip_gated_crossattn",
        ),
        default="auto",
        help="Model forward variant. 'auto' infers from checkpoint fields/state_dict.",
    )
    parser.add_argument("--anchor-count", type=int, default=128, help="Used for dynamic_anchor (or auto).")
    parser.add_argument("--topk-anchors", type=int, default=4, help="Used for dyn_attnpool_topk (or auto).")
    parser.add_argument(
        "--cross-attn-heads",
        type=int,
        default=12,
        help="For sentence_preset_s2ip_gated_crossattn: must match training (hidden/ heads).",
    )
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
    state = ckpt.get("model_state_dict", {})
    cfg = ckpt.get("config") or {}
    cfg_variant = (cfg.get("variant") or "").strip()
    top_variant = (ckpt.get("variant") or "").strip()
    has_dyn = int(ckpt.get("anchor_count", 0)) > 0 or any(k.startswith("dynamic_anchors") for k in state.keys())
    has_gate = any(k.startswith("gate.") for k in state.keys())
    has_router_q = any(k.startswith("router_q") for k in state.keys())
    has_sentence_anchor = "anchor_raw" in state
    has_gca = any(k.startswith("cross_attn.") for k in state.keys())
    has_sentence_sep_proj = any(k.startswith("q_proj.") for k in state.keys()) or any(
        k.startswith("a_proj.") for k in state.keys()
    )
    variant = args.model_variant
    if variant == "auto":
        if top_variant == "sentence_preset_s2ip_gated_crossattn" or cfg_variant == "sentence_preset_s2ip_gated_crossattn":
            variant = "sentence_preset_s2ip_gated_crossattn"
        elif has_gca and has_sentence_anchor:
            variant = "sentence_preset_s2ip_gated_crossattn"
        elif cfg_variant in ("dyn_gatebias", "dyn_attnpool_topk", "sentence_preset_s2ip"):
            variant = cfg_variant
        elif has_sentence_anchor:
            variant = "sentence_preset_s2ip"
        elif has_gate:
            variant = "dyn_gatebias"
        elif has_router_q:
            variant = "dyn_attnpool_topk"
        else:
            variant = "dynamic_anchor" if has_dyn else "no_anchor"

    if variant == "dynamic_anchor":
        anchor_count = int(ckpt.get("anchor_count", args.anchor_count))
        model = OrderGPT2_DynamicAnchor(vocab_size=vocab, anchor_count=anchor_count).to(device)
    elif variant == "dyn_gatebias":
        anchor_count = int(ckpt.get("anchor_count", args.anchor_count))
        gate_dimwise = bool((cfg.get("variant_params") or {}).get("gate_dimwise", False))
        model = OrderGPT2_DynGateBias(vocab_size=vocab, anchor_count=anchor_count, gate_dimwise=gate_dimwise).to(device)
    elif variant == "dyn_attnpool_topk":
        anchor_count = int(ckpt.get("anchor_count", args.anchor_count))
        topk = int((cfg.get("variant_params") or {}).get("topk_anchors") or args.topk_anchors)
        model = OrderGPT2_DynAttnPoolTopK(vocab_size=vocab, anchor_count=anchor_count, topk_anchors=topk).to(device)
    elif variant == "sentence_preset_s2ip":
        # Anchor count comes from buffer shape; fall back to args if missing.
        anchor_count = int((cfg.get("sentence_anchors") or {}).get("count") or args.anchor_count)
        topk = int((cfg.get("sentence_anchors") or {}).get("topk_prepend") or args.topk_anchors)
        separate_proj = bool((cfg.get("sentence_anchors") or {}).get("separate_proj", False)) or bool(
            has_sentence_sep_proj
        )
        anchor_map = str((cfg.get("sentence_anchors") or {}).get("anchor_map") or "none")
        model = OrderGPT2_SentencePresetS2IP(
            vocab_size=vocab,
            anchor_count=anchor_count,
            topk_anchors=topk,
            separate_proj=separate_proj,
            anchor_map=anchor_map,
        ).to(device)
    elif variant == "sentence_preset_s2ip_gated_crossattn":
        anchor_count = int((cfg.get("sentence_anchors") or {}).get("count") or args.anchor_count)
        topk = int((cfg.get("sentence_anchors") or {}).get("topk_prepend") or args.topk_anchors)
        separate_proj = bool((cfg.get("sentence_anchors") or {}).get("separate_proj", False)) or bool(
            has_sentence_sep_proj
        )
        anchor_map = str((cfg.get("sentence_anchors") or {}).get("anchor_map") or "none")
        heads = int(
            ((cfg.get("sentence_anchors") or {}).get("gated_cross_attn") or {}).get("heads")
            or args.cross_attn_heads
        )
        model = OrderGPT2_SentencePresetS2IP_GatedCrossAttn(
            vocab_size=vocab,
            anchor_count=anchor_count,
            topk_anchors=topk,
            separate_proj=separate_proj,
            anchor_map=anchor_map,
            cross_attn_heads=heads,
        ).to(device)
    else:
        model = OrderGPT2_NoAnchor(vocab_size=vocab).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    n_side = 5
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stock_tag = args.stock.replace("_", "")
    if variant in (
        "dynamic_anchor",
        "dyn_gatebias",
        "dyn_attnpool_topk",
        "sentence_preset_s2ip",
        "sentence_preset_s2ip_gated_crossattn",
    ):
        exp_name = f"fixed_start_model_blankgpt2_tokens_dynamic_anchor_txncomplete_{stock_tag}"
    else:
        exp_name = f"fixed_start_model_blankgpt2_tokens_openbidanchor_txncomplete_{stock_tag}"
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
    _log(f"[INFO] tokenizable_rows={n} token_exact_match_rate_all={token_acc:.6f} tail_after_seed={tail_acc:.6f}")

    binpack = load_bin_record(args.bin_record_path)
    br_split = bool(binpack.get("raw", {}).get("split_cancel_sides", False))
    if br_split:
        raise RuntimeError("This script expects split_cancel_sides=False (5-side) bin_record.")
    anchor_meta = binpack.get("raw", {}).get("price_anchor_by_stock", {}).get(args.stock)
    if not anchor_meta:
        raise RuntimeError(f"Missing anchor metadata for {args.stock} in bin_record")
    anchor_price = float(anchor_meta["anchor_bid_price"])

    decode_rng = np.random.default_rng(rep.DECODE_SEED)
    cur_t_ms = 0
    synthetic_order_id = 40_000_000
    raw_generation_rows = []
    lobster_message_rows = []
    lobster_book_rows = []
    rejected_crossing_posts = 0

    token_ids = pred.tolist()
    for idx, (row, token_id) in enumerate(zip(token_df.itertuples(index=False), token_ids)):
        token_id = int(token_id)
        pbin, qbin, ibin, sbin = decode_order_token(
            token_id, rep.PRICE_BIN_NUM, rep.QTY_BIN_NUM, rep.INTERVAL_BIN_NUM, n_side
        )
        ev, dt_ms = decode_event_from_token_open_anchor(
            token_id,
            binpack,
            anchor_price,
            cur_t_ms,
            price_bin_num=rep.PRICE_BIN_NUM,
            qty_bin_num=rep.QTY_BIN_NUM,
            interval_bin_num=rep.INTERVAL_BIN_NUM,
            n_side=n_side,
            decode_method="sample",
            rng=decode_rng,
        )
        cur_t_ms = int(ev.t_ms)
        action = apply_event_to_book_open_anchor_txn_complete(book, ev, split_cancel_sides=False)
        event_ts = start_ts + pd.Timedelta(milliseconds=cur_t_ms)

        original_price = float(np.round(float(getattr(row, "Price")) / 0.01) * 0.01)
        original_qty = int(max(0, int(getattr(row, "OrderQty"))))
        original_side = int(getattr(row, "Side"))
        gt_tok = int(getattr(row, "order_token"))

        if action.get("rejected", False):
            rejected_crossing_posts += 1

        raw_generation_rows.append(
            {
                "gen_idx": int(idx),
                "event_dt_real": pd.Timestamp(getattr(row, "event_dt")).isoformat(),
                "event_dt_decoded": event_ts.isoformat(),
                "gt_order_token": int(gt_tok),
                "model_order_token": int(token_id),
                "real_side": int(original_side),
                "real_price": float(original_price),
                "real_qty": int(original_qty),
                "token": int(token_id),
                "price_bin": int(pbin),
                "qty_bin": int(qbin),
                "interval_bin": int(ibin),
                "side_bin": int(sbin),
                "action": action.get("action"),
                "event_kind": action.get("event_kind"),
                "removed": int(action.get("removed", 0)),
                "rejected": bool(action.get("rejected", False)),
            }
        )

        time_sec = rep._sec_after_midnight(event_ts)
        event_kind = action.get("event_kind", "")
        action_name = action.get("action", "")
        price_int = rep._price_to_lobster_int(ev.abs_price)

        msg_type = None
        msg_size = None
        msg_direction = None

        if event_kind == "post" and not action.get("rejected", False) and action_name == "POST_BID":
            msg_type = 1
            msg_size = int(ev.qty)
            msg_direction = 1
        elif event_kind == "post" and not action.get("rejected", False) and action_name == "POST_ASK":
            msg_type = 1
            msg_size = int(ev.qty)
            msg_direction = -1
        elif event_kind == "cancel":
            removed = int(action.get("removed", 0))
            if removed > 0:
                msg_type = 3 if removed >= int(ev.qty) else 2
                msg_size = removed
                if action.get("resting_side") == "ask":
                    msg_direction = -1
                elif action.get("resting_side") == "bid":
                    msg_direction = 1

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
            lobster_book_rows.append(rep._book_to_lobster_row(book, levels=rep.LOB_LEVELS))

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
                    int(rep._price_to_lobster_int(fill_price)),
                    int(fill_dir),
                ]
            )
            lobster_book_rows.append(rep._book_to_lobster_row(book, levels=rep.LOB_LEVELS))

    raw_csv, msg_file, ob_file = rep._write_outputs(
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
        "checkpoint": args.checkpoint,
        "window_len": w,
        "greedy": not args.sample,
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "inference_seed": int(args.inference_seed),
        "token_exact_match_rate": token_acc,
        "tail_match_after_seed": tail_acc,
        "tokenizable_rows": int(n),
        "rejected_crossing_posts": int(rejected_crossing_posts),
        "raw_generation_csv": raw_csv,
        "lobster_message_csv": msg_file,
        "lobster_orderbook_csv": ob_file,
        "paths": {
            "processed_real_flow_path": args.processed_real_flow_path,
            "bin_record_path": args.bin_record_path,
            "real_ref_dir": args.real_ref_dir,
        },
    }
    with open(os.path.join(exp_dir, "generation_notes.json"), "w", encoding="utf-8") as fh:
        json.dump(notes, fh, indent=2)

    _log("[INFO] Running eval vs clean reference...")
    metrics, summary_path, eval_log_path = rep._evaluate(exp_dir, args.real_ref_dir)
    notes["metrics_summary_json"] = summary_path
    notes["eval_log"] = eval_log_path
    with open(os.path.join(exp_dir, "generation_notes.json"), "w", encoding="utf-8") as fh:
        json.dump(notes, fh, indent=2)

    if args.baseline_metrics_json and os.path.isfile(args.baseline_metrics_json):
        cmp_path = os.path.join(exp_dir, "comparison_vs_direct_token_replay.json")
        rep_report = _compare_to_baseline(summary_path, args.baseline_metrics_json, cmp_path)
        _log(f"[INFO] Wrote comparison: {cmp_path}")
        for k in sorted(rep_report.get("ratio_model_over_direct", {}))[:12]:
            r = rep_report["ratio_model_over_direct"][k]
            d = rep_report["delta_model_minus_direct"][k]
            _log(f"  [{k}] ratio_model/direct_ws={r} delta={d:.6f}")

    _log(f"[DONE] metrics_summary: {summary_path}")


if __name__ == "__main__":
    main()
