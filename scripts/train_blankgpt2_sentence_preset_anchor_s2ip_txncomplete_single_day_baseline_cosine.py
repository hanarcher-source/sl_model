#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFERENCE COPY — sentence preset + S2IP with **cosine** routing (hard top-K prepend).

- Keep this file as a stable fallback when experimenting elsewhere.
- For new behavior (dot scores, extra regularizers, etc.), copy and rename to a new
  `train_blankgpt2_*_variant_*.py` script rather than editing the canonical trainer or
  this snapshot in place.

Pretrained GPT-2 + sentence semantic anchors (preset txt) + S2IP-style alignment.

Anchors:
  - Load sentences from preset_anchors.txt (same format as GPT2_new_head/sentence_semantic_anchor).
  - Each anchor = mean of GPT-2 last hidden states over valid (non-pad) tokens per sentence
    (same as hidden_state_trainable_w_reg_v2.py); one batched forward at init.
  - Optional: center anchors (remove mean over anchor bank).
  - Optional: make anchors trainable (nn.Parameter).

Additional anchor option (paper-faithful alternative):
  - Frequency-based codebook anchors from the LOB token space:
    select top-N most frequent `order_token` ids from the training day and initialize
    anchor bank from `order_embedding.weight[token_id]` rows.
  - Optional learnable anchor mapping f(·) on anchors (linear or small MLP).

Routing / prefix:
  - query = mean(order embeddings over the LOB window)
  - Projection into a joint space:
      - shared proj, or
      - separate q_proj / a_proj (as seen in earlier sentence-anchor prototypes).
  - Cosine similarity scores; **hard top-K** anchor rows prepended (no softmax mixture for the prefix).

Regularization (S2IP-style, training only):
  - align = mean(cosine scores of the retrieved top-K) over the batch
  - total_loss = CE - lambda_align * align
  - Optional warmup: lambda_align = 0 for first --align-warmup-steps

Val / early stopping: CE only (no align term).

Defaults: max 3 epochs, patience 3 (same spirit as other blank GPT-2 txncomplete scripts).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from contextlib import nullcontext
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Model

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from train_blankgpt2_openbidanchor_txncomplete_single_day import (
    BATCH_SIZE,
    DEFAULT_VOCAB_SIZE,
    LR,
    MIN_DELTA,
    PRINT_EVERY_STEPS,
    SEED,
    WINDOW_LEN,
    build_per_stock_splits,
    find_stock_joblib,
    make_dataloader_generator,
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

DEFAULT_PRESET_ANCHORS = (
    "/finance_ML/zhanghaohan/GPT2_new_head/sentence_semantic_anchor/preset_anchors/preset_anchors.txt"
)
DEFAULT_OUTPUT_ROOT = (
    "/finance_ML/zhanghaohan/stock_language_model/training_runs/"
    "pool_0709_0710_train0709_pretrained_gpt2_sentence_preset_s2ip_win50"
)


def _clean_anchor_line(line: str) -> str:
    s = line.strip()
    if not s:
        return ""
    s = re.sub(r"^\s*\d+\.\s*", "", s)
    s = s.strip()
    if (s.startswith("\u201c") and s.endswith("\u201d")) or (
        s.startswith('"') and s.endswith('"')
    ) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    s = s.strip("\u201c\u201d\"' ").strip()
    return s


def load_anchor_sentences(path: str, max_sentences: int) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preset anchor file not found: {path}")
    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            t = _clean_anchor_line(raw)
            if t:
                lines.append(t)
    n = int(max_sentences)
    if len(lines) < n:
        raise RuntimeError(f"Need >= {n} anchor lines, got {len(lines)} from {path}")
    return lines[:n]


class ConcatWindowDatasetLocal(Dataset):
    def __init__(self, samples, window: int):
        self.samples = samples
        self.window = int(window)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, i = self.samples[idx]
        x = seq[i : i + self.window]
        y = seq[i + self.window]
        return x, y


def make_day_loaders_and_datasets(df, window_len, batch_size, shuffle_seed):
    train_samples, val_samples, test_samples, split_stats = build_per_stock_splits(
        df, window_len
    )
    if (
        split_stats["total"]["train"] <= 0
        or split_stats["total"]["val"] <= 0
        or split_stats["total"]["test"] <= 0
    ):
        raise RuntimeError("Not enough windows after per-stock splitting.")

    train_ds = ConcatWindowDatasetLocal(train_samples, window_len)
    val_ds = ConcatWindowDatasetLocal(val_samples, window_len)
    test_ds = ConcatWindowDatasetLocal(test_samples, window_len)
    dl_gen = make_dataloader_generator(shuffle_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=dl_gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, split_stats


@torch.no_grad()
def build_anchor_matrix_mean_last_hidden(
    gpt2: GPT2Model,
    sentences: list[str],
    tokenizer,
    anchor_max_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """[A, H] frozen vectors: mean of last-layer hidden states over non-pad positions (v2 recipe)."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    toks = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=int(anchor_max_tokens),
        return_tensors="pt",
    )
    input_ids = toks["input_ids"].to(device)
    attn = toks["attention_mask"].float().to(device)
    gpt2.eval()
    out = gpt2(input_ids=input_ids, attention_mask=attn.long())
    hs = out.last_hidden_state
    mask = attn.unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    anchor_raw = (hs * mask).sum(dim=1) / denom
    return anchor_raw.detach()


class OrderGPT2_SentencePresetS2IP(nn.Module):
    """
    Sentence anchors (mean last-layer hidden over valid tokens).
    Trainable: order emb, head, GPT-2, and projections; anchors optional trainable.
    Prefix = hard top-K projected anchors by cosine to projected query.
    """

    def __init__(
        self,
        vocab_size: int,
        sentences: list[str] | None,
        *,
        anchor_source: str = "sentence",
        anchor_token_ids: list[int] | None = None,
        anchor_vocab_ids: list[int] | None = None,
        anchor_map: str = "none",
        pretrained_name: str,
        local_files_only: bool,
        topk_anchors: int,
        anchor_max_tokens: int,
        center_anchors: bool,
        trainable_anchors: bool,
        separate_proj: bool,
    ):
        super().__init__()
        self.anchor_source = str(anchor_source)
        self.gpt2 = GPT2Model.from_pretrained(
            pretrained_name,
            local_files_only=bool(local_files_only),
        )
        hidden_size = int(self.gpt2.config.hidden_size)
        self.order_embedding = nn.Embedding(int(vocab_size), hidden_size)
        self.head = nn.Linear(hidden_size, int(vocab_size))
        self.separate_proj = bool(separate_proj)
        if self.separate_proj:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.a_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Anchor bank init
        if self.anchor_source == "sentence":
            if not sentences:
                raise ValueError("anchor_source='sentence' requires non-empty sentences")
            tok = AutoTokenizer.from_pretrained(
                pretrained_name,
                local_files_only=bool(local_files_only),
            )
            dev = next(self.gpt2.parameters()).device
            anchor_raw = build_anchor_matrix_mean_last_hidden(
                self.gpt2,
                sentences,
                tok,
                anchor_max_tokens=anchor_max_tokens,
                device=dev,
            )
            anchor_token_ids_tensor = None
        elif self.anchor_source == "freq":
            if not anchor_token_ids:
                raise ValueError("anchor_source='freq' requires anchor_token_ids")
            ids = torch.tensor(anchor_token_ids, dtype=torch.long, device=self.order_embedding.weight.device)
            anchor_raw = self.order_embedding.weight.index_select(0, ids).detach().clone()
            anchor_token_ids_tensor = ids.detach().cpu()
        elif self.anchor_source == "gpt2_vocab":
            if not anchor_vocab_ids:
                raise ValueError("anchor_source='gpt2_vocab' requires anchor_vocab_ids")
            ids = torch.tensor(anchor_vocab_ids, dtype=torch.long, device=self.gpt2.wte.weight.device)
            # GPT-2 token embeddings (semantic word tokens)
            anchor_raw = self.gpt2.wte.weight.index_select(0, ids).detach().clone()
            anchor_token_ids_tensor = ids.detach().cpu()
        else:
            raise ValueError(f"Unknown anchor_source: {self.anchor_source!r}")
        if bool(center_anchors):
            anchor_raw = anchor_raw - anchor_raw.mean(dim=0, keepdim=True)

        if bool(trainable_anchors):
            self.anchor_raw = nn.Parameter(anchor_raw, requires_grad=True)
        else:
            self.register_buffer("anchor_raw", anchor_raw)

        if anchor_token_ids_tensor is not None:
            self.register_buffer("anchor_token_ids", anchor_token_ids_tensor)
        else:
            self.anchor_token_ids = None

        # Optional learnable mapping f(·) on anchor vectors (before routing/prefix)
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
        self.anchor_count = int(anchor_raw.shape[0])
        self.topk_anchors = int(min(topk_anchors, self.anchor_count))
        self.center_anchors = bool(center_anchors)
        self.trainable_anchors = bool(trainable_anchors)

    def forward(self, x: torch.Tensor, forced_topk_idx: torch.Tensor | None = None):
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
        qn = F.normalize(q, dim=1)
        an = F.normalize(a, dim=1)
        scores = torch.matmul(qn, an.t())
        k = self.topk_anchors
        if forced_topk_idx is None:
            topk = torch.topk(scores, k=k, dim=1)
            topk_scores = topk.values
            topk_idx = topk.indices
        else:
            topk_idx = forced_topk_idx
            topk_scores = scores.gather(1, topk_idx)
        align_mean = topk_scores.mean()
        prompt = a[topk_idx]
        gpt_input = torch.cat([prompt, emb], dim=1)
        outputs = self.gpt2(inputs_embeds=gpt_input)
        h = outputs.last_hidden_state
        logits = self.head(h[:, -1, :])
        return logits, scores, align_mean, topk_idx, topk_scores


@torch.no_grad()
def eval_ce(model, loader, device, criterion):
    model.eval()
    total = 0.0
    cnt = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _, _, _, _ = model(x)
        ce = criterion(logits, y)
        total += ce.item()
        cnt += 1
    return total / max(cnt, 1)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


@torch.no_grad()
def _router_diag_from_scores(
    scores: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_scores: torch.Tensor,
    *,
    softmax_temp: float,
) -> dict:
    """
    Compute router diagnostics from raw cosine scores.
    This is diagnostic-only and does not affect training behavior.
    """
    bsz, a = scores.shape
    k = topk_idx.shape[1]

    # top-1 usage
    top1_idx = topk_idx[:, 0]
    top1_unique = int(torch.unique(top1_idx).numel())
    top1_counts = torch.bincount(top1_idx, minlength=a).float()
    top1_prob = top1_counts / float(max(bsz, 1))
    top1_max_share = float(top1_prob.max().item())

    # HHI (Herfindahl-Hirschman index): sum p^2 (collapse -> large)
    hhi = float((top1_prob * top1_prob).sum().item())

    # Diversity summaries over the top-1 histogram (not the diagnostic softmax)
    top1_entropy = -(top1_prob * (top1_prob.clamp_min(1e-12)).log()).sum()
    top1_neff = torch.exp(top1_entropy)
    unif = torch.full_like(top1_prob, 1.0 / float(a))
    top1_kl_to_uniform = float((top1_prob * (top1_prob.clamp_min(1e-12) / unif).log()).sum().item())

    # Top-K usage histogram (counts how often anchors appear in the retrieved set)
    flat = topk_idx.reshape(-1)
    topk_counts = torch.bincount(flat, minlength=a).float()
    topk_prob = topk_counts / float(max(int(flat.numel()), 1))
    topk_unique = int(torch.unique(flat).numel())
    topk_max_share = float(topk_prob.max().item())
    topk_entropy = -(topk_prob * (topk_prob.clamp_min(1e-12)).log()).sum()
    topk_neff = torch.exp(topk_entropy)
    topk_kl_to_uniform = float((topk_prob * (topk_prob.clamp_min(1e-12) / unif).log()).sum().item())

    # Diagnostic softmax over all anchors for entropy / Neff
    tau = float(max(softmax_temp, 1e-6))
    p = torch.softmax(scores / tau, dim=1)
    ent = -(p * (p.clamp_min(1e-12)).log()).sum(dim=1)  # [B]
    neff = torch.exp(ent)  # effective anchors

    # gaps
    top2 = torch.topk(scores, k=min(2, a), dim=1).values
    gap_12 = (top2[:, 0] - top2[:, 1]) if top2.shape[1] == 2 else torch.zeros(bsz, device=scores.device)
    top_k1 = torch.topk(scores, k=min(k + 1, a), dim=1).values
    gap_k_k1 = (
        (top_k1[:, k - 1] - top_k1[:, k]) if top_k1.shape[1] == (k + 1) else torch.zeros(bsz, device=scores.device)
    )

    # top-k mass under diagnostic softmax
    topk_mass = p.gather(1, topk_idx).sum(dim=1)

    # score scale
    score_mean = scores.mean()
    score_std = scores.std(unbiased=False)

    # summarize
    def _summ(x: torch.Tensor) -> dict:
        x = x.detach().float().cpu()
        return {
            "mean": float(x.mean().item()),
            "std": float(x.std(unbiased=False).item()),
            "p10": float(torch.quantile(x, 0.10).item()),
            "p50": float(torch.quantile(x, 0.50).item()),
            "p90": float(torch.quantile(x, 0.90).item()),
            "min": float(x.min().item()),
            "max": float(x.max().item()),
        }

    # log top-5 anchors by count
    top5 = torch.topk(top1_counts, k=min(5, a)).indices.cpu().tolist()
    top5_counts = {int(i): int(top1_counts[int(i)].item()) for i in top5}

    return {
        "bsz": int(bsz),
        "anchor_count": int(a),
        "k": int(k),
        "top1_unique": int(top1_unique),
        "topk_unique": int(topk_unique),
        "top1_top5_counts": top5_counts,
        "hhi_top1": float(hhi),
        "top1_max_share": float(top1_max_share),
        "topk_max_share": float(topk_max_share),
        "top1_entropy_hist": float(top1_entropy.detach().item()),
        "top1_neff_hist": float(top1_neff.detach().item()),
        "top1_kl_to_uniform": float(top1_kl_to_uniform),
        "topk_entropy_hist": float(topk_entropy.detach().item()),
        "topk_neff_hist": float(topk_neff.detach().item()),
        "topk_kl_to_uniform": float(topk_kl_to_uniform),
        "entropy_softmax": _summ(ent),
        "neff_softmax": _summ(neff),
        "gap_1_2": _summ(gap_12),
        "gap_k_k1": _summ(gap_k_k1),
        "topk_mass_softmax": _summ(topk_mass),
        "score_mean": float(score_mean.item()),
        "score_std": float(score_std.item()),
        "align_topk_mean": float(topk_scores.mean().detach().item()),
    }


@torch.no_grad()
def _anchor_geometry_diag(model: OrderGPT2_SentencePresetS2IP) -> dict:
    """
    Anchor bank geometry diagnostics (projected anchors).
    """
    dev = next(model.parameters()).device
    a_raw = model.anchor_raw
    if getattr(model, "anchor_map", None) is not None:
        a_raw = model.anchor_map(a_raw)
    if model.separate_proj:
        a = model.a_proj(a_raw)
    else:
        a = model.proj(a_raw)
    an = F.normalize(a, dim=1)
    norms = a.norm(dim=1)
    # pairwise cosine (A x A), A=128 -> OK
    cos = torch.matmul(an, an.t())
    a_cnt = cos.shape[0]
    mask = ~torch.eye(a_cnt, dtype=torch.bool, device=cos.device)
    off = cos[mask]
    return {
        "anchor_count": int(a_cnt),
        "proj_anchor_norm": {
            "mean": float(norms.mean().item()),
            "std": float(norms.std(unbiased=False).item()),
            "min": float(norms.min().item()),
            "max": float(norms.max().item()),
        },
        "pairwise_cos_offdiag": {
            "mean": float(off.mean().item()),
            "std": float(off.std(unbiased=False).item()),
            "p90": float(torch.quantile(off, 0.90).item()),
            "p99": float(torch.quantile(off, 0.99).item()),
            "max": float(off.max().item()),
        },
        "device": str(dev),
        "trainable_anchors": bool(getattr(model, "trainable_anchors", False)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stock", required=True)
    p.add_argument("--day", default="", help="YYYYMMDD in preprocess filename.")
    p.add_argument(
        "--data-dir",
        default=(
            "/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/processed_real_flow/"
            "pool_0709_0710_openbidanchor_txncomplete"
        ),
    )
    p.add_argument("--preset-anchors", default=DEFAULT_PRESET_ANCHORS)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--window-len", type=int, default=WINDOW_LEN)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--epochs", type=int, default=3, help="Max epochs (default 3).")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--anchor-count", type=int, default=128)
    p.add_argument("--topk-anchors", type=int, default=5)
    p.add_argument("--anchor-max-tokens", type=int, default=128)
    p.add_argument(
        "--anchor-source",
        choices=("sentence", "freq", "gpt2_vocab"),
        default="sentence",
        help="Anchor bank source: preset sentences (default), frequency-based order_token codebook, or GPT-2 vocab token embeddings.",
    )
    p.add_argument(
        "--anchor-vocab-select",
        choices=("random", "low_id"),
        default="random",
        help="If --anchor-source=gpt2_vocab: how to pick V' GPT-2 token ids.",
    )
    p.add_argument(
        "--anchor-map",
        choices=("none", "linear", "mlp"),
        default="none",
        help="Optional learnable mapping f(·) applied to anchor vectors before routing/prefix.",
    )
    p.add_argument(
        "--center-anchors",
        action="store_true",
        help="Center anchor bank by subtracting mean over anchors (feature-wise).",
    )
    p.add_argument(
        "--trainable-anchors",
        action="store_true",
        help="Make anchor_raw an nn.Parameter (otherwise frozen buffer).",
    )
    p.add_argument(
        "--separate-proj",
        action="store_true",
        help="Use separate q_proj and a_proj (otherwise a shared proj).",
    )
    p.add_argument("--align-lambda", type=float, default=5e-2)
    p.add_argument("--align-warmup-steps", type=int, default=20)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--pretrained-backbone-name", default="gpt2")
    p.add_argument(
        "--pretrained-local-only",
        action="store_true",
        help="HF local_files_only=True (no download).",
    )
    p.add_argument("--no-sampling-plot", action="store_true")
    # Phase-1 router diagnostics (log-only; no behavior change)
    p.add_argument(
        "--router-diag",
        action="store_true",
        help="Emit router health diagnostics (log + optional JSONL).",
    )
    p.add_argument(
        "--router-diag-every-steps",
        type=int,
        default=200,
        help="Emit router diagnostics every N training steps (default 200).",
    )
    p.add_argument(
        "--router-diag-softmax-temp",
        type=float,
        default=1.0,
        help="Temperature for diagnostic softmax(scores/tau) when computing entropy/Neff (default 1.0).",
    )
    p.add_argument(
        "--router-diag-jsonl",
        default="",
        help="If set, append router diagnostics JSON lines to this path.",
    )
    p.add_argument(
        "--prefix-shuffle-probe",
        action="store_true",
        help="Once per epoch, run a small probe comparing CE with normal vs shuffled prefix.",
    )
    p.add_argument(
        "--prefix-shuffle-probe-batches",
        type=int,
        default=2,
        help="How many val batches to use for the shuffle-prefix CE probe (default 2).",
    )
    args = p.parse_args()

    stock = str(args.stock).strip()
    stock_tag = stock.replace("_", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sample_gen = torch.Generator()
    sample_gen.manual_seed(args.seed + 999)

    data_path = find_stock_joblib(args.data_dir, stock, day=args.day)
    out_base = os.path.join(args.output_root, stock_tag)
    model_cache_dir = os.path.join(out_base, "model_cache")
    plot_dir = os.path.join(out_base, "plots")
    os.makedirs(model_cache_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    day_tag = args.day.strip() if args.day.strip() else "anyday"
    exp_name = (
        f"preGPT2sentS2IP_{day_tag}_txncomplete_{stock_tag}_win{args.window_len}"
        f"_k{args.topk_anchors}"
    )
    timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_best_ckpt_path = os.path.join(model_cache_dir, f"{exp_name}_{timestamp_run}_best.pt")
    global_meta_path = os.path.join(model_cache_dir, f"{exp_name}_{timestamp_run}_meta.json")

    print(f"[device] {device} | stock={stock} | sentence_preset_s2ip | topk={args.topk_anchors}")
    print(f"[data] {data_path}")

    df = joblib.load(data_path)
    df = df[df["SecurityID"] == stock]
    df = df.sort_values(
        ["SecurityID", "TransactDT_MS", "ChannelNo", "ApplSeqNum"],
        kind="mergesort",
    )
    tok_max = int(df["order_token"].max()) if len(df) else -1
    if tok_max >= args.vocab_size:
        raise ValueError(f"order_token max={tok_max} >= vocab_size={args.vocab_size}")

    # Anchor source selection
    sentences: list[str] | None = None
    anchor_token_ids: list[int] | None = None
    anchor_token_counts: list[int] | None = None
    anchor_vocab_ids: list[int] | None = None
    if str(args.anchor_source) == "sentence":
        sentences = load_anchor_sentences(args.preset_anchors, args.anchor_count)
        print(
            f"[anchors] source=sentence | {args.preset_anchors} | n={len(sentences)} | mean_last_hidden_over_valid_tokens"
        )
    elif str(args.anchor_source) == "freq":
        vc = df["order_token"].value_counts()
        top = vc.head(int(args.anchor_count))
        anchor_token_ids = [int(i) for i in top.index.tolist()]
        anchor_token_counts = [int(c) for c in top.values.tolist()]
        print(f"[anchors] source=freq | n={len(anchor_token_ids)} | top order_token ids by frequency (train day)")
        print(f"[anchors] top10_ids={anchor_token_ids[:10]} top10_counts={anchor_token_counts[:10]}")
    else:
        # Paper-style: anchors are derived from GPT-2 vocab token embeddings (semantic word tokens).
        # We pick V' token ids and use gpt2.wte.weight rows as anchor vectors.
        # Selection is a proxy for "frequency" since we don't have corpus counts here.
        a = int(args.anchor_count)
        if str(args.anchor_vocab_select) == "low_id":
            anchor_vocab_ids = list(range(a))
        else:
            rng = np.random.RandomState(int(args.seed) + 2026)
            vocab_size = int(GPT2Model.from_pretrained(str(args.pretrained_backbone_name), local_files_only=bool(args.pretrained_local_only)).config.vocab_size)
            anchor_vocab_ids = rng.choice(vocab_size, size=a, replace=False).tolist()
        print(f"[anchors] source=gpt2_vocab | select={args.anchor_vocab_select} | n={len(anchor_vocab_ids)}")
        print(f"[anchors] top10_vocab_ids={anchor_vocab_ids[:10]}")

    train_loader, val_loader, test_loader, split_stats = make_day_loaders_and_datasets(
        df=df,
        window_len=args.window_len,
        batch_size=args.batch_size,
        shuffle_seed=args.seed + 123,
    )

    model = OrderGPT2_SentencePresetS2IP(
        vocab_size=args.vocab_size,
        sentences=sentences,
        anchor_source=str(args.anchor_source),
        anchor_token_ids=anchor_token_ids,
        anchor_vocab_ids=anchor_vocab_ids,
        anchor_map=str(args.anchor_map),
        pretrained_name=str(args.pretrained_backbone_name),
        local_files_only=bool(args.pretrained_local_only),
        topk_anchors=int(args.topk_anchors),
        anchor_max_tokens=int(args.anchor_max_tokens),
        center_anchors=bool(args.center_anchors),
        trainable_anchors=bool(args.trainable_anchors),
        separate_proj=bool(args.separate_proj),
    ).to(device)
    if args.grad_checkpointing:
        model.gpt2.gradient_checkpointing_enable()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    use_amp = bool(args.amp) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_meta = {
        "exp_name": exp_name,
        "timestamp": timestamp_run,
        "seed": args.seed,
        "stock": stock,
        "train_day_in_filename": args.day.strip() or None,
        "data_path": data_path,
        "data_dir": args.data_dir,
        "vocab_size": args.vocab_size,
        "preset_anchors_path": args.preset_anchors,
        "sentence_anchors": {
            "count": int(args.anchor_count),
            "anchor_source": str(args.anchor_source),
            "anchor_map": str(args.anchor_map),
            "vector": (
                "gpt2_mean_last_hidden_valid_tokens"
                if str(args.anchor_source) == "sentence"
                else (
                    "order_embedding_rows_selected_by_freq"
                    if str(args.anchor_source) == "freq"
                    else "gpt2_wte_rows_selected_as_semantic_anchors"
                )
            ),
            "freq_token_ids": anchor_token_ids if str(args.anchor_source) == "freq" else None,
            "freq_token_counts": anchor_token_counts if str(args.anchor_source) == "freq" else None,
            "gpt2_vocab_ids": anchor_vocab_ids if str(args.anchor_source) == "gpt2_vocab" else None,
            "gpt2_vocab_select": str(args.anchor_vocab_select) if str(args.anchor_source) == "gpt2_vocab" else None,
            "topk_prepend": int(args.topk_anchors),
            "no_softmax_prefix": True,
            "center_anchors": bool(args.center_anchors),
            "trainable_anchors": bool(args.trainable_anchors),
            "separate_proj": bool(args.separate_proj),
        },
        "align_reg_s2ip_style": {
            "lambda": float(args.align_lambda),
            "warmup_steps": int(args.align_warmup_steps),
            "term": "CE - lambda * mean(topK_cosine_scores)",
        },
        "pretrained_backbone": str(args.pretrained_backbone_name),
        "window_len": args.window_len,
        "batch_size": args.batch_size,
        "grad_accum_steps": int(args.grad_accum_steps),
        "amp": use_amp,
        "lr": args.lr,
        "epochs_max": args.epochs,
        "early_stop_patience": int(args.patience),
        "split": split_stats,
        "epochs": [],
        "output": {"base": out_base, "model_cache": model_cache_dir, "plots": plot_dir},
        "router_diagnostics": {
            "enabled": bool(args.router_diag),
            "every_steps": int(args.router_diag_every_steps),
            "softmax_temp": float(args.router_diag_softmax_temp),
            "jsonl_path": str(args.router_diag_jsonl) if str(args.router_diag_jsonl).strip() else None,
            "prefix_shuffle_probe": bool(args.prefix_shuffle_probe),
            "prefix_shuffle_probe_batches": int(args.prefix_shuffle_probe_batches),
        },
    }

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_state = None
    global_step = 0

    jsonl_path = str(args.router_diag_jsonl).strip()
    if jsonl_path:
        os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)) or ".", exist_ok=True)
        # write a short header line once (as JSON) to aid later parsing
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "run_header", "exp_name": exp_name, "timestamp": timestamp_run, "stock": stock}) + "\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_ce_sum = 0.0
        epoch_align_sum = 0.0
        epoch_tot_sum = 0.0
        epoch_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        optimizer.zero_grad(set_to_none=True)

        for x, y in pbar:
            global_step += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            autocast_ctx = (
                torch.cuda.amp.autocast(enabled=use_amp) if device.type == "cuda" else nullcontext()
            )
            with autocast_ctx:
                logits, scores, align_mean, topk_idx, topk_scores = model(x)
                ce_loss = criterion(logits, y)
                lam = (
                    0.0
                    if global_step <= int(args.align_warmup_steps)
                    else float(args.align_lambda)
                )
                total_loss = ce_loss - lam * align_mean
                total_loss = total_loss / max(int(args.grad_accum_steps), 1)

            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if (epoch_batches + 1) % max(int(args.grad_accum_steps), 1) == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_ce_sum += ce_loss.item()
            epoch_align_sum += float(align_mean.detach().item())
            epoch_tot_sum += (ce_loss - lam * align_mean).item()
            epoch_batches += 1
            if (epoch_batches % PRINT_EVERY_STEPS) == 0:
                pbar.set_postfix(
                    ce=f"{ce_loss.item():.3f}",
                    align=f"{align_mean.item():.3f}",
                    lam=f"{lam:.0e}",
                )

            if bool(args.router_diag) and (global_step % max(int(args.router_diag_every_steps), 1) == 0):
                diag = _router_diag_from_scores(
                    scores.detach(),
                    topk_idx.detach(),
                    topk_scores.detach(),
                    softmax_temp=float(args.router_diag_softmax_temp),
                )
                payload = {
                    "type": "router_health",
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "ce": float(ce_loss.detach().item()),
                    "align_mean": float(align_mean.detach().item()),
                    "lambda_align": float(lam),
                    "router": diag,
                }
                print("[router_health]", json.dumps(payload, sort_keys=True))
                if jsonl_path:
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(payload) + "\n")

        if (epoch_batches % max(int(args.grad_accum_steps), 1)) != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_ce = epoch_ce_sum / max(epoch_batches, 1)
        train_align = epoch_align_sum / max(epoch_batches, 1)
        val_ce = eval_ce(model, val_loader, device, criterion)
        test_ce = eval_ce(model, test_loader, device, criterion)
        if bool(args.router_diag):
            geo = _anchor_geometry_diag(model)
            geo_payload = {"type": "anchor_geometry", "epoch": int(epoch), "global_step": int(global_step), "geometry": geo}
            print("[anchor_geometry]", json.dumps(geo_payload, sort_keys=True))
            if jsonl_path:
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(geo_payload) + "\n")

        if bool(args.prefix_shuffle_probe):
            model.eval()
            probe_batches = max(int(args.prefix_shuffle_probe_batches), 1)
            ce_norm_sum = 0.0
            ce_shuf_sum = 0.0
            n_probe = 0
            with torch.no_grad():
                for bi, (vx, vy) in enumerate(val_loader):
                    if bi >= probe_batches:
                        break
                    vx = vx.to(device, non_blocking=True)
                    vy = vy.to(device, non_blocking=True)
                    logits_n, scores_n, _, topk_idx_n, _ = model(vx)
                    ce_n = criterion(logits_n, vy)
                    perm = torch.randperm(topk_idx_n.shape[0], device=device)
                    forced = topk_idx_n[perm]
                    logits_s, _, _, _, _ = model(vx, forced_topk_idx=forced)
                    ce_s = criterion(logits_s, vy)
                    ce_norm_sum += float(ce_n.detach().item())
                    ce_shuf_sum += float(ce_s.detach().item())
                    n_probe += 1
            if n_probe > 0:
                ce_norm = ce_norm_sum / n_probe
                ce_shuf = ce_shuf_sum / n_probe
                delta = ce_shuf - ce_norm
                probe_payload = {
                    "type": "prefix_shuffle_probe",
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "val_ce_normal": float(ce_norm),
                    "val_ce_shuffled": float(ce_shuf),
                    "delta": float(delta),
                    "batches": int(n_probe),
                }
                print("[prefix_shuffle_probe]", json.dumps(probe_payload, sort_keys=True))
                if jsonl_path:
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(probe_payload) + "\n")
        print(
            f"\n[Epoch {epoch}] train_ce={train_ce:.4f} train_align={train_align:.4f} | "
            f"val_ce={val_ce:.4f} test_ce={test_ce:.4f}"
        )

        run_meta["epochs"].append(
            {
                "epoch": epoch,
                "train_ce": float(train_ce),
                "train_align": float(train_align),
                "val_ce": float(val_ce),
                "test_ce": float(test_ce),
            }
        )

        if (best_val - val_ce) > MIN_DELTA:
            best_val = val_ce
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "exp_name": exp_name,
                    "timestamp": timestamp_run,
                    "seed": args.seed,
                    "stock": stock,
                    "best_epoch": best_epoch,
                    "best_val_ce": float(best_val),
                    "vocab_size": args.vocab_size,
                    "model_state_dict": best_state,
                    "config": run_meta,
                    "variant": "sentence_preset_s2ip",
                },
                global_best_ckpt_path,
            )
            run_meta["best_epoch"] = best_epoch
            run_meta["best_val_ce"] = float(best_val)
            run_meta["ckpt_path"] = global_best_ckpt_path
            with open(global_meta_path, "w", encoding="utf-8") as f:
                json.dump(run_meta, f, indent=2)
            print(f"  -> new best val_ce={best_val:.4f} -> {global_best_ckpt_path}")
        else:
            bad_epochs += 1
            print(f"  -> no improvement ({bad_epochs}/{args.patience})")

        if bad_epochs >= int(args.patience):
            print(f"Early stop at epoch {epoch}. Best val_ce={best_val:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    final_test_ce = eval_ce(model, test_loader, device, criterion)
    print("\n==================== FINAL ====================")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val CE: {best_val:.6f}")
    print(f"Test CE (best weights): {final_test_ce:.6f}")
    print(f"Checkpoint: {global_best_ckpt_path}")
    print("================================================\n")

    run_meta["final_test_ce_best_weights"] = float(final_test_ce)
    with open(global_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    if not args.no_sampling_plot and plt is not None:

        @torch.no_grad()
        def diagnostics_sampling_plot():
            model.eval()
            all_preds = []
            for x, _ in tqdm(test_loader, desc="Test sampling", unit="batch"):
                x = x.to(device, non_blocking=True)
                logits, _, _ = model(x)
                probs = F.softmax(logits, dim=1)
                preds = torch.multinomial(probs.cpu(), 1, generator=sample_gen).squeeze(1)
                all_preds.append(preds)
            preds = torch.cat(all_preds).numpy()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(plot_dir, f"{exp_name}_testpred_sampling_{ts}.png")
            plt.figure(figsize=(10, 6))
            plt.hist(preds, bins=200)
            plt.title(f"{exp_name} — test prediction distribution (sampling)")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            print("Saved plot:", out_path)

        diagnostics_sampling_plot()
    elif not args.no_sampling_plot and plt is None:
        print("[skip] matplotlib not available; no sampling plot.")

    print("Run complete.")


if __name__ == "__main__":
    main()
