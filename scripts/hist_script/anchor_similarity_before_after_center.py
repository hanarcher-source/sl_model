#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick anchor similarity diagnostic for the current sentence preset anchors.

Anchor vector = GPT-2 last-layer hidden-state mean-pool over valid tokens (masked),
same as our sentence-preset pipeline.

Prints pairwise cosine similarity stats (off-diagonal) for:
  (1) raw anchors
  (2) centered anchors (demean feature-wise)

Pooling modes (GPT-2 last-layer hidden states):
  mean — masked mean over valid tokens (training default)
  last — hidden at last *valid* (non-pad) position per sentence (causal context)
  both — print mean then last
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2Model


def _clean_anchor_line(line: str) -> str:
    s = line.strip()
    if not s:
        return ""
    s = re.sub(r"^\s*\d+\.\s*", "", s).strip()
    if (s.startswith("\u201c") and s.endswith("\u201d")) or (
        s.startswith('"') and s.endswith('"')
    ) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s.strip("\u201c\u201d\"' ").strip()


def load_anchor_sentences(path: str, expected_n: int) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            t = _clean_anchor_line(raw)
            if t:
                lines.append(t)
    if len(lines) < expected_n:
        raise RuntimeError(f"Need >= {expected_n} anchors, got {len(lines)}")
    return lines[:expected_n]


def _pairwise_offdiag_vals(S: np.ndarray) -> np.ndarray:
    a = S.shape[0]
    mask = ~np.eye(a, dtype=bool)
    return S[mask]


def _describe(vals: np.ndarray, name: str) -> None:
    qs = [0.0, 0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99, 1.0]
    qv = np.quantile(vals, qs)
    print(f"\n[{name}]")
    print(f"  count: {vals.size}")
    print(f"  mean/std: {vals.mean():.6f} / {vals.std():.6f}")
    print(f"  min/max:  {vals.min():.6f} / {vals.max():.6f}")
    print("  quantiles:")
    for q, v in zip(qs, qv):
        print(f"    p{int(q*100):02d}: {v:.6f}")


def _forward_last_hidden(
    sentences: List[str],
    *,
    pretrained_name: str,
    max_len: int,
    device: torch.device,
    local_files_only: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model = GPT2Model.from_pretrained(pretrained_name, local_files_only=bool(local_files_only)).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(pretrained_name, use_fast=True, local_files_only=bool(local_files_only))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    toks = tok(
        sentences,
        padding=True,
        truncation=True,
        max_length=int(max_len),
        return_tensors="pt",
    )
    input_ids = toks["input_ids"].to(device)
    attn = toks["attention_mask"].float().to(device)
    out = model(input_ids=input_ids, attention_mask=attn.long())
    hs = out.last_hidden_state
    return hs, attn


@torch.no_grad()
def anchors_from_hidden(
    hs: torch.Tensor,
    attn: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """[N,H] on CPU. pooling: 'mean' or 'last'."""
    if pooling == "mean":
        mask = attn.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return ((hs * mask).sum(dim=1) / denom).detach().cpu()
    if pooling == "last":
        # last valid (non-pad) index per row
        lengths = attn.long().sum(dim=1)
        last_idx = (lengths - 1).clamp_min(0)
        b, t, h = hs.shape
        idx = last_idx.view(b, 1, 1).expand(b, 1, h)
        return hs.gather(1, idx).squeeze(1).detach().cpu()
    raise ValueError(f"pooling must be mean or last, got {pooling!r}")


@torch.no_grad()
def compute_anchor_last_hidden_meanpool(
    sentences: List[str],
    *,
    pretrained_name: str,
    max_len: int,
    device: torch.device,
    local_files_only: bool,
) -> torch.Tensor:
    hs, attn = _forward_last_hidden(
        sentences,
        pretrained_name=pretrained_name,
        max_len=max_len,
        device=device,
        local_files_only=local_files_only,
    )
    return anchors_from_hidden(hs, attn, "mean")


@torch.no_grad()
def compute_anchor_last_hidden_last_token(
    sentences: List[str],
    *,
    pretrained_name: str,
    max_len: int,
    device: torch.device,
    local_files_only: bool,
) -> torch.Tensor:
    hs, attn = _forward_last_hidden(
        sentences,
        pretrained_name=pretrained_name,
        max_len=max_len,
        device=device,
        local_files_only=local_files_only,
    )
    return anchors_from_hidden(hs, attn, "last")


def cosine_offdiag(anchor: torch.Tensor) -> np.ndarray:
    xn = F.normalize(anchor, dim=1)
    S = (xn @ xn.t()).cpu().numpy()
    return _pairwise_offdiag_vals(S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--anchor-text-path",
        default="/finance_ML/zhanghaohan/GPT2_new_head/sentence_semantic_anchor/preset_anchors/preset_anchors.txt",
    )
    ap.add_argument("--anchor-count", type=int, default=128)
    ap.add_argument("--anchor-max-tokens", type=int, default=128)
    ap.add_argument("--pretrained-name", default="gpt2")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument(
        "--pooling",
        choices=("mean", "last", "both"),
        default="both",
        help="mean = masked mean over tokens; last = last valid token hidden; both = run both.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Anchor similarity (raw vs centered) ===")
    print("device:", device)
    print("anchor file:", args.anchor_text_path)
    print("anchor_count:", args.anchor_count, "anchor_max_tokens:", args.anchor_max_tokens)
    print("pretrained:", args.pretrained_name, "local_files_only:", bool(args.local_files_only))
    print("pooling:", args.pooling)

    sents = load_anchor_sentences(args.anchor_text_path, expected_n=int(args.anchor_count))

    modes: List[str]
    if args.pooling == "both":
        modes = ["mean", "last"]
    else:
        modes = [str(args.pooling)]

    for mode in modes:
        if mode == "mean":
            anchor = compute_anchor_last_hidden_meanpool(
                sents,
                pretrained_name=str(args.pretrained_name),
                max_len=int(args.anchor_max_tokens),
                device=device,
                local_files_only=bool(args.local_files_only),
            )
        else:
            anchor = compute_anchor_last_hidden_last_token(
                sents,
                pretrained_name=str(args.pretrained_name),
                max_len=int(args.anchor_max_tokens),
                device=device,
                local_files_only=bool(args.local_files_only),
            )
        label = "MEAN pool" if mode == "mean" else "LAST valid token"
        print(f"\n--- {label} ---")
        off_raw = cosine_offdiag(anchor)
        _describe(off_raw, f"Pairwise cosine (raw, {label}, off-diagonal)")

        anchor_c = anchor - anchor.mean(dim=0, keepdim=True)
        off_c = cosine_offdiag(anchor_c)
        _describe(off_c, f"Pairwise cosine (centered, {label}, off-diagonal)")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

