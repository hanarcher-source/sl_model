#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frequency-based codebook anchor similarity diagnostic.

We select top-N most frequent `order_token` ids from a txn-complete joblib and
extract their embedding vectors from a provided model checkpoint (e.g. blankGPT2
no-anchor checkpoint). We then report pairwise cosine similarities:
  (1) raw embedding rows
  (2) centered rows (demean feature-wise across the selected bank)

This helps answer: are token-embedding anchors already diverse in the LOB token
embedding space, or do they also collapse into a shared direction?
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import joblib
import numpy as np
import torch
import torch.nn.functional as F


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


def cosine_offdiag(anchor: torch.Tensor) -> np.ndarray:
    xn = F.normalize(anchor, dim=1)
    S = (xn @ xn.t()).cpu().numpy()
    return _pairwise_offdiag_vals(S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joblib", required=True, help="Processed realflow joblib for a stock/day.")
    ap.add_argument("--stock", required=True, help="SecurityID (e.g. 000617_XSHE).")
    ap.add_argument("--checkpoint", required=True, help="Path to a training checkpoint with order_embedding.")
    ap.add_argument("--anchor-count", type=int, default=128)
    ap.add_argument("--vocab-size", type=int, default=40560)
    ap.add_argument("--topk-print", type=int, default=10)
    args = ap.parse_args()

    df = joblib.load(args.joblib)
    df = df[df["SecurityID"] == str(args.stock)]
    if len(df) == 0:
        raise RuntimeError(f"No rows for stock={args.stock} in {args.joblib}")
    tok_max = int(df["order_token"].max())
    if tok_max >= int(args.vocab_size):
        raise ValueError(f"order_token max={tok_max} >= vocab_size={args.vocab_size}")

    vc = df["order_token"].value_counts()
    top = vc.head(int(args.anchor_count))
    token_ids: List[int] = [int(i) for i in top.index.tolist()]
    token_counts: List[int] = [int(c) for c in top.values.tolist()]

    print("=== Freq codebook anchor similarity ===")
    print("joblib:", args.joblib)
    print("stock:", args.stock)
    print("checkpoint:", args.checkpoint)
    print("anchor_count:", args.anchor_count)
    print("top_ids:", token_ids[: int(args.topk_print)])
    print("top_counts:", token_counts[: int(args.topk_print)])

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    if not isinstance(state, dict):
        raise RuntimeError("Could not locate model_state_dict in checkpoint.")

    key = "order_embedding.weight"
    if key not in state:
        hits = [k for k in state.keys() if k.endswith(key)]
        if not hits:
            raise KeyError(f"Missing {key} in checkpoint state_dict.")
        key = hits[0]
    W = state[key].float()  # [V,H]

    ids = torch.tensor(token_ids, dtype=torch.long)
    A = W.index_select(0, ids)  # [N,H]
    off_raw = cosine_offdiag(A)
    _describe(off_raw, "Pairwise cosine (raw embedding rows, off-diagonal)")

    Ac = A - A.mean(dim=0, keepdim=True)
    off_c = cosine_offdiag(Ac)
    _describe(off_c, "Pairwise cosine (centered rows, off-diagonal)")

    out = {
        "joblib": args.joblib,
        "stock": args.stock,
        "checkpoint": args.checkpoint,
        "anchor_count": int(args.anchor_count),
        "token_ids": token_ids,
        "token_counts": token_counts,
        "raw": {
            "mean": float(off_raw.mean()),
            "std": float(off_raw.std()),
            "min": float(off_raw.min()),
            "max": float(off_raw.max()),
        },
        "centered": {
            "mean": float(off_c.mean()),
            "std": float(off_c.std()),
            "min": float(off_c.min()),
            "max": float(off_c.max()),
        },
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "freq_codebook_anchor_similarity.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote:", out_path)


if __name__ == "__main__":
    main()

