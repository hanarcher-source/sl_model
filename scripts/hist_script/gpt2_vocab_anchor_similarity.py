#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-2 vocab anchor similarity diagnostic (paper-style).

Anchors are selected rows from GPT-2's token embedding table (wte.weight).
We report pairwise cosine similarity off-diagonal for:
  (1) raw selected embeddings
  (2) centered embeddings (demean feature-wise across the selected bank)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Model


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
    ap.add_argument("--pretrained-name", default="gpt2")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--anchor-count", type=int, default=128)
    ap.add_argument("--select", choices=("random", "low_id"), default="random")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = GPT2Model.from_pretrained(str(args.pretrained_name), local_files_only=bool(args.local_files_only))
    W = model.wte.weight.detach().cpu().float()
    V = int(W.shape[0])
    n = int(args.anchor_count)

    if str(args.select) == "low_id":
        ids = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.RandomState(int(args.seed) + 2026)
        ids = rng.choice(V, size=n, replace=False).astype(np.int64)

    A = W.index_select(0, torch.tensor(ids, dtype=torch.long))
    print("=== GPT-2 vocab anchor similarity ===")
    print("pretrained:", args.pretrained_name, "local_files_only:", bool(args.local_files_only))
    print("vocab_size:", V, "anchor_count:", n, "select:", args.select, "seed:", args.seed)
    print("top10_vocab_ids:", ids[:10].tolist())

    off_raw = cosine_offdiag(A)
    _describe(off_raw, "Pairwise cosine (raw GPT-2 wte rows, off-diagonal)")

    Ac = A - A.mean(dim=0, keepdim=True)
    off_c = cosine_offdiag(Ac)
    _describe(off_c, "Pairwise cosine (centered GPT-2 wte rows, off-diagonal)")

    out = {
        "pretrained_name": str(args.pretrained_name),
        "local_files_only": bool(args.local_files_only),
        "vocab_size": V,
        "anchor_count": n,
        "select": str(args.select),
        "seed": int(args.seed),
        "vocab_ids": ids.tolist(),
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
    out_path = os.path.join(os.getcwd(), f"gpt2_vocab_anchor_similarity_{args.select}_n{n}_seed{args.seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote:", out_path)


if __name__ == "__main__":
    main()

