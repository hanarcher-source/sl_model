#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a hybrid book-state model:
  - time is autoregressive (predict snapshot t+1 from previous L seconds)
  - within snapshot, predict 20 level-slots in parallel (20 heads over joint K=P*V codebook)

Data:
  joblib produced by preprocess_bookstate_mdl628_anchor5m_bins_20250709.py
  columns: SecurityID, TransactDT_SEC, book_token_00..19
"""

from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2Config, GPT2Model


SEED = 42
DEFAULT_CONTEXT_SEC = 60
DEFAULT_STRIDE_SEC = 5
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 2e-4
DEFAULT_EPOCHS = 3
DEFAULT_PATIENCE = 3
MIN_DELTA = 1e-4


def _make_dl_gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _as_seconds(ts: torch.Tensor) -> torch.Tensor:
    # ts is int64 nanoseconds in pandas->numpy sometimes; we store as python datetime in joblib,
    # but for continuity we only need ordering, so return placeholder.
    return ts


def _build_valid_starts(times_ns: np.ndarray, context: int, stride: int) -> np.ndarray:
    """
    times_ns: int64 nanoseconds (monotone, per-second)
    valid start i satisfies:
      times[i + context] - times[i] == context seconds
      and no gaps within (per-second grid)
    We approximate by checking endpoints because preprocessing keeps last row per second; any missing second
    will break the endpoint difference.
    """
    if times_ns.size <= context:
        return np.zeros((0,), dtype=np.int64)
    # 1 second in ns
    sec = np.int64(1_000_000_000)
    max_i = times_ns.size - context - 1
    idx = np.arange(0, max_i + 1, max(1, int(stride)), dtype=np.int64)
    ok = (times_ns[idx + context] - times_ns[idx]) == (np.int64(context) * sec)
    return idx[ok]


class BookStateWindowDataset(Dataset):
    def __init__(self, tokens_2d: np.ndarray, times_ns: np.ndarray, context: int, stride: int):
        self.tokens = tokens_2d.astype(np.int64, copy=False)  # [N,20]
        self.times_ns = times_ns.astype(np.int64, copy=False)  # [N]
        self.context = int(context)
        self.starts = _build_valid_starts(self.times_ns, context=int(context), stride=int(stride))

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, idx: int):
        i = int(self.starts[idx])
        x = self.tokens[i : i + self.context]  # [L,20]
        y = self.tokens[i + self.context]  # [20]
        return torch.from_numpy(x), torch.from_numpy(y)


class BookStateTemporalParallelHeads(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        d_model: int = 768,
        n_slots: int = 20,
        *,
        init_pretrained_backbone: bool = False,
        pretrained_backbone_name: str = "gpt2",
        pretrained_local_only: bool = True,
    ):
        super().__init__()
        self.K = int(codebook_size)
        self.n_slots = int(n_slots)

        # Slot-aware code embedding: token embedding + per-slot embedding
        self.tok_emb = nn.Embedding(self.K, int(d_model))
        self.slot_emb = nn.Embedding(self.n_slots, int(d_model))

        # Temporal backbone over per-second snapshots
        # - blank: random init GPT2Config (smallish)
        # - pretrained: load HF GPT-2 and keep inputs_embeds path
        if bool(init_pretrained_backbone):
            self.gpt2 = GPT2Model.from_pretrained(
                str(pretrained_backbone_name),
                local_files_only=bool(pretrained_local_only),
            )
        else:
            cfg = GPT2Config(
                n_embd=int(d_model),
                n_layer=6,
                n_head=12,
                n_positions=1024,
                vocab_size=1,  # unused (we feed inputs_embeds)
            )
            self.gpt2 = GPT2Model(cfg)

        # 20 parallel heads
        self.head = nn.Linear(int(d_model), self.n_slots * self.K)

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """
        x_tokens: [B,L,20] int64
        returns logits: [B,20,K]
        """
        B, L, S = x_tokens.shape
        if S != self.n_slots:
            raise ValueError(f"Expected {self.n_slots} slots, got {S}")
        dev = x_tokens.device
        slot_ids = torch.arange(self.n_slots, device=dev).view(1, 1, self.n_slots)
        emb = self.tok_emb(x_tokens) + self.slot_emb(slot_ids)  # [B,L,S,D]
        snap = emb.mean(dim=2)  # [B,L,D]
        out = self.gpt2(inputs_embeds=snap)
        h_last = out.last_hidden_state[:, -1, :]  # [B,D]
        logits = self.head(h_last).view(B, self.n_slots, self.K)
        return logits


@torch.no_grad()
def _eval_ce(model: nn.Module, loader: DataLoader, device: torch.device, K: int) -> float:
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)  # [B,20]
        logits = model(x)  # [B,20,K]
        loss = 0.0
        for s in range(y.shape[1]):
            loss = loss + crit(logits[:, s, :], y[:, s])
        loss = loss / float(y.shape[1])
        tot += float(loss.item())
        n += 1
    return tot / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-joblib", required=True)
    p.add_argument("--stock", required=True, help="e.g. 000617_XSHE")
    p.add_argument("--output-root", default="/finance_ML/zhanghaohan/stock_language_model/training_runs/bookstate_parallelheads_anchor5m")
    p.add_argument("--context-sec", type=int, default=DEFAULT_CONTEXT_SEC)
    p.add_argument("--stride-sec", type=int, default=DEFAULT_STRIDE_SEC)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--codebook-size", type=int, required=True, help="K = P*V joint vocab size.")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--init-pretrained-backbone", action="store_true", help="Initialize GPT-2 backbone from pretrained weights.")
    p.add_argument("--pretrained-backbone-name", default="gpt2")
    p.add_argument("--pretrained-local-only", action="store_true", help="Use local_files_only=True (no download).")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = joblib.load(args.data_joblib)
    df = df[df["SecurityID"] == str(args.stock)].copy()
    if df.empty:
        raise RuntimeError(f"No rows for stock={args.stock} in {args.data_joblib}")
    df = df.sort_values(["TransactDT_SEC"], kind="mergesort").reset_index(drop=True)

    tok_cols = [f"book_token_{j:02d}" for j in range(20)]
    missing = [c for c in tok_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing token cols: {missing}")

    tokens = df[tok_cols].to_numpy(np.int64)  # [N,20]
    # Convert timestamps to ns since epoch for continuity checks
    times = pd_to_ns(df["TransactDT_SEC"].to_numpy())

    # train/val/test split by time (60/20/20)
    n = tokens.shape[0]
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    n_test = n - n_train - n_val

    tr = slice(0, n_train)
    va = slice(n_train, n_train + n_val)
    te = slice(n_train + n_val, n)

    train_ds = BookStateWindowDataset(tokens[tr], times[tr], context=args.context_sec, stride=args.stride_sec)
    val_ds = BookStateWindowDataset(tokens[va], times[va], context=args.context_sec, stride=max(1, args.stride_sec))
    test_ds = BookStateWindowDataset(tokens[te], times[te], context=args.context_sec, stride=max(1, args.stride_sec))

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(f"Not enough continuous windows. Got lens train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    gen = _make_dl_gen(args.seed + 123)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = BookStateTemporalParallelHeads(
        codebook_size=args.codebook_size,
        init_pretrained_backbone=bool(args.init_pretrained_backbone),
        pretrained_backbone_name=str(args.pretrained_backbone_name),
        pretrained_local_only=bool(args.pretrained_local_only),
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(args.lr))
    use_amp = bool(args.amp) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    crit = nn.CrossEntropyLoss()

    stock_tag = str(args.stock).replace("_", "")
    os.makedirs(args.output_root, exist_ok=True)
    out_base = os.path.join(args.output_root, stock_tag)
    os.makedirs(out_base, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(out_base, f"bookstate_ph_{stock_tag}_{ts}_best.pt")
    meta_path = os.path.join(out_base, f"bookstate_ph_{stock_tag}_{ts}_meta.json")

    run_meta = {
        "timestamp": ts,
        "stock": args.stock,
        "data_joblib": args.data_joblib,
        "context_sec": int(args.context_sec),
        "stride_sec": int(args.stride_sec),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "epochs_max": int(args.epochs),
        "patience": int(args.patience),
        "codebook_size": int(args.codebook_size),
        "backbone_init": (
            f"pretrained:{args.pretrained_backbone_name}"
            if bool(args.init_pretrained_backbone)
            else "blank_random_config"
        ),
        "pretrained_local_only": bool(args.pretrained_local_only),
        "splits": {"n": int(n), "train": int(n_train), "val": int(n_val), "test": int(n_test)},
        "windows": {"train": int(len(train_ds)), "val": int(len(val_ds)), "test": int(len(test_ds))},
        "epochs": [],
    }

    best_val = float("inf")
    bad = 0
    best_state = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        ce_sum = 0.0
        n_batches = 0
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp) if device.type == "cuda" else nullcontext()
            with autocast_ctx:
                logits = model(x)  # [B,20,K]
                loss = 0.0
                for s in range(y.shape[1]):
                    loss = loss + crit(logits[:, s, :], y[:, s])
                loss = loss / float(y.shape[1])
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            ce_sum += float(loss.item())
            n_batches += 1
            if n_batches % 50 == 0:
                pbar.set_postfix(train_ce=f"{loss.item():.3f}")

        train_ce = ce_sum / max(n_batches, 1)
        val_ce = _eval_ce(model, val_loader, device, K=int(args.codebook_size))
        test_ce = _eval_ce(model, test_loader, device, K=int(args.codebook_size))
        print(f"\n[Epoch {epoch}] train_ce={train_ce:.4f} val_ce={val_ce:.4f} test_ce={test_ce:.4f}")
        run_meta["epochs"].append({"epoch": epoch, "train_ce": float(train_ce), "val_ce": float(val_ce), "test_ce": float(test_ce)})

        if (best_val - val_ce) > MIN_DELTA:
            best_val = float(val_ce)
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "config": run_meta, "best_val": best_val}, ckpt_path)
            run_meta["best_val"] = best_val
            run_meta["ckpt_path"] = ckpt_path
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(run_meta, f, indent=2)
            print("  -> new best", ckpt_path)
        else:
            bad += 1
            print(f"  -> no improvement ({bad}/{args.patience})")
            if bad >= int(args.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_test = _eval_ce(model, test_loader, device, K=int(args.codebook_size))
    run_meta["final_test_ce_best_weights"] = float(final_test)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    print("[done] best_val", best_val, "final_test", final_test)
    print("[out]", ckpt_path)


def pd_to_ns(arr) -> np.ndarray:
    # pandas Timestamp array -> int64 ns
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.datetime64):
        return a.astype("datetime64[ns]").astype(np.int64)
    # object timestamps
    return pd.Series(a).astype("datetime64[ns]").to_numpy().astype(np.int64)


if __name__ == "__main__":
    main()

