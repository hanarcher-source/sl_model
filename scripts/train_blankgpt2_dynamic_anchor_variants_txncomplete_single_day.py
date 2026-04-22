#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic-anchor variants (single day/stock, pooled preprocess).

Two variants designed to reduce harm on stocks where the original prepend-token anchor hurts.

Variant A: dyn_gatebias
  - Router: mean-pool query (same as V5 baseline)
  - Anchor: softmax mixture over learnable anchors
  - Injection: gated residual bias on token embeddings (no extra token)
  - Objective: CE + warmup * (gated margin reg on top-2 anchor score gap)

Variant B: dyn_attnpool_topk
  - Router: attention pooling over window (learned router query vector)
  - Anchor: sparse top-k mixture (k configurable, default 4)
  - Injection: prepend weighted anchor token (V5-style)
  - Objective: CE + (entropy bonus + load-balance KL) on router probs (scheduled)

Diagnostics (logged per epoch):
  - router entropy, Neff, top-1 usage concentration
  - variant-specific: gate stats (A), router pooling attention entropy (B)
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from train_blankgpt2_openbidanchor_txncomplete_single_day import (  # noqa: E402
    DEFAULT_VOCAB_SIZE,
    LR,
    MIN_DELTA,
    PRINT_EVERY_STEPS,
    SEED,
    WINDOW_LEN,
    BATCH_SIZE,
    find_stock_joblib,
    build_per_stock_splits,
    make_dataloader_generator,
)


DEFAULT_OUTPUT_ROOT = (
    "/finance_ML/zhanghaohan/stock_language_model/training_runs/"
    "pool_0709_0710_train0709_blank_gpt2_dynamic_anchor_variants_win50"
)


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
    train_samples, val_samples, test_samples, split_stats = build_per_stock_splits(df, window_len)
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
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader, split_stats


def _entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=1)


def _neff_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = (p * p).sum(dim=1).clamp_min(eps)
    return 1.0 / denom


@dataclass
class RouterDiag:
    ent_sum: float = 0.0
    ent_sq_sum: float = 0.0
    neff_sum: float = 0.0
    neff_sq_sum: float = 0.0
    count: int = 0
    top1_counts: Optional[Counter] = None
    gate_sum: float = 0.0
    gate_sq_sum: float = 0.0
    gate_min: float = 1e9
    gate_max: float = -1e9
    pool_ent_sum: float = 0.0
    pool_ent_sq_sum: float = 0.0


def _router_diag_update(
    diag: RouterDiag,
    probs: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    pool_attn: Optional[torch.Tensor] = None,
):
    with torch.no_grad():
        ent = _entropy_from_probs(probs)
        neff = _neff_from_probs(probs)
        ent_m = float(ent.mean().item())
        neff_m = float(neff.mean().item())
        diag.ent_sum += ent_m
        diag.ent_sq_sum += ent_m * ent_m
        diag.neff_sum += neff_m
        diag.neff_sq_sum += neff_m * neff_m
        diag.count += 1

        top1 = probs.argmax(dim=1).detach().cpu().numpy().tolist()
        if diag.top1_counts is None:
            diag.top1_counts = Counter()
        diag.top1_counts.update(top1)

        if gate is not None:
            g = gate.detach().float().view(-1)
            gm = float(g.mean().item())
            diag.gate_sum += gm
            diag.gate_sq_sum += gm * gm
            diag.gate_min = min(diag.gate_min, float(g.min().item()))
            diag.gate_max = max(diag.gate_max, float(g.max().item()))

        if pool_attn is not None:
            pe = _entropy_from_probs(pool_attn)
            pem = float(pe.mean().item())
            diag.pool_ent_sum += pem
            diag.pool_ent_sq_sum += pem * pem


def _router_diag_finalize(diag: RouterDiag, anchor_count: int, topk_show: int = 5) -> Dict:
    def mv(sumv, sqsumv, n):
        if n <= 0:
            return None, None
        mean = sumv / n
        var = max(0.0, (sqsumv / n) - mean * mean)
        return mean, var

    ent_mean, ent_var = mv(diag.ent_sum, diag.ent_sq_sum, diag.count)
    neff_mean, neff_var = mv(diag.neff_sum, diag.neff_sq_sum, diag.count)
    out = {
        "router_entropy_mean": ent_mean,
        "router_entropy_std": float(np.sqrt(ent_var)) if ent_var is not None else None,
        "router_neff_mean": neff_mean,
        "router_neff_std": float(np.sqrt(neff_var)) if neff_var is not None else None,
        "router_entropy_max_uniform": float(np.log(float(anchor_count))) if anchor_count > 0 else None,
        "router_neff_max_uniform": float(anchor_count) if anchor_count > 0 else None,
    }

    if diag.top1_counts:
        total = sum(diag.top1_counts.values())
        top = diag.top1_counts.most_common(topk_show)
        out["router_top1_total"] = int(total)
        out["router_top1_topk"] = [
            {"anchor": int(a), "count": int(c), "pct": float(c) / float(total) if total else None}
            for a, c in top
        ]

    if diag.gate_max > -1e8:
        g_mean, g_var = mv(diag.gate_sum, diag.gate_sq_sum, diag.count)
        out["gate_mean"] = g_mean
        out["gate_std"] = float(np.sqrt(g_var)) if g_var is not None else None
        out["gate_min"] = diag.gate_min
        out["gate_max"] = diag.gate_max

    if diag.pool_ent_sum > 0.0:
        pe_mean, pe_var = mv(diag.pool_ent_sum, diag.pool_ent_sq_sum, diag.count)
        out["pool_attn_entropy_mean"] = pe_mean
        out["pool_attn_entropy_std"] = float(np.sqrt(pe_var)) if pe_var is not None else None

    return out


class DynamicAnchorBase(nn.Module):
    def __init__(self, vocab_size: int, anchor_count: int):
        super().__init__()
        self.gpt2 = GPT2Model(GPT2Config())
        hidden_size = self.gpt2.config.hidden_size
        self.hidden_size = hidden_size
        self.order_embedding = nn.Embedding(int(vocab_size), hidden_size)
        self.dynamic_anchors = nn.Parameter(torch.randn(int(anchor_count), hidden_size) * 0.02)
        self.head = nn.Linear(hidden_size, int(vocab_size))
        self.anchor_count = int(anchor_count)

    def _anchor_scores(self, query: torch.Tensor) -> torch.Tensor:
        return torch.matmul(query, self.dynamic_anchors.t())


class DynamicAnchor_GateBias(DynamicAnchorBase):
    def __init__(self, vocab_size: int, anchor_count: int, gate_dimwise: bool = False):
        super().__init__(vocab_size=vocab_size, anchor_count=anchor_count)
        out_dim = self.hidden_size if gate_dimwise else 1
        self.gate = nn.Linear(self.hidden_size, out_dim)
        self.gate_dimwise = bool(gate_dimwise)

    def forward(self, x):
        emb = self.order_embedding(x)  # [B,T,H]
        query = emb.mean(dim=1)  # [B,H]
        scores = self._anchor_scores(query)  # [B,A]
        probs = torch.softmax(scores, dim=1)
        weighted_anchor = torch.matmul(probs, self.dynamic_anchors)  # [B,H]
        g = torch.sigmoid(self.gate(query))  # [B,1] or [B,H]
        emb2 = emb + g.unsqueeze(1) * weighted_anchor.unsqueeze(1)
        outputs = self.gpt2(inputs_embeds=emb2)
        h = outputs.last_hidden_state
        logits = self.head(h[:, -1, :])
        g_scalar = g.mean(dim=1) if self.gate_dimwise else g.view(-1)
        return logits, probs, scores, g_scalar


class DynamicAnchor_AttnPoolTopK(DynamicAnchorBase):
    def __init__(self, vocab_size: int, anchor_count: int, topk_anchors: int = 4):
        super().__init__(vocab_size=vocab_size, anchor_count=anchor_count)
        self.router_q = nn.Parameter(torch.randn(self.hidden_size) * 0.02)
        self.topk_anchors = int(topk_anchors)

    def forward(self, x):
        emb = self.order_embedding(x)  # [B,T,H]
        attn_scores = (emb * self.router_q.view(1, 1, -1)).sum(dim=2)  # [B,T]
        attn = torch.softmax(attn_scores, dim=1)  # [B,T]
        query = torch.sum(attn.unsqueeze(2) * emb, dim=1)  # [B,H]
        scores = self._anchor_scores(query)  # [B,A]
        k = max(1, min(self.topk_anchors, scores.shape[1]))
        topk_vals, topk_idx = torch.topk(scores, k=k, dim=1)
        masked = torch.full_like(scores, float("-inf"))
        masked.scatter_(1, topk_idx, topk_vals)
        probs = torch.softmax(masked, dim=1)
        weighted_anchor = torch.matmul(probs, self.dynamic_anchors)  # [B,H]
        anchor_token = weighted_anchor.unsqueeze(1)
        gpt_input = torch.cat([anchor_token, emb], dim=1)
        outputs = self.gpt2(inputs_embeds=gpt_input)
        h = outputs.last_hidden_state
        logits = self.head(h[:, -1, :])
        return logits, probs, scores, attn


@torch.no_grad()
def eval_ce(model, loader, device, criterion, variant: str):
    model.eval()
    total = 0.0
    cnt = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _, _, _ = model(x)
        ce = criterion(logits, y)
        total += ce.item()
        cnt += 1
    return total / max(cnt, 1)


def _weight_after_warmup(global_step: int, warmup_steps: int, base: float) -> float:
    if base <= 0:
        return 0.0
    if warmup_steps <= 0:
        return float(base)
    return float(base) if global_step > int(warmup_steps) else 0.0


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
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--window-len", type=int, default=WINDOW_LEN)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--anchor-count", type=int, default=128)
    p.add_argument("--variant", choices=("dyn_gatebias", "dyn_attnpool_topk"), required=True)

    # Variant A
    p.add_argument("--gate-dimwise", action="store_true")

    # Variant B
    p.add_argument("--topk-anchors", type=int, default=4)

    # Objective
    p.add_argument("--reg-mode", choices=("margin_gated", "entropy_lb"), required=True)
    p.add_argument("--reg-warmup-steps", type=int, default=20)
    p.add_argument("--reg-lambda", type=float, default=1e-2)
    p.add_argument("--margin-m", type=float, default=1.0)
    p.add_argument("--ent-lambda", type=float, default=1e-3)
    p.add_argument("--lb-lambda", type=float, default=1e-3)
    p.add_argument("--ent-warmup-steps", type=int, default=20)
    p.add_argument("--lb-warmup-steps", type=int, default=20)

    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-checkpointing", action="store_true")
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
    out_base = os.path.join(args.output_root, stock_tag, args.variant)
    model_cache_dir = os.path.join(out_base, "model_cache")
    plot_dir = os.path.join(out_base, "plots")
    os.makedirs(model_cache_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    day_tag = args.day.strip() if args.day.strip() else "anyday"
    exp_name = f"{args.variant}_{day_tag}_txncomplete_{stock_tag}_win{args.window_len}"
    timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_best_ckpt_path = os.path.join(model_cache_dir, f"{exp_name}_{timestamp_run}_best.pt")
    global_meta_path = os.path.join(model_cache_dir, f"{exp_name}_{timestamp_run}_meta.json")

    print(f"[device] {device} | stock={stock} | variant={args.variant} | reg_mode={args.reg_mode}")
    print(f"[data] {data_path}")

    df = joblib.load(data_path)
    df = df[df["SecurityID"] == stock]
    df = df.sort_values(["SecurityID", "TransactDT_MS", "ChannelNo", "ApplSeqNum"], kind="mergesort")
    tok_max = int(df["order_token"].max()) if len(df) else -1
    if tok_max >= args.vocab_size:
        raise ValueError(f"order_token max={tok_max} >= vocab_size={args.vocab_size}")

    train_loader, val_loader, test_loader, split_stats = make_day_loaders_and_datasets(
        df=df,
        window_len=args.window_len,
        batch_size=args.batch_size,
        shuffle_seed=args.seed + 123,
    )

    if args.variant == "dyn_gatebias":
        model = DynamicAnchor_GateBias(
            vocab_size=args.vocab_size,
            anchor_count=args.anchor_count,
            gate_dimwise=bool(args.gate_dimwise),
        ).to(device)
    else:
        model = DynamicAnchor_AttnPoolTopK(
            vocab_size=args.vocab_size,
            anchor_count=args.anchor_count,
            topk_anchors=int(args.topk_anchors),
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
        "seed": int(args.seed),
        "stock": stock,
        "train_day_in_filename": args.day.strip() or None,
        "data_path": data_path,
        "data_dir": args.data_dir,
        "vocab_size": int(args.vocab_size),
        "anchor_count": int(args.anchor_count),
        "window_len": int(args.window_len),
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "amp": bool(use_amp),
        "lr": float(args.lr),
        "epochs_max": int(args.epochs),
        "early_stop_patience": int(args.patience),
        "backbone_init": "random_GPT2Config",
        "variant": args.variant,
        "variant_params": {
            "gate_dimwise": bool(args.gate_dimwise),
            "topk_anchors": int(args.topk_anchors) if args.variant != "dyn_gatebias" else None,
        },
        "objective": {
            "reg_mode": args.reg_mode,
            "reg_warmup_steps": int(args.reg_warmup_steps),
            "margin": {"lambda": float(args.reg_lambda), "margin_m": float(args.margin_m)},
            "entropy": {"lambda": float(args.ent_lambda), "warmup_steps": int(args.ent_warmup_steps)},
            "load_balance": {"lambda": float(args.lb_lambda), "warmup_steps": int(args.lb_warmup_steps)},
        },
        "split": split_stats,
        "epochs": [],
        "output": {"base": out_base, "model_cache": model_cache_dir, "plots": plot_dir},
    }

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_state = None
    global_step = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_ce_sum = 0.0
        epoch_reg_sum = 0.0
        epoch_tot_sum = 0.0
        epoch_batches = 0
        router_diag = RouterDiag()

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
                logits, probs, scores, extra = model(x)
                ce_loss = criterion(logits, y)

                reg_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
                lam = 0.0

                if args.reg_mode == "margin_gated":
                    top2 = torch.topk(scores, k=2, dim=1).values
                    margin = top2[:, 0] - top2[:, 1]
                    per = torch.relu(float(args.margin_m) - margin)  # [B]
                    # gated reg: only enforce when anchor is actually used (variant A)
                    if args.variant == "dyn_gatebias":
                        per = per * extra.detach()
                    reg_loss = per.mean()
                    lam = _weight_after_warmup(global_step, int(args.reg_warmup_steps), float(args.reg_lambda))

                elif args.reg_mode == "entropy_lb":
                    ent = _entropy_from_probs(probs).mean()
                    pbar_probs = probs.mean(dim=0).clamp_min(1e-12)
                    uniform = torch.full_like(pbar_probs, 1.0 / float(pbar_probs.shape[0]))
                    kl = (pbar_probs * (pbar_probs / uniform).log()).sum()
                    ent_w = _weight_after_warmup(global_step, int(args.ent_warmup_steps), float(args.ent_lambda))
                    lb_w = _weight_after_warmup(global_step, int(args.lb_warmup_steps), float(args.lb_lambda))
                    # Minimize (-entropy) to maximize entropy; plus KL to uniform for load-balance.
                    reg_loss = (-ent) * ent_w + kl * lb_w
                    lam = 1.0  # already embedded in ent_w/lb_w
                else:
                    raise ValueError(f"Unknown reg_mode {args.reg_mode}")

                total_loss = ce_loss + lam * reg_loss
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

            # diagnostics aggregation
            if args.variant == "dyn_gatebias":
                _router_diag_update(router_diag, probs, gate=extra, pool_attn=None)
            else:
                _router_diag_update(router_diag, probs, gate=None, pool_attn=extra)

            epoch_ce_sum += float(ce_loss.item())
            epoch_reg_sum += float(reg_loss.item()) if torch.is_tensor(reg_loss) else float(reg_loss)
            epoch_tot_sum += float((ce_loss + lam * reg_loss).item()) if torch.is_tensor(reg_loss) else float(ce_loss.item())
            epoch_batches += 1

            if (epoch_batches % PRINT_EVERY_STEPS) == 0:
                pbar.set_postfix(ce=f"{ce_loss.item():.3f}", reg=f"{float(reg_loss.item()):.3f}", lam=f"{lam:.0e}")

        if (epoch_batches % max(int(args.grad_accum_steps), 1)) != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_ce = epoch_ce_sum / max(epoch_batches, 1)
        train_reg = epoch_reg_sum / max(epoch_batches, 1)
        val_ce = eval_ce(model, val_loader, device, criterion, args.variant)
        test_ce = eval_ce(model, test_loader, device, criterion, args.variant)

        router_stats = _router_diag_finalize(router_diag, anchor_count=int(args.anchor_count))
        print(
            f"\n[Epoch {epoch}] train_ce={train_ce:.4f} train_reg={train_reg:.4f} | "
            f"val_ce={val_ce:.4f} test_ce={test_ce:.4f}"
        )
        print(f"[RouterDiag] {json.dumps(router_stats)}")

        run_meta["epochs"].append(
            {
                "epoch": int(epoch),
                "train_ce": float(train_ce),
                "train_reg": float(train_reg),
                "val_ce": float(val_ce),
                "test_ce": float(test_ce),
                "router_diag": router_stats,
            }
        )

        if (best_val - val_ce) > MIN_DELTA:
            best_val = val_ce
            best_epoch = int(epoch)
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "exp_name": exp_name,
                    "timestamp": timestamp_run,
                    "seed": int(args.seed),
                    "stock": stock,
                    "best_epoch": best_epoch,
                    "best_val_ce": float(best_val),
                    "vocab_size": int(args.vocab_size),
                    "anchor_count": int(args.anchor_count),
                    "model_state_dict": best_state,
                    "config": run_meta,
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

    final_test_ce = eval_ce(model, test_loader, device, criterion, args.variant)
    print("\n==================== FINAL ====================")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val CE: {best_val:.6f}")
    print(f"Test CE (best weights): {final_test_ce:.6f}")
    print(f"Checkpoint: {global_best_ckpt_path}")
    print("================================================\n")

    run_meta["final_test_ce_best_weights"] = float(final_test_ce)
    with open(global_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    @torch.no_grad()
    def diagnostics_sampling_plot():
        model.eval()
        all_preds = []
        for x, _ in tqdm(test_loader, desc="Test sampling", unit="batch"):
            x = x.to(device, non_blocking=True)
            logits, _, _, _ = model(x)
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
    print("Run complete.")


if __name__ == "__main__":
    main()

