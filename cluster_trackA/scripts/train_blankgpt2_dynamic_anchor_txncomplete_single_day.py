#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blank GPT-2 + learned dynamic anchor mixture (V5-style) on one pooled preprocess day.

Matches fivedays_train_dynamic_anchor_1st_stock_V5_blank.py mechanism on a *single* day/stock:
  - query = mean(order embeddings over window)
  - softmax scores over ANCHOR_COUNT learnable anchor vectors
  - prepend weighted anchor as extra token, then GPT-2 on [anchor, emb...]
  - CE + margin regularization on top-2 score gap (warmup on reg)

Data pathing matches train_blankgpt2_openbidanchor_txncomplete_single_day.py (openbidanchor_txncomplete joblibs).
"""

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from datetime import datetime

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

from train_blankgpt2_openbidanchor_txncomplete_single_day import (
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
    "pool_0709_0710_train0709_blank_gpt2_dynamic_anchor_win50"
)


class ConcatWindowDatasetLocal(Dataset):
    """Same as imported ConcatWindowDataset (explicit for clarity)."""

    def __init__(self, samples, window):
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
        return logits, anchor_probs, scores


@torch.no_grad()
def eval_ce(model, loader, device, criterion):
    model.eval()
    total = 0.0
    cnt = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _, _ = model(x)
        ce = criterion(logits, y)
        total += ce.item()
        cnt += 1
    return total / max(cnt, 1)


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
    p.add_argument("--reg-lambda", type=float, default=1e-2)
    p.add_argument("--margin-m", type=float, default=1.0)
    p.add_argument("--reg-warmup-steps", type=int, default=20)
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
    out_base = os.path.join(args.output_root, stock_tag)
    model_cache_dir = os.path.join(out_base, "model_cache")
    plot_dir = os.path.join(out_base, "plots")
    os.makedirs(model_cache_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    day_tag = args.day.strip() if args.day.strip() else "anyday"
    exp_name = f"blankGPT2dyn_{day_tag}_txncomplete_{stock_tag}_win{args.window_len}"
    timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_best_ckpt_path = os.path.join(model_cache_dir, f"{exp_name}_{timestamp_run}_best.pt")
    global_meta_path = os.path.join(model_cache_dir, f"{exp_name}_{timestamp_run}_meta.json")

    print(f"[device] {device} | stock={stock} | dynamic_anchor")
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

    train_loader, val_loader, test_loader, split_stats = make_day_loaders_and_datasets(
        df=df,
        window_len=args.window_len,
        batch_size=args.batch_size,
        shuffle_seed=args.seed + 123,
    )

    model = OrderGPT2_DynamicAnchor(
        vocab_size=args.vocab_size,
        anchor_count=args.anchor_count,
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
        "anchor_count": args.anchor_count,
        "reg": {
            "lambda": args.reg_lambda,
            "margin_m": args.margin_m,
            "warmup_steps": args.reg_warmup_steps,
        },
        "window_len": args.window_len,
        "batch_size": args.batch_size,
        "grad_accum_steps": int(args.grad_accum_steps),
        "amp": use_amp,
        "lr": args.lr,
        "epochs_max": args.epochs,
        "early_stop_patience": int(args.patience),
        "backbone_init": "random_GPT2Config",
        "ablation": {"mode": "dynamic_anchor_V5_style"},
        "split": split_stats,
        "epochs": [],
        "output": {"base": out_base, "model_cache": model_cache_dir, "plots": plot_dir},
    }

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_state = None
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_ce_sum = 0.0
        epoch_reg_sum = 0.0
        epoch_tot_sum = 0.0
        epoch_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        optimizer.zero_grad(set_to_none=True)

        for x, y in pbar:
            global_step += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp) if device.type == "cuda" else nullcontext()
            with autocast_ctx:
                logits, _, scores = model(x)
                ce_loss = criterion(logits, y)
                top2 = torch.topk(scores, k=2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]
                reg_loss = torch.relu(float(args.margin_m) - margin).mean()
                lam = 0.0 if global_step <= int(args.reg_warmup_steps) else float(args.reg_lambda)
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

            epoch_ce_sum += ce_loss.item()
            epoch_reg_sum += reg_loss.item()
            epoch_tot_sum += (ce_loss + lam * reg_loss).item()
            epoch_batches += 1
            if (epoch_batches % PRINT_EVERY_STEPS) == 0:
                pbar.set_postfix(ce=f"{ce_loss.item():.3f}", reg=f"{reg_loss.item():.3f}", lam=f"{lam:.0e}")

        if (epoch_batches % max(int(args.grad_accum_steps), 1)) != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_ce = epoch_ce_sum / max(epoch_batches, 1)
        train_reg = epoch_reg_sum / max(epoch_batches, 1)
        val_ce = eval_ce(model, val_loader, device, criterion)
        test_ce = eval_ce(model, test_loader, device, criterion)
        print(
            f"\n[Epoch {epoch}] train_ce={train_ce:.4f} train_reg={train_reg:.4f} | "
            f"val_ce={val_ce:.4f} test_ce={test_ce:.4f}"
        )

        run_meta["epochs"].append(
            {
                "epoch": epoch,
                "train_ce": float(train_ce),
                "train_reg": float(train_reg),
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
                    "anchor_count": args.anchor_count,
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
    print("Run complete.")


if __name__ == "__main__":
    main()
