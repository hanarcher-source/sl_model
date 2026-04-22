#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blank/random-init GPT2 + no-anchor head, single processed day (open-bid-anchor txn-complete flow).

Reads one per-stock joblib (e.g. 20250709_openbidanchor_txncomplete), 60/20/20 split, trains up to --epochs.
"""

import argparse
import glob
import json
import os
import random
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
from contextlib import nullcontext

# Defaults match preprocess_real_lob_*_openbidanchor_txncomplete.py (n_side=5, no split cancel)
DEFAULT_VOCAB_SIZE = 26 * 26 * 12 * 5  # 40560
DEFAULT_DATA_DIR = (
    "/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/processed_real_flow/"
    "20250709_openbidanchor_txncomplete"
)
DEFAULT_OUTPUT_ROOT = (
    "/finance_ML/zhanghaohan/stock_language_model/training_runs/"
    "20250709_openbidanchor_txncomplete_blank_gpt2"
)

SEED = 42
WINDOW_LEN = 50
BATCH_SIZE = 256
LR = 1e-4
DEFAULT_PATIENCE = 3
MIN_DELTA = 1e-4
PRINT_EVERY_STEPS = 50


def make_dataloader_generator(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def find_stock_joblib(data_dir: str, stock: str, day: str = "") -> str:
    """If day is set (YYYYMMDD), match that calendar slice only (needed when data-dir holds 0709+0710)."""
    stock_tag = stock.replace("_", "")
    day = str(day).strip()
    if day:
        pat = os.path.join(
            data_dir,
            f"final_result_for_merge_realflow_openbidanchor_txncomplete_{day}_{stock_tag}_*.joblib",
        )
    else:
        pat = os.path.join(
            data_dir,
            f"final_result_for_merge_realflow_openbidanchor_txncomplete_*_{stock_tag}_*.joblib",
        )
    matches = sorted(glob.glob(pat))
    if not matches:
        raise FileNotFoundError(f"No joblib for stock={stock} under {data_dir} (glob {pat!r})")
    return matches[-1]


class ConcatWindowDataset(Dataset):
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


def build_per_stock_splits(
    df,
    window,
    stock_col="SecurityID",
    time_col="TransactDT_MS",
    channel_col="ChannelNo",
    seq_col="ApplSeqNum",
    token_col="order_token",
):
    train_samples, val_samples, test_samples = [], [], []
    stats = {"per_stock": {}, "total": {}}

    for sid, g in df.groupby(stock_col, sort=False):
        g = g.sort_values([time_col, channel_col, seq_col], kind="mergesort")

        seq = torch.as_tensor(g[token_col].values.astype(np.int64), dtype=torch.long)
        n = max(0, seq.numel() - int(window))

        if n <= 0:
            stats["per_stock"][sid] = {
                "seq_len": int(seq.numel()),
                "windows": 0,
                "train": 0,
                "val": 0,
                "test": 0,
            }
            continue

        n_train = int(0.6 * n)
        n_val = int(0.2 * n)
        n_test = n - n_train - n_val

        train_idx = list(range(0, n_train))
        val_idx = list(range(n_train, n_train + n_val))
        test_idx = list(range(n_train + n_val, n))

        train_samples.extend([(seq, i) for i in train_idx])
        val_samples.extend([(seq, i) for i in val_idx])
        test_samples.extend([(seq, i) for i in test_idx])

        stats["per_stock"][sid] = {
            "seq_len": int(seq.numel()),
            "windows": int(n),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        }

    stats["total"] = {
        "train": len(train_samples),
        "val": len(val_samples),
        "test": len(test_samples),
        "all": len(train_samples) + len(val_samples) + len(test_samples),
    }
    return train_samples, val_samples, test_samples, stats


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

    train_ds = ConcatWindowDataset(train_samples, window_len)
    val_ds = ConcatWindowDataset(val_samples, window_len)
    test_ds = ConcatWindowDataset(test_samples, window_len)

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


class OrderGPT2_NoAnchor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        init_pretrained_backbone: bool = False,
        pretrained_backbone_name: str = "gpt2",
        pretrained_local_only: bool = False,
    ):
        super().__init__()
        if init_pretrained_backbone:
            self.gpt2 = GPT2Model.from_pretrained(
                pretrained_backbone_name,
                local_files_only=bool(pretrained_local_only),
            )
        else:
            # IMPORTANT: avoid any network calls on clusters without outbound SSL.
            # GPT2Config() matches the standard GPT-2 "small" architecture defaults.
            self.gpt2 = GPT2Model(GPT2Config())
        hidden_size = self.gpt2.config.hidden_size
        self.order_embedding = nn.Embedding(int(vocab_size), hidden_size)
        self.head = nn.Linear(hidden_size, int(vocab_size))

    def forward(self, x):
        emb = self.order_embedding(x)
        outputs = self.gpt2(inputs_embeds=emb)
        h = outputs.last_hidden_state
        logits = self.head(h[:, -1, :])
        return logits


@torch.no_grad()
def eval_ce(model, loader, device, criterion):
    model.eval()
    total = 0.0
    cnt = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        ce = criterion(logits, y)
        total += ce.item()
        cnt += 1
    return total / max(cnt, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stock", required=True, help="e.g. 000617_XSHE")
    p.add_argument(
        "--day",
        default="",
        help="Trade date YYYYMMDD embedded in preprocess filename (use when data-dir contains multiple days).",
    )
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early stopping patience on val CE (no improvement for this many epochs).",
    )
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--window-len", type=int, default=WINDOW_LEN)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate gradients for this many steps before optimizer.step().",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision (autocast + GradScaler) when running on CUDA.",
    )
    p.add_argument(
        "--grad-checkpointing",
        action="store_true",
        help="Enable GPT-2 gradient checkpointing to reduce activation memory (slower).",
    )
    p.add_argument(
        "--init-pretrained-backbone",
        action="store_true",
        help="Initialize GPT2Model backbone from pretrained weights (keeps custom token embedding/head random).",
    )
    p.add_argument(
        "--pretrained-backbone-name",
        default="gpt2",
        help="HF model name/path to load for pretrained backbone (default: gpt2).",
    )
    p.add_argument(
        "--pretrained-local-only",
        action="store_true",
        help="Pass local_files_only=True to from_pretrained (avoid network).",
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
    exp_name = f"blankGPT2_{day_tag}_txncomplete_{stock_tag}_win{args.window_len}"
    timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_best_ckpt_path = os.path.join(model_cache_dir, f"{exp_name}_{timestamp_run}_best.pt")
    global_meta_path = os.path.join(model_cache_dir, f"{exp_name}_{timestamp_run}_meta.json")

    print(f"[device] {device} | stock={stock}")
    print(f"[data] {data_path}")
    print(f"[vocab_size] {args.vocab_size}")
    print(f"[out] {out_base}")

    df = joblib.load(data_path)
    df = df[df["SecurityID"] == stock]
    df = df.sort_values(
        ["SecurityID", "TransactDT_MS", "ChannelNo", "ApplSeqNum"],
        kind="mergesort",
    )
    print(f"[rows] {len(df)}")

    tok_max = int(df["order_token"].max()) if len(df) else -1
    if tok_max >= args.vocab_size:
        raise ValueError(
            f"order_token max={tok_max} >= vocab_size={args.vocab_size}; increase --vocab-size"
        )

    train_loader, val_loader, test_loader, split_stats = make_day_loaders_and_datasets(
        df=df,
        window_len=args.window_len,
        batch_size=args.batch_size,
        shuffle_seed=args.seed + 123,
    )

    for sid, st in split_stats["per_stock"].items():
        print(
            f"  {sid}: seq_len={st['seq_len']} windows={st['windows']} "
            f"train={st['train']} val={st['val']} test={st['test']}"
        )
    print("TOTAL:", split_stats["total"])

    model = OrderGPT2_NoAnchor(
        vocab_size=args.vocab_size,
        init_pretrained_backbone=bool(args.init_pretrained_backbone),
        pretrained_backbone_name=str(args.pretrained_backbone_name),
        pretrained_local_only=bool(args.pretrained_local_only),
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
        "order_token_max_observed": tok_max,
        "window_len": args.window_len,
        "batch_size": args.batch_size,
        "grad_accum_steps": int(args.grad_accum_steps),
        "amp": bool(use_amp),
        "grad_checkpointing": bool(args.grad_checkpointing),
        "lr": args.lr,
        "epochs_max": args.epochs,
        "early_stop_patience": int(args.patience),
        "backbone_init": (
            f"pretrained:{args.pretrained_backbone_name}"
            if args.init_pretrained_backbone
            else f"random_config:{args.pretrained_backbone_name}"
        ),
        "ablation": {"mode": "no_anchor"},
        "split": split_stats,
        "epochs": [],
        "output": {
            "base": out_base,
            "model_cache": model_cache_dir,
            "plots": plot_dir,
        },
    }

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_ce_sum = 0.0
        epoch_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        optimizer.zero_grad(set_to_none=True)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp) if device.type == "cuda" else nullcontext()
            with autocast_ctx:
                logits = model(x)
                ce_loss = criterion(logits, y)
                ce_loss = ce_loss / max(int(args.grad_accum_steps), 1)
            if use_amp:
                scaler.scale(ce_loss).backward()
            else:
                ce_loss.backward()

            if (epoch_batches + 1) % max(int(args.grad_accum_steps), 1) == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_ce_sum += ce_loss.item()
            epoch_batches += 1
            if (epoch_batches % PRINT_EVERY_STEPS) == 0:
                pbar.set_postfix(ce=f"{ce_loss.item():.3f}")

        # flush any remaining grads if batches not divisible by accum
        if (epoch_batches % max(int(args.grad_accum_steps), 1)) != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_ce = epoch_ce_sum / max(epoch_batches, 1)
        val_ce = eval_ce(model, val_loader, device, criterion)
        test_ce = eval_ce(model, test_loader, device, criterion)
        print(
            f"\n[Epoch {epoch}] train_ce={train_ce:.4f} | val_ce={val_ce:.4f} test_ce={test_ce:.4f}"
        )

        run_meta["epochs"].append(
            {
                "epoch": epoch,
                "train_ce": float(train_ce),
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
            logits = model(x)
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
