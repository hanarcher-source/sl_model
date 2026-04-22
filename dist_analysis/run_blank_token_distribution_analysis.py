#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2Config, GPT2Model


PROJECT_ROOT = "/finance_ML/zhanghaohan/stock_language_model"
SOURCE_ROOT = "/finance_ML/zhanghaohan/GPT2_new_head_multi_d"
MODEL_CACHE_DIR = os.path.join(SOURCE_ROOT, "model_cache")
ANALYSIS_ROOT = os.path.join(PROJECT_ROOT, "dist_analysis")

VOCAB_SIZE = 24336
DEFAULT_BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_CONFIGS = [
    {
        "stock": "000617_XSHE",
        "meta_json": os.path.join(
            MODEL_CACHE_DIR,
            "blankGPT2_multiday_continue_617_stock_win50_20260325_202035.json",
        ),
        "checkpoint": os.path.join(
            MODEL_CACHE_DIR,
            "blankGPT2_multiday_continue_617_stock_win50_20260325_202035.pt",
        ),
    },
    {
        "stock": "000981_XSHE",
        "meta_json": os.path.join(
            MODEL_CACHE_DIR,
            "blankGPT2_multiday_continue_981_stock_win50_20260325_202032.json",
        ),
        "checkpoint": os.path.join(
            MODEL_CACHE_DIR,
            "blankGPT2_multiday_continue_981_stock_win50_20260325_202032.pt",
        ),
    },
    {
        "stock": "002263_XSHE",
        "meta_json": os.path.join(
            MODEL_CACHE_DIR,
            "blankGPT2_multiday_continue_2263_stock_win50_20260325_202031.json",
        ),
        "checkpoint": os.path.join(
            MODEL_CACHE_DIR,
            "blankGPT2_multiday_continue_2263_stock_win50_20260325_202031.pt",
        ),
    },
    {
        "stock": "002366_XSHE",
        "meta_json": os.path.join(
            MODEL_CACHE_DIR,
            "blankGPT2_multiday_continue_2366_stock_win50_20260325_202032.json",
        ),
        "checkpoint": os.path.join(
            MODEL_CACHE_DIR,
            "blankGPT2_multiday_continue_2366_stock_win50_20260325_202032.pt",
        ),
    },
]


DEFAULT_SEED = 42


def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_set_global_seed(DEFAULT_SEED)


class ConcatWindowDataset(Dataset):
    def __init__(self, samples, window):
        self.samples = samples
        self.window = int(window)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, start_idx = self.samples[idx]
        x = seq[start_idx:start_idx + self.window]
        y = seq[start_idx + self.window]
        return x, y


class OrderGPT2NoAnchor(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        gpt2_config = GPT2Config.from_pretrained("gpt2")
        self.gpt2 = GPT2Model(gpt2_config)
        hidden_size = self.gpt2.config.hidden_size
        self.order_embedding = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb = self.order_embedding(x)
        outputs = self.gpt2(inputs_embeds=emb)
        h = outputs.last_hidden_state
        return self.head(h[:, -1, :])


@dataclass
class DistributionArtifacts:
    stock: str
    experiment_name: str
    checkpoint_path: str
    metadata_path: str
    predicted_plot: str
    true_plot: str
    predicted_counts_csv: str
    true_counts_csv: str
    predicted_samples_csv: str
    true_samples_csv: str
    summary_json: str
    sample_count: int


def _safe_stock_tag(stock: str) -> str:
    return stock.replace("_", "")


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_checkpoint_state(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(f"Checkpoint missing model_state_dict: {path}")
    return state


def _build_test_dataset_for_day(df: pd.DataFrame, stock: str, window_len: int):
    df = df[df["SecurityID"] == stock].copy()
    df = df.sort_values(["SecurityID", "TransactDT_MS", "ChannelNo", "ApplSeqNum"], kind="mergesort")

    seq = torch.as_tensor(df["order_token"].values.astype(np.int64), dtype=torch.long)
    n = max(0, seq.numel() - int(window_len))
    if n <= 0:
        return ConcatWindowDataset([], window_len), {
            "seq_len": int(seq.numel()),
            "windows": 0,
            "test": 0,
        }

    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    n_test = n - n_train - n_val
    test_idx = list(range(n_train + n_val, n))
    samples = [(seq, idx) for idx in test_idx]
    ds = ConcatWindowDataset(samples, window_len)
    return ds, {
        "seq_len": int(seq.numel()),
        "windows": int(n),
        "test": int(n_test),
    }


def _collect_true_targets(loader: DataLoader) -> np.ndarray:
    all_targets = []
    for _, y in tqdm(loader, desc="Collecting true targets", unit="batch", leave=False):
        all_targets.append(y.cpu())
    if not all_targets:
        return np.array([], dtype=np.int64)
    return torch.cat(all_targets).numpy().astype(np.int64)


@torch.no_grad()
def _sample_predictions(model: nn.Module, loader: DataLoader, sample_seed: int) -> np.ndarray:
    model.eval()
    sample_gen = torch.Generator(device="cpu")
    sample_gen.manual_seed(sample_seed)

    all_preds = []
    for x, _ in tqdm(loader, desc="Sampling predictions", unit="batch", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.multinomial(probs.cpu(), 1, generator=sample_gen).squeeze(1)
        all_preds.append(preds)

    if not all_preds:
        return np.array([], dtype=np.int64)
    return torch.cat(all_preds).numpy().astype(np.int64)


def _distribution_table(tokens: np.ndarray, vocab_size: int) -> pd.DataFrame:
    if tokens.size == 0:
        return pd.DataFrame(columns=["token_id", "count", "probability"])

    counts = np.bincount(tokens, minlength=vocab_size)
    nonzero = np.nonzero(counts)[0]
    total = counts.sum()
    return pd.DataFrame(
        {
            "token_id": nonzero.astype(np.int64),
            "count": counts[nonzero].astype(np.int64),
            "probability": counts[nonzero].astype(np.float64) / float(total),
        }
    ).sort_values("token_id").reset_index(drop=True)


def _raw_samples_table(tokens: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"token_id": tokens.astype(np.int64)})


def _shared_histogram_settings(tokens_a: np.ndarray, tokens_b: np.ndarray, bins: int = 200):
    combined = np.concatenate([tokens_a, tokens_b])
    if combined.size == 0:
        edges = np.linspace(0.0, 1.0, bins + 1)
        ymax = 1.0
        return edges, ymax

    lo = float(np.min(combined))
    hi = float(np.max(combined))
    if lo == hi:
        lo -= 0.5
        hi += 0.5

    edges = np.linspace(lo, hi, bins + 1)
    counts_a, _ = np.histogram(tokens_a, bins=edges)
    counts_b, _ = np.histogram(tokens_b, bins=edges)
    ymax = float(max(int(counts_a.max(initial=0)), int(counts_b.max(initial=0)), 1)) * 1.05
    return edges, ymax


def _plot_histogram(tokens: np.ndarray, title: str, out_path: str, bin_edges: np.ndarray, ymax: float):
    plt.figure(figsize=(10, 6))
    plt.hist(tokens, bins=bin_edges)
    plt.title(title)
    plt.xlabel("Token ID")
    plt.ylabel("Frequency")
    plt.ylim(0, ymax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _prepare_loader(meta: dict, stock: str, batch_size: int):
    window_len = int(meta["window_len"])
    day_entries = meta.get("per_day", {})
    datasets = []
    day_summaries = []

    for day in meta.get("day_list", []):
        day_meta = day_entries.get(day)
        if day_meta is None:
            continue
        data_path = day_meta.get("data_path")
        if not data_path or not os.path.isfile(data_path):
            raise FileNotFoundError(f"Missing day data for {stock}: {data_path}")

        df = joblib.load(data_path)
        test_ds, observed = _build_test_dataset_for_day(df, stock=stock, window_len=window_len)
        expected = ((day_meta.get("split") or {}).get("per_stock") or {}).get(stock, {})
        expected_test = int(expected.get("test", len(test_ds)))
        if len(test_ds) != expected_test:
            raise RuntimeError(
                f"Test split mismatch for {stock} day={day}: observed={len(test_ds)} expected={expected_test}"
            )
        datasets.append(test_ds)
        day_summaries.append(
            {
                "day": day,
                "data_path": data_path,
                "observed": observed,
                "expected_test_windows": expected_test,
            }
        )

    if not datasets:
        raise RuntimeError(f"No test datasets reconstructed for {stock}")

    all_test_ds = ConcatDataset(datasets)
    loader = DataLoader(
        all_test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return loader, day_summaries


def run_one(config: dict, run_dir: str) -> DistributionArtifacts:
    stock = config["stock"]
    meta_path = config["meta_json"]
    ckpt_path = config["checkpoint"]

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing metadata json: {meta_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    meta = _load_json(meta_path)
    exp_name = str(meta["exp_name"])
    batch_size = int(meta.get("batch_size", DEFAULT_BATCH_SIZE))
    seed = int(meta.get("seed", DEFAULT_SEED))
    sample_seed = seed + 999

    _set_global_seed(seed)

    stock_dir = os.path.join(run_dir, _safe_stock_tag(stock))
    os.makedirs(stock_dir, exist_ok=True)

    loader, day_summaries = _prepare_loader(meta=meta, stock=stock, batch_size=batch_size)

    model = OrderGPT2NoAnchor(vocab_size=VOCAB_SIZE)
    model.load_state_dict(_load_checkpoint_state(ckpt_path), strict=True)
    model.to(DEVICE).eval()

    predicted_tokens = _sample_predictions(model, loader, sample_seed=sample_seed)
    true_tokens = _collect_true_targets(loader)

    predicted_counts = _distribution_table(predicted_tokens, vocab_size=VOCAB_SIZE)
    true_counts = _distribution_table(true_tokens, vocab_size=VOCAB_SIZE)
    predicted_samples = _raw_samples_table(predicted_tokens)
    true_samples = _raw_samples_table(true_tokens)

    predicted_plot = os.path.join(stock_dir, f"{exp_name}_predicted_distribution.png")
    true_plot = os.path.join(stock_dir, f"{exp_name}_true_distribution.png")
    predicted_counts_csv = os.path.join(stock_dir, f"{exp_name}_predicted_counts.csv")
    true_counts_csv = os.path.join(stock_dir, f"{exp_name}_true_counts.csv")
    predicted_samples_csv = os.path.join(stock_dir, f"{exp_name}_predicted_samples.csv")
    true_samples_csv = os.path.join(stock_dir, f"{exp_name}_true_samples.csv")
    summary_json = os.path.join(stock_dir, f"{exp_name}_distribution_summary.json")

    bin_edges, ymax = _shared_histogram_settings(predicted_tokens, true_tokens)

    _plot_histogram(
        predicted_tokens,
        title=f"{exp_name} - out-of-sample predicted token distribution",
        out_path=predicted_plot,
        bin_edges=bin_edges,
        ymax=ymax,
    )
    _plot_histogram(
        true_tokens,
        title=f"{exp_name} - out-of-sample true token distribution",
        out_path=true_plot,
        bin_edges=bin_edges,
        ymax=ymax,
    )

    predicted_counts.to_csv(predicted_counts_csv, index=False)
    true_counts.to_csv(true_counts_csv, index=False)
    predicted_samples.to_csv(predicted_samples_csv, index=False)
    true_samples.to_csv(true_samples_csv, index=False)

    summary = {
        "stock": stock,
        "experiment_name": exp_name,
        "checkpoint_path": ckpt_path,
        "metadata_path": meta_path,
        "device": str(DEVICE),
        "seed": seed,
        "sample_seed": sample_seed,
        "window_len": int(meta["window_len"]),
        "batch_size": batch_size,
        "num_samples": int(predicted_tokens.size),
        "unique_predicted_tokens": int(predicted_counts.shape[0]),
        "unique_true_tokens": int(true_counts.shape[0]),
        "outputs": {
            "predicted_plot": predicted_plot,
            "true_plot": true_plot,
            "predicted_counts_csv": predicted_counts_csv,
            "true_counts_csv": true_counts_csv,
            "predicted_samples_csv": predicted_samples_csv,
            "true_samples_csv": true_samples_csv,
        },
        "days": day_summaries,
    }
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return DistributionArtifacts(
        stock=stock,
        experiment_name=exp_name,
        checkpoint_path=ckpt_path,
        metadata_path=meta_path,
        predicted_plot=predicted_plot,
        true_plot=true_plot,
        predicted_counts_csv=predicted_counts_csv,
        true_counts_csv=true_counts_csv,
        predicted_samples_csv=predicted_samples_csv,
        true_samples_csv=true_samples_csv,
        summary_json=summary_json,
        sample_count=int(predicted_tokens.size),
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze out-of-sample token distributions for blank GPT2 checkpoints.")
    parser.add_argument(
        "--stock",
        action="append",
        dest="stocks",
        default=None,
        help="Optional stock filter. Can be passed multiple times, e.g. --stock 000617_XSHE --stock 002263_XSHE.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional existing output directory to reuse so multiple jobs can write into the same run folder.",
    )
    args = parser.parse_args()

    selected_configs = CHECKPOINT_CONFIGS
    if args.stocks:
        wanted = set(args.stocks)
        selected_configs = [cfg for cfg in CHECKPOINT_CONFIGS if cfg["stock"] in wanted]
        if not selected_configs:
            raise ValueError(f"No configured checkpoints matched requested stocks: {sorted(wanted)}")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join(ANALYSIS_ROOT, f"blank_token_dist_run_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)

    manifest = {
        "created_at": run_ts,
        "project_root": PROJECT_ROOT,
        "source_root": SOURCE_ROOT,
        "analysis_root": run_dir,
        "device": str(DEVICE),
        "default_seed": DEFAULT_SEED,
        "stocks": [],
    }

    for config in selected_configs:
        stock = config["stock"]
        print("=" * 88)
        print(f"Running token distribution analysis for {stock}")
        print("=" * 88)
        artifacts = run_one(config=config, run_dir=run_dir)
        manifest["stocks"].append({
            "stock": artifacts.stock,
            "experiment_name": artifacts.experiment_name,
            "checkpoint_path": artifacts.checkpoint_path,
            "metadata_path": artifacts.metadata_path,
            "sample_count": artifacts.sample_count,
            "predicted_plot": artifacts.predicted_plot,
            "true_plot": artifacts.true_plot,
            "predicted_counts_csv": artifacts.predicted_counts_csv,
            "true_counts_csv": artifacts.true_counts_csv,
            "predicted_samples_csv": artifacts.predicted_samples_csv,
            "true_samples_csv": artifacts.true_samples_csv,
            "summary_json": artifacts.summary_json,
        })
        print(f"Saved analysis for {stock} under {os.path.dirname(artifacts.summary_json)}")

    manifest_name = "manifest.json"
    if args.stocks and len(selected_configs) == 1:
        manifest_name = f"manifest_{_safe_stock_tag(selected_configs[0]['stock'])}.json"
    manifest_path = os.path.join(run_dir, manifest_name)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("=" * 88)
    print("Analysis complete")
    print(f"manifest={manifest_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()