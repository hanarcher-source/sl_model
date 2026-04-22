#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def _find_one(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched pattern: {pattern}")
    return matches[-1]


def _load_prob_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {"token_id", "probability"}.issubset(df.columns):
        raise ValueError(f"Expected token_id/probability columns in {csv_path}")
    return df[["token_id", "probability"]].copy()


def _load_token_samples(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if "token_id" not in df.columns:
        raise ValueError(f"Expected token_id column in {csv_path}")
    return df["token_id"].to_numpy(dtype=np.int64)


def _align_probabilities(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    merged = true_df.rename(columns={"probability": "p_true"}).merge(
        pred_df.rename(columns={"probability": "p_pred"}),
        on="token_id",
        how="outer",
    ).fillna(0.0)
    return merged.sort_values("token_id").reset_index(drop=True)


def _l1_by_group_discrete(aligned: pd.DataFrame) -> float:
    return float(0.5 * np.abs(aligned["p_true"] - aligned["p_pred"]).sum())


def _js_divergence_bits(aligned: pd.DataFrame) -> float:
    p = aligned["p_true"].to_numpy(dtype=float)
    q = aligned["p_pred"].to_numpy(dtype=float)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def _wasserstein_zscore(true_samples: np.ndarray, pred_samples: np.ndarray) -> float:
    combined = np.concatenate([true_samples.astype(float), pred_samples.astype(float)])
    mean = float(np.mean(combined))
    std = float(np.std(combined))
    if std == 0.0:
        return 0.0
    true_z = (true_samples.astype(float) - mean) / std
    pred_z = (pred_samples.astype(float) - mean) / std
    return float(wasserstein_distance(true_z, pred_z))


def _summarize_stock(stock_dir: str) -> dict:
    summary_json = _find_one(os.path.join(stock_dir, "*_distribution_summary.json"))
    summary = json.load(open(summary_json, "r", encoding="utf-8"))

    true_counts_csv = summary["outputs"]["true_counts_csv"]
    pred_counts_csv = summary["outputs"]["predicted_counts_csv"]
    true_samples_csv = summary["outputs"]["true_samples_csv"]
    pred_samples_csv = summary["outputs"]["predicted_samples_csv"]

    true_probs = _load_prob_table(true_counts_csv)
    pred_probs = _load_prob_table(pred_counts_csv)
    true_samples = _load_token_samples(true_samples_csv)
    pred_samples = _load_token_samples(pred_samples_csv)
    aligned = _align_probabilities(true_probs, pred_probs)

    overlap_true = set(true_probs["token_id"].tolist())
    overlap_pred = set(pred_probs["token_id"].tolist())
    overlap = len(overlap_true & overlap_pred)
    union = len(overlap_true | overlap_pred)

    return {
        "stock": summary["stock"],
        "experiment_name": summary["experiment_name"],
        "sample_count_true": int(true_samples.size),
        "sample_count_pred": int(pred_samples.size),
        "unique_true_tokens": int(true_probs.shape[0]),
        "unique_pred_tokens": int(pred_probs.shape[0]),
        "token_support_overlap": int(overlap),
        "token_support_jaccard": float(overlap / union) if union else 1.0,
        "l1_by_group": _l1_by_group_discrete(aligned),
        "js_divergence_bits": _js_divergence_bits(aligned),
        "wasserstein_zscore": _wasserstein_zscore(true_samples, pred_samples),
        "true_counts_csv": true_counts_csv,
        "predicted_counts_csv": pred_counts_csv,
        "true_samples_csv": true_samples_csv,
        "predicted_samples_csv": pred_samples_csv,
        "summary_json": summary_json,
    }


def main():
    parser = argparse.ArgumentParser(description="Quantify token-distribution dissimilarity for a saved analysis run.")
    parser.add_argument("run_dir", help="Path to a dist_analysis run directory containing per-stock output folders.")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    stock_dirs = [
        os.path.join(run_dir, name)
        for name in sorted(os.listdir(run_dir))
        if os.path.isdir(os.path.join(run_dir, name))
    ]
    if not stock_dirs:
        raise RuntimeError(f"No stock directories found under {run_dir}")

    rows = [_summarize_stock(stock_dir) for stock_dir in stock_dirs]
    df = pd.DataFrame(rows).sort_values("stock").reset_index(drop=True)

    csv_path = os.path.join(run_dir, "token_distribution_metrics_summary.csv")
    json_path = os.path.join(run_dir, "token_distribution_metrics_summary.json")
    log_path = os.path.join(run_dir, "token_distribution_metrics_summary.txt")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    lines = []
    lines.append("Token distribution dissimilarity summary")
    lines.append(f"run_dir={run_dir}")
    lines.append("")
    lines.append(
        "stock | samples | unique_true | unique_pred | support_jaccard | L1_by_group | JS_bits | Wasserstein_zscore"
    )
    for row in rows:
        lines.append(
            f"{row['stock']} | {row['sample_count_true']} | {row['unique_true_tokens']} | "
            f"{row['unique_pred_tokens']} | {row['token_support_jaccard']:.4f} | "
            f"{row['l1_by_group']:.6f} | {row['js_divergence_bits']:.6f} | {row['wasserstein_zscore']:.6f}"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- L1_by_group is the exact discrete total-variation style distance over token probabilities.")
    lines.append("- JS_bits is Jensen-Shannon divergence in bits.")
    lines.append("- Wasserstein_zscore mirrors the lob_bench style more closely, but is less semantically clean because token IDs are categorical codes.")

    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("Saved:")
    print(csv_path)
    print(json_path)
    print(log_path)
    print("")
    print(df[["stock", "l1_by_group", "js_divergence_bits", "wasserstein_zscore"]].to_string(index=False))


if __name__ == "__main__":
    main()