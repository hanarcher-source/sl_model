#!/usr/bin/env python3
"""
Generate deterministic cluster captions (anchor sentences) from saved Track A artifacts.

Input per stock dir:
  - train_window_features.npz  (X)
  - labels_k_<K>.npz           (labels, k)

Output (written into the same stock dir):
  - preset_anchors_clustered_k<K>.txt   (one line per cluster id)
  - captions_debug_k<K>.json            (centroid feature means + chosen buckets/traits)

Captions are derived only from Track A hand features (no reference comparison).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np


# Must match build_train_window_clusters_track_a.py feature order.
FEATURES: List[Tuple[str, str]] = [
    ("pm_mean", "Price-Mid diff mean"),
    ("pm_std", "Price-Mid diff std"),
    ("pm_abs_mean", "Abs(Price-Mid diff) mean"),
    ("log_interval_mean", "log(1+interval_ms) mean"),
    ("log_qty_mean", "log(1+OrderQty) mean"),
    ("frac_side_49", "fraction side=49"),
    ("frac_side_50", "fraction side=50"),
    ("frac_side_99", "fraction side=99 (cancel)"),
    ("frac_side_129", "fraction side=129 (exec)"),
    ("frac_side_130", "fraction side=130"),
    ("mid_return", "mid return over window"),
    ("tod_hour", "time-of-day hour.fraction"),
]


def _quantile_edges(x: np.ndarray) -> List[float]:
    # 4 edges → 5 buckets
    return [float(np.quantile(x, q)) for q in (0.2, 0.4, 0.6, 0.8)]


def _bucket(v: float, edges: List[float], labels: List[str]) -> str:
    for i, e in enumerate(edges):
        if v <= e:
            return labels[i]
    return labels[-1]


def _top_deltas(z: np.ndarray, feat_names: List[str], k: int, exclude: set[str]) -> List[Tuple[str, float]]:
    pairs = [(feat_names[i], float(z[i])) for i in range(len(z)) if feat_names[i] not in exclude]
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return pairs[:k]


def _phrase_for_delta(name: str, z: float) -> str:
    # Use only categorical language.
    if name == "pm_mean":
        return "off-mid bias" if z > 0 else "near-mid bias"
    if name == "pm_std":
        return "volatile quotes" if z > 0 else "stable quotes"
    if name == "frac_side_49":
        return "more side49" if z > 0 else "less side49"
    if name == "frac_side_50":
        return "more side50" if z > 0 else "less side50"
    if name == "frac_side_130":
        return "more side130" if z > 0 else "less side130"
    return f"{name}:{'high' if z > 0 else 'low'}"


def make_caption(
    feat_mean: np.ndarray,
    feat_z: np.ndarray,
    global_edges: Dict[str, List[float]],
    stock_tag: str,
    cluster_id: int,
) -> Tuple[str, Dict[str, Any]]:
    feat_names = [n for n, _ in FEATURES]
    fm = {feat_names[i]: float(feat_mean[i]) for i in range(len(feat_names))}

    spread_bucket = _bucket(fm["pm_abs_mean"], global_edges["pm_abs_mean"], ["very tight", "tight", "medium", "wide", "very wide"])
    tempo_bucket = _bucket(fm["log_interval_mean"], global_edges["log_interval_mean"], ["very fast", "fast", "normal", "slow", "very slow"])
    size_bucket = _bucket(fm["log_qty_mean"], global_edges["log_qty_mean"], ["very small", "small", "medium", "large", "very large"])
    cancel_bucket = _bucket(fm["frac_side_99"], global_edges["frac_side_99"], ["very low cancels", "low cancels", "normal cancels", "high cancels", "very high cancels"])
    exec_bucket = _bucket(fm["frac_side_129"], global_edges["frac_side_129"], ["very low executions", "low executions", "normal executions", "high executions", "very high executions"])
    drift_bucket = _bucket(fm["mid_return"], global_edges["mid_return"], ["down", "slightly down", "flat", "slightly up", "up"])

    tod = fm["tod_hour"]
    if tod < 10.0:
        session = "open/morning"
    elif tod < 12.0:
        session = "late morning"
    elif tod < 14.0:
        session = "early afternoon"
    else:
        session = "late afternoon"

    exclude = {
        "pm_abs_mean",
        "log_interval_mean",
        "log_qty_mean",
        "frac_side_99",
        "frac_side_129",
        "mid_return",
        "tod_hour",
    }
    deltas = _top_deltas(feat_z, feat_names, k=3, exclude=exclude)
    traits = [_phrase_for_delta(n, z) for n, z in deltas]

    sig = (
        f"spr:{spread_bucket}|tmp:{tempo_bucket}|sz:{size_bucket}|"
        f"cxl:{cancel_bucket}|exe:{exec_bucket}|dr:{drift_bucket}|sess:{session}"
    )

    line = (
        f"{spread_bucket} quotes, {tempo_bucket} tempo, {size_bucket} size, {cancel_bucket}, {exec_bucket}, "
        f"drift={drift_bucket}, session={session}. Traits: {', '.join(traits)}. sig={sig}"
    )

    dbg = {
        "stock": stock_tag,
        "cluster": int(cluster_id),
        "buckets": {
            "spread": spread_bucket,
            "tempo": tempo_bucket,
            "size": size_bucket,
            "cancels": cancel_bucket,
            "executions": exec_bucket,
            "drift": drift_bucket,
            "session": session,
        },
        "traits": traits,
        "sig": sig,
        "feature_means": fm,
        "top_deltas": deltas,
    }
    return line, dbg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock-dir", required=True, help=".../<RUN>/<STOCK_TAG>")
    ap.add_argument("--out-prefix", default="preset_anchors_clustered")
    args = ap.parse_args()

    stock_dir = os.path.abspath(args.stock_dir)
    stock_tag = os.path.basename(stock_dir.rstrip("/"))

    feat_path = os.path.join(stock_dir, "train_window_features.npz")
    if not os.path.isfile(feat_path):
        raise FileNotFoundError(feat_path)

    label_hits = sorted([p for p in os.listdir(stock_dir) if p.startswith("labels_k_") and p.endswith(".npz")])
    if not label_hits:
        raise FileNotFoundError(f"No labels_k_*.npz in {stock_dir}")
    labels_path = os.path.join(stock_dir, label_hits[-1])

    feat = np.load(feat_path)
    lab = np.load(labels_path)

    X = np.asarray(feat["X"], dtype=np.float64)
    labels = np.asarray(lab["labels"], dtype=np.int32).ravel()
    k = int(np.asarray(lab.get("k", labels.max() + 1)).ravel()[0])

    if X.shape[1] != len(FEATURES):
        raise RuntimeError(f"Unexpected feature dim {X.shape[1]} (expected {len(FEATURES)})")

    fn = [n for n, _ in FEATURES]
    edges: Dict[str, List[float]] = {}
    for name in ("pm_abs_mean", "log_interval_mean", "log_qty_mean", "frac_side_99", "frac_side_129", "mid_return"):
        edges[name] = _quantile_edges(X[:, fn.index(name)])

    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12

    lines: List[str] = []
    debugs: List[Dict[str, Any]] = []
    for c in range(k):
        mask = labels == c
        if not mask.any():
            lines.append(f"(empty cluster {c})")
            continue
        m = X[mask].mean(axis=0)
        z = (m - mu) / sd
        line, dbg = make_caption(m, z, edges, stock_tag, c)
        lines.append(line)
        debugs.append(dbg)

    out_txt = os.path.join(stock_dir, f"{args.out_prefix}_k{k}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip() + "\n")

    out_json = os.path.join(stock_dir, f"captions_debug_k{k}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"stock": stock_tag, "k": k, "edges": edges, "captions": debugs}, f, indent=2)

    print(f"[wrote] {out_txt}")
    print(f"[wrote] {out_json}")


if __name__ == "__main__":
    main()

