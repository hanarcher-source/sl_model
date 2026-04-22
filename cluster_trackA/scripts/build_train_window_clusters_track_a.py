#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track A — build per-window hand features on TRAIN split only, sweep K, write diagnostics + labels.

No real-vs-generated comparison: each row is one contiguous window of merged real-flow events.

Diagnostics (per K, per stock) help choose a *reasonable* K without claiming optimality:
  - inertia (elbow)
  - silhouette (on a random subsample; expensive on full N)
  - Calinski–Harabasz, Davies–Bouldin
  - balance: min cluster count / N, max cluster count / N, effective #clusters exp(entropy(p))

Heuristic recommended_k (see --sil-eps, --min-cluster-frac):
  1) Drop K where smallest cluster fraction < min_cluster_frac OR smallest count < min_cluster_points.
  2) Among remaining, prefer largest silhouette; tie-break lower K.
  3) "Knee" note: report first K (ascending) where silhouette <= best_sil - sil_eps (plateau past meaningful gain).

Outputs under --out-dir/<STOCK_TAG>/:
  - train_window_features.npz  (X, window_start_idx, meta)
  - k_sweep_metrics.json       (per-K: intra/inter centroid distances in standardized space; see intra_inter_centroid_metrics)
  - labels_k_<K>.npz            (only for recommended K unless --save-all-labels)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.preprocessing import StandardScaler
except ImportError as e:  # pragma: no cover
    raise SystemExit("Please install scikit-learn: pip install scikit-learn") from e


DEFAULT_DATA_DIR = (
    "/finance_ML/zhanghaohan/stock_language_model/saved_LOB_stream/processed_real_flow/"
    "pool_0709_0710_openbidanchor_txncomplete"
)


def find_latest_joblib(data_dir: str, day: str, stock_tag: str) -> str:
    pat = os.path.join(
        data_dir,
        f"final_result_for_merge_realflow_*_{day}_{stock_tag}_*.joblib",
    )
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(f"No joblib matching {pat!r}")
    return hits[-1]


def train_window_index_ranges(n_events: int, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same 60/20/20 split as train_blankgpt2_openbidanchor_txncomplete_single_day.build_per_stock_splits."""
    n = max(0, n_events - int(window))
    if n <= 0:
        raise ValueError("Sequence shorter than window")
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx = np.arange(n_train + n_val, n, dtype=np.int64)
    return train_idx, val_idx, test_idx


def _safe_series(w: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in w.columns:
        return pd.Series(default, index=w.index)
    s = pd.to_numeric(w[col], errors="coerce")
    return s.fillna(default)


def window_feature_vector(w: pd.DataFrame) -> np.ndarray:
    """Single-window hand stats (1D float vector)."""
    pm = _safe_series(w, "Price_Mid_diff")
    iv = _safe_series(w, "interval_ms")
    qy = _safe_series(w, "OrderQty")
    mp = _safe_series(w, "MidPrice")

    side = w["Side"] if "Side" in w.columns else pd.Series(0.0, index=w.index)
    side = pd.to_numeric(side, errors="coerce").fillna(-1)

    feats: List[float] = [
        float(pm.mean()),
        float(pm.std(ddof=0)),
        float(pm.abs().mean()),
        float(np.log1p(iv.clip(lower=0)).mean()),
        float(np.log1p(qy.clip(lower=0)).mean()),
        float((side == 49).mean()),
        float((side == 50).mean()),
        float((side == 99).mean()),
        float((side == 129).mean()),
        float((side == 130).mean()),
    ]
    if mp.notna().any() and mp.iloc[0] != 0:
        feats.append(float((mp.iloc[-1] - mp.iloc[0]) / (abs(float(mp.iloc[0])) + 1e-9)))
    else:
        feats.append(0.0)

    if "TransactDT_MS" in w.columns and len(w) > 0:
        t0 = pd.to_datetime(w["TransactDT_MS"].iloc[0])
        feats.append(float(t0.hour) + float(t0.minute) / 60.0)
    else:
        feats.append(12.0)

    return np.asarray(feats, dtype=np.float64)


def build_train_feature_matrix(
    df: pd.DataFrame,
    window: int,
    train_idx: np.ndarray,
    stride: int,
    max_windows: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns X (n_win, n_feat), used_start_indices."""
    df = df.sort_values(["TransactDT_MS", "ChannelNo", "ApplSeqNum"], kind="mergesort").reset_index(drop=True)
    n = len(df)
    picks = train_idx[:: max(1, int(stride))]
    if max_windows > 0 and picks.size > max_windows:
        picks = rng.choice(picks, size=max_windows, replace=False)
        picks = np.sort(picks)

    rows: List[np.ndarray] = []
    used: List[int] = []
    n_pick = int(picks.size)
    report_every = max(5000, n_pick // 20)
    t0 = time.perf_counter()
    print(f"[features_build] window_starts_to_scan={n_pick} report_every={report_every}", flush=True)
    for j, i in enumerate(picks):
        i = int(i)
        if i + window > n:
            continue
        w = df.iloc[i : i + window]
        rows.append(window_feature_vector(w))
        used.append(i)
        done = len(rows)
        if done == 1 or done % report_every == 0:
            dt = time.perf_counter() - t0
            rate = done / max(dt, 1e-9)
            print(
                f"[features_build] windows_done={done} elapsed_sec={dt:.1f} rate_hz={rate:.1f}",
                flush=True,
            )
    if not rows:
        raise RuntimeError("No windows collected (check stride / max_windows / dataframe length).")
    return np.stack(rows, axis=0), np.asarray(used, dtype=np.int64)


def intra_inter_centroid_metrics(
    Xs: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> Dict[str, Any]:
    """
    Euclidean distances in the *same scaled feature space* as KMeans (StandardScaler output).

    - **Intra:** mean distance of each point to its cluster centroid (global mean).
      Also std of per-cluster mean intra distances (how uniform tightness is).
    - **Inter:** pairwise distances between *all* fitted centroids (mean / min / max).
    - **intra_over_inter_mean:** intra_mean / inter_pairwise_mean — lower ⇒ centroids more
      separated relative to within-cluster spread (informal separation ratio).
    """
    k = int(centers.shape[0])
    intra_means: List[float] = []
    w_sum = 0.0
    n_pts = 0
    for c in range(k):
        mask = labels == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        d = np.linalg.norm(Xs[mask] - centers[c], axis=1)
        intra_means.append(float(np.mean(d)))
        w_sum += float(np.sum(d))
        n_pts += n_c
    intra_mean = float(w_sum / max(1, n_pts))
    intra_std_across = float(np.std(intra_means)) if len(intra_means) > 1 else 0.0

    pdists: List[float] = []
    for i in range(k):
        for j in range(i + 1, k):
            pdists.append(float(np.linalg.norm(centers[i] - centers[j])))
    inter_mean = float(np.mean(pdists)) if pdists else 0.0
    inter_min = float(np.min(pdists)) if pdists else 0.0
    inter_max = float(np.max(pdists)) if pdists else 0.0

    ratio = (intra_mean / inter_mean) if inter_mean > 1e-15 else None

    return {
        "space": "standardized_features_euclidean",
        "intra_mean_dist_to_centroid": intra_mean,
        "intra_std_of_per_cluster_mean_dist": intra_std_across,
        "inter_centroid_pairwise_mean": inter_mean,
        "inter_centroid_pairwise_min": inter_min,
        "inter_centroid_pairwise_max": inter_max,
        "intra_over_inter_mean": ratio,
        "n_nonempty_clusters": len(intra_means),
    }


def cluster_balance_stats(labels: np.ndarray, n_clusters: int) -> Dict[str, float]:
    counts = np.bincount(labels, minlength=n_clusters)
    p = counts.astype(np.float64) / max(1.0, float(counts.sum()))
    pnz = p[p > 0]
    ent = -float(np.sum(pnz * np.log(pnz + 1e-15)))
    return {
        "min_cluster_frac": float(counts.min() / max(1, counts.sum())),
        "max_cluster_frac": float(counts.max() / max(1, counts.sum())),
        "n_nonempty_clusters": int(np.sum(counts > 0)),
        "effective_clusters_exp_entropy": float(math.exp(ent)),
    }


def sweep_k(
    X: np.ndarray,
    k_list: List[int],
    random_state: int,
    silhouette_subsample: int,
    use_minibatch: bool,
) -> Tuple[List[Dict[str, Any]], Dict[int, np.ndarray]]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    row_by_k: Dict[int, Dict[str, Any]] = {}
    all_labels: Dict[int, np.ndarray] = {}

    n = Xs.shape[0]
    rng = np.random.default_rng(random_state)
    sil_idx = np.arange(n)
    if silhouette_subsample > 0 and n > silhouette_subsample:
        sil_idx = rng.choice(np.arange(n), size=silhouette_subsample, replace=False)

    ks_valid = [k for k in k_list if 2 <= k <= n - 1]
    sweep_t0 = time.perf_counter()
    print(
        f"[k_sweep] n_samples={n} backend={'MiniBatchKMeans' if use_minibatch else 'KMeans'} "
        f"n_k={len(ks_valid)} ks={ks_valid} sil_subsample={len(sil_idx)}",
        flush=True,
    )
    for step, k in enumerate(ks_valid, start=1):
        t_k = time.perf_counter()
        km = (
            MiniBatchKMeans(
                n_clusters=k,
                random_state=random_state,
                batch_size=min(4096, max(256, n // 4)),
                n_init="auto",
            )
            if use_minibatch
            else KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        )
        labels = km.fit_predict(Xs).astype(np.int32)
        all_labels[k] = labels
        bal = cluster_balance_stats(labels, k)
        ii = intra_inter_centroid_metrics(Xs, labels, km.cluster_centers_)
        row: Dict[str, Any] = {
            "k": int(k),
            "inertia": float(km.inertia_),
            "silhouette_full_sample": False,
            "silhouette": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
            **bal,
            **ii,
        }
        if len(np.unique(labels[sil_idx])) >= 2:
            row["silhouette"] = float(silhouette_score(Xs[sil_idx], labels[sil_idx]))
            row["silhouette_full_sample"] = sil_idx.size == n
        try:
            row["calinski_harabasz"] = float(calinski_harabasz_score(Xs, labels))
            row["davies_bouldin"] = float(davies_bouldin_score(Xs, labels))
        except Exception:
            pass
        row_by_k[k] = row
        dt_k = time.perf_counter() - t_k
        sil_s = f"{row['silhouette']:.4f}" if row.get("silhouette") is not None else "na"
        i_over_e = row.get("intra_over_inter_mean")
        ioe_s = f"{i_over_e:.4f}" if i_over_e is not None else "na"
        print(
            f"[k_sweep] step={step}/{len(ks_valid)} k={k} inertia={row['inertia']:.4g} "
            f"silhouette={sil_s} min_clust_frac={row['min_cluster_frac']:.4f} "
            f"intra_mean={row['intra_mean_dist_to_centroid']:.4f} "
            f"inter_mean={row['inter_centroid_pairwise_mean']:.4f} intra/inter={ioe_s} sec={dt_k:.2f}",
            flush=True,
        )

    print(f"[k_sweep] total_sec={time.perf_counter() - sweep_t0:.2f}", flush=True)
    ordered = [row_by_k[k] for k in sorted(row_by_k)]
    return ordered, all_labels


def recommend_k(
    metrics: List[Dict[str, Any]],
    min_cluster_frac: float,
    min_cluster_points: int,
    sil_eps: float,
    n_samples: int,
) -> Dict[str, Any]:
    """Pick K with best silhouette among feasible; report plateau K."""
    feasible = []
    for r in metrics:
        k = r["k"]
        sil = r.get("silhouette")
        if sil is None:
            continue
        min_frac = r["min_cluster_frac"]
        min_count_est = min_frac * n_samples
        if min_frac < min_cluster_frac or min_count_est < min_cluster_points:
            continue
        feasible.append(r)
    if not feasible:
        feasible = [r for r in metrics if r.get("silhouette") is not None]
    best = max(feasible, key=lambda r: float(r["silhouette"]))
    best_sil = float(best["silhouette"])
    best_k = int(best["k"])

    plateau_k = None
    for r in sorted(feasible, key=lambda x: x["k"]):
        if float(r["silhouette"]) >= best_sil - sil_eps:
            plateau_k = int(r["k"])
        else:
            break

    return {
        "recommended_k": best_k,
        "recommended_silhouette": best_sil,
        "plateau_k_at_eps": plateau_k,
        "sil_eps": sil_eps,
        "feasible_ks": [int(r["k"]) for r in feasible],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Track A: train-window features + K sweep + labels.")
    ap.add_argument("--stock-tag", required=True, help="e.g. 000617XSHE (no underscore)")
    ap.add_argument("--day", default="20250709", help="Trade date string in joblib filename")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--joblib", default="", help="Explicit path; else latest under data-dir")
    ap.add_argument("--window-len", type=int, default=50)
    ap.add_argument("--stride", type=int, default=1, help="Take every stride-th train window start (speed).")
    ap.add_argument("--max-windows", type=int, default=0, help="0 = no cap; else random subsample cap.")
    ap.add_argument("--k-min", type=int, default=8)
    ap.add_argument("--k-max", type=int, default=48)
    ap.add_argument("--k-step", type=int, default=4)
    ap.add_argument("--silhouette-subsample", type=int, default=15000)
    ap.add_argument("--min-cluster-frac", type=float, default=0.01)
    ap.add_argument("--min-cluster-points", type=int, default=80)
    ap.add_argument("--sil-eps", type=float, default=0.015, help="Plateau: sil within this of best → stop growing K.")
    ap.add_argument("--minibatch", action="store_true", help="Use MiniBatchKMeans (faster, large N).")
    ap.add_argument("--minibatch-n", type=int, default=200_000, help="Use minibatch if n_windows >= this.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--save-all-labels", action="store_true")
    args = ap.parse_args()

    stock_t0 = time.perf_counter()
    jpath = args.joblib.strip() or find_latest_joblib(args.data_dir, args.day, args.stock_tag)
    print(f"[load] {jpath}", flush=True)
    t_ld = time.perf_counter()
    df = joblib.load(jpath)
    print(f"[load] rows={len(df)} sec={time.perf_counter() - t_ld:.2f}", flush=True)
    if "SecurityID" in df.columns:
        tags = df["SecurityID"].astype(str).str.replace("_", "", regex=False).unique()
        if len(tags) > 1:
            print(f"[warn] multiple SecurityID in frame: {tags.tolist()[:5]}...", flush=True)
        df = df[df["SecurityID"].astype(str).str.replace("_", "", regex=False) == args.stock_tag].copy()

    window = int(args.window_len)
    train_idx, val_idx, test_idx = train_window_index_ranges(len(df), window)
    rng = np.random.default_rng(int(args.random_state))
    X, used_idx = build_train_feature_matrix(
        df, window, train_idx, int(args.stride), int(args.max_windows), rng
    )
    print(f"[features] n_windows={X.shape[0]} dim={X.shape[1]} train_starts_used={used_idx.size}", flush=True)

    k_list = list(range(int(args.k_min), int(args.k_max) + 1, int(args.k_step)))
    use_mb = bool(args.minibatch) or X.shape[0] >= int(args.minibatch_n)
    ordered, labels_by_k = sweep_k(
        X,
        k_list,
        int(args.random_state),
        int(args.silhouette_subsample),
        use_mb,
    )

    rec = recommend_k(
        ordered,
        float(args.min_cluster_frac),
        int(args.min_cluster_points),
        float(args.sil_eps),
        n_samples=X.shape[0],
    )

    out_root = os.path.join(args.out_dir, args.stock_tag)
    os.makedirs(out_root, exist_ok=True)

    meta = {
        "stock_tag": args.stock_tag,
        "day": args.day,
        "joblib": jpath,
        "window_len": window,
        "stride": int(args.stride),
        "max_windows": int(args.max_windows),
        "n_train_windows_total": int(train_idx.size),
        "n_windows_used": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "k_sweep": ordered,
        "recommendation": rec,
        "notes": {
            "silhouette": "Subsampled unless n <= silhouette_subsample; higher is more separated clusters.",
            "davies_bouldin": "Lower is better (compact, well-separated).",
            "calinski_harabasz": "Higher is better.",
            "min_cluster_frac": "Drop K with tiny clusters before trusting silhouette.",
            "plateau_k_at_eps": "Smallest K in ascending order that stays within sil_eps of best silhouette (parsimony).",
        },
    }
    with open(os.path.join(out_root, "k_sweep_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    np.savez_compressed(
        os.path.join(out_root, "train_window_features.npz"),
        X=X,
        window_start_idx=used_idx,
        stock_tag=np.array(args.stock_tag),
        day=np.array(args.day),
    )

    rk = int(rec["recommended_k"])
    np.savez_compressed(
        os.path.join(out_root, f"labels_k_{rk}.npz"),
        labels=labels_by_k[rk],
        window_start_idx=used_idx,
        k=rk,
    )
    if args.save_all_labels:
        for k, lab in labels_by_k.items():
            np.savez_compressed(os.path.join(out_root, f"labels_k_{k}.npz"), labels=lab, window_start_idx=used_idx, k=k)

    print(json.dumps(rec, indent=2), flush=True)
    print(
        f"[done] stock={args.stock_tag} total_sec={time.perf_counter() - stock_t0:.1f} "
        f"wrote {out_root}/k_sweep_metrics.json labels_k_{rk}.npz",
        flush=True,
    )


if __name__ == "__main__":
    main()
