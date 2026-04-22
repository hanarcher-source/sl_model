#!/usr/bin/env python3
"""
Intra / inter centroid distance metrics from saved Track A artifacts only (no re-clustering).

Uses:
  - train_window_features.npz  (array X)
  - labels_k_<K>.npz            (labels, k)

StandardScaler fit on X, centroids = per-cluster mean in scaled space (same geometry as KMeans).

Examples:
  python posthoc_intra_inter_from_npz.py \\
    --features-npz .../000617XSHE/train_window_features.npz \\
    --labels-npz .../000617XSHE/labels_k_8.npz

  python posthoc_intra_inter_from_npz.py \\
    --run-root .../cluster_runs/pool0709_train_54357 \\
    --write-json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler


def intra_inter_centroid_metrics(
    Xs: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> Dict[str, Any]:
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


def compute_centroids(Xs: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    centers = np.zeros((k, Xs.shape[1]), dtype=np.float64)
    for c in range(k):
        mask = labels == c
        if mask.any():
            centers[c] = Xs[mask].mean(axis=0)
    return centers


def run_one(features_npz: str, labels_npz: str) -> Dict[str, Any]:
    feat = np.load(features_npz)
    lab = np.load(labels_npz)
    X = np.asarray(feat["X"], dtype=np.float64)
    labels = np.asarray(lab["labels"], dtype=np.int32).ravel()
    k_meta = lab.get("k")
    if k_meta is not None:
        k = int(np.asarray(k_meta).ravel()[0])
    else:
        k = int(labels.max()) + 1

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    centers = compute_centroids(Xs, labels, k)
    out = intra_inter_centroid_metrics(Xs, labels, centers)
    out["k"] = k
    out["n_points"] = int(X.shape[0])
    out["features_npz"] = os.path.abspath(features_npz)
    out["labels_npz"] = os.path.abspath(labels_npz)
    out["note"] = "post-hoc from saved npz; centroids=cluster means on standardized X"
    return out


def find_label_npz(stock_dir: str) -> Optional[str]:
    hits = sorted(glob.glob(os.path.join(stock_dir, "labels_k_*.npz")))
    return hits[-1] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-npz", default="", help="train_window_features.npz")
    ap.add_argument("--labels-npz", default="", help="labels_k_*.npz")
    ap.add_argument(
        "--run-root",
        default="",
        help="Parent run dir (contains <STOCK>/train_window_features.npz + labels_k_*.npz)",
    )
    ap.add_argument(
        "--write-json",
        action="store_true",
        help="With --run-root, write <STOCK>/intra_inter_metrics.json per stock",
    )
    args = ap.parse_args()

    if args.run_root:
        root = os.path.abspath(args.run_root)
        rows = []
        for name in sorted(os.listdir(root)):
            sd = os.path.join(root, name)
            if not os.path.isdir(sd):
                continue
            fp = os.path.join(sd, "train_window_features.npz")
            lp = find_label_npz(sd)
            if not os.path.isfile(fp) or not lp:
                continue
            out = run_one(fp, lp)
            out["stock_tag"] = name
            rows.append(out)
            print(f"### {name} k={out['k']}")
            print(json.dumps({k: v for k, v in out.items() if k not in ("features_npz", "labels_npz")}, indent=2))
            if args.write_json:
                jpath = os.path.join(sd, "intra_inter_metrics.json")
                with open(jpath, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)
                print(f"[wrote] {jpath}")
        if not rows:
            print("No stock subdirs with train_window_features.npz + labels_k_*.npz found.", file=sys.stderr)
            sys.exit(1)
        return

    if not args.features_npz or not args.labels_npz:
        ap.error("Need --features-npz and --labels-npz, or --run-root")
    out = run_one(args.features_npz, args.labels_npz)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
