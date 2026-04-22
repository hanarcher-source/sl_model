#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed-start autoregressive generation + evaluation for 1Hz tokenized book-state model (parallel heads).

Inputs:
  - checkpoint from train_bookstate_parallelheads_anchor5m.py
  - preprocessed 1Hz book-token joblib from preprocess_bookstate_mdl628_anchor5m_bins_20250709.py

Procedure:
  - pick a start time (HH:MM:SS) for a stock
  - take context_sec snapshots as seed (ground truth)
  - generate horizon_sec snapshots autoregressively (sample or greedy) using the model
  - evaluate generated vs real over the same horizon:
      * per-slot token histogram L1 distance (avg over 20 slots)
      * token marginal entropy (avg over slots)
      * per-slot mean/var of decoded (price_bin, vol_bin) indices

This is not a full LOBSTER-stream evaluation; it evaluates at the snapshot level.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _resolve_lob_bench_on_path():
    # Mirror eval_generated_stream.py path discovery.
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, "..", "LOB_bench")),
        os.path.abspath(os.path.join(here, "..", "..", "LOB_bench")),
        os.path.abspath(os.path.join(here, "..", "..", "..", "LOB_bench")),
        os.path.abspath(os.path.join(here, "..", "..", "..", "lob_bench-main")),  # common alt
        os.path.abspath(os.path.join(here, "..", "..", "lob_bench-main")),
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            return p
    return None


def _stock_tag(stock: str) -> str:
    return str(stock).replace("_", "")


def _parse_hms(hms: str) -> pd.Timestamp:
    t = pd.to_datetime(str(hms).strip(), format="%H:%M:%S", errors="raise")
    return pd.Timestamp(year=1900, month=1, day=1, hour=t.hour, minute=t.minute, second=t.second)


def _load_model(ckpt_path: str, K: int) -> nn.Module:
    from train_bookstate_parallelheads_anchor5m import BookStateTemporalParallelHeads

    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj.get("model_state_dict", obj)
    # Infer whether this checkpoint used a pretrained GPT-2 backbone.
    # Pretrained checkpoints include GPT-2's token embedding matrix wte with vocab size 50257.
    init_pretrained = False
    try:
        wte = state.get("gpt2.wte.weight", None)
        if wte is not None and hasattr(wte, "shape") and int(wte.shape[0]) > 10:
            init_pretrained = True
    except Exception:
        init_pretrained = False

    model = BookStateTemporalParallelHeads(
        codebook_size=int(K),
        init_pretrained_backbone=bool(init_pretrained),
        pretrained_backbone_name="gpt2",
        pretrained_local_only=True,
    )
    model.load_state_dict(state, strict=True)
    return model


def _hist_l1(a: np.ndarray, b: np.ndarray, K: int) -> float:
    ha = np.bincount(a.astype(np.int64), minlength=K).astype(np.float64)
    hb = np.bincount(b.astype(np.int64), minlength=K).astype(np.float64)
    if ha.sum() > 0:
        ha /= ha.sum()
    if hb.sum() > 0:
        hb /= hb.sum()
    return float(np.abs(ha - hb).sum())


def _entropy(p: np.ndarray) -> float:
    p = p.astype(np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _marginal_entropy(tokens: np.ndarray, K: int) -> float:
    h = np.bincount(tokens.astype(np.int64), minlength=K).astype(np.float64)
    if h.sum() <= 0:
        return 0.0
    h /= h.sum()
    return _entropy(h)


@dataclass
class EvalSummary:
    stock: str
    start_time: str
    context_sec: int
    horizon_sec: int
    K: int
    seed_mode: str
    decode_P: int
    decode_V: int
    token_l1_avg_slots: float
    token_entropy_real_avg_slots: float
    token_entropy_gen_avg_slots: float
    price_bin_mean_abs_diff_avg_slots: float
    vol_bin_mean_abs_diff_avg_slots: float


def _lb_median(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _lb_iqm(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    if n == 1:
        return float(s[0])
    idx25 = (n - 1) * 0.25
    idx75 = (n - 1) * 0.75

    def _interp(idx: float) -> float:
        lo = int(np.floor(idx))
        hi = int(np.ceil(idx))
        if lo == hi:
            return float(s[lo])
        w = idx - lo
        return float(s[lo] * (1.0 - w) + s[hi] * w)

    q25 = _interp(idx25)
    q75 = _interp(idx75)
    core = [v for v in s if q25 <= v <= q75]
    if not core:
        return float(sum(s)) / float(len(s))
    return float(sum(core)) / float(len(core))


def _lb_aggregate(vals: list[float]) -> dict:
    xs = [float(x) for x in vals if np.isfinite(x)]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "iqm": None}
    return {
        "n": int(len(xs)),
        "mean": float(sum(xs)) / float(len(xs)),
        "median": _lb_median(xs),
        "iqm": _lb_iqm(xs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stock", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-joblib", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start-time", default="10:00:00")
    p.add_argument("--context-sec", type=int, default=60)
    p.add_argument("--horizon-sec", type=int, default=600)
    p.add_argument("--codebook-size", type=int, default=1271)
    p.add_argument("--P", type=int, default=41)
    p.add_argument("--V", type=int, default=31)
    p.add_argument("--meta-json", default="", help="Meta JSON from preprocessing (needed for vol_edges when computing snapshot metrics).")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--run-tag", default="", help="Optional string tag appended to output filename (e.g. temp0p7).")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tag = str(args.run_tag).strip()
    tag_part = f"_{tag}" if tag else ""
    out_json = os.path.join(
        args.output_dir,
        f"eval_bookstate_1s_{_stock_tag(args.stock)}_{args.start_time.replace(':','')}_T{args.horizon_sec}_ctx{args.context_sec}{tag_part}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    df = joblib.load(args.data_joblib)
    df = df[df["SecurityID"] == str(args.stock)].copy()
    if df.empty:
        raise RuntimeError(f"No rows for stock={args.stock} in {args.data_joblib}")
    df = df.sort_values(["TransactDT_SEC"], kind="mergesort").reset_index(drop=True)

    tok_cols = [f"book_token_{j:02d}" for j in range(20)]
    tokens = df[tok_cols].to_numpy(np.int64)
    times = pd.Series(df["TransactDT_SEC"]).astype("datetime64[ns]").to_numpy()
    start_ts = _parse_hms(args.start_time)

    # locate the first index at/after start_ts
    idx0 = int(np.searchsorted(times.astype("datetime64[ns]"), np.datetime64(start_ts)))
    i_seed0 = idx0
    i_seed1 = idx0 + int(args.context_sec)
    i_real_end = i_seed1 + int(args.horizon_sec)
    if i_real_end >= tokens.shape[0]:
        raise RuntimeError("Not enough data after start-time for requested context+horizon.")

    seed = tokens[i_seed0:i_seed1]  # [L,20]
    real = tokens[i_seed1:i_real_end]  # [T,20]

    # Optional snapshot-metric evaluation (requires absolute and anchor cols in joblib + meta.json for vol_edges)
    have_raw = ("AskPrice1" in df.columns) and ("BidPrice1" in df.columns) and ("AskPrice1_A" in df.columns) and ("BidPrice1_A" in df.columns)
    vol_edges = None
    tick_size = 0.01
    if have_raw:
        meta_path = str(args.meta_json).strip()
        if not meta_path:
            raise RuntimeError("Raw/anchor columns present but --meta-json not provided (needed for vol_edges).")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        vol_edges = np.asarray(meta["vol_edges"], dtype=np.float64)
        tick_size = float(meta.get("tick_size", tick_size))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.checkpoint, K=int(args.codebook_size)).to(device)
    model.eval()

    gen = np.zeros_like(real)
    ctx = seed.copy()
    temp = float(max(args.temperature, 1e-6))

    with torch.no_grad():
        for t in range(int(args.horizon_sec)):
            x = torch.from_numpy(ctx[-int(args.context_sec) :]).unsqueeze(0).to(device)  # [1,L,20]
            logits = model(x).squeeze(0)  # [20,K]
            if args.sample:
                probs = torch.softmax(logits / temp, dim=1)
                next_tok = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tok = torch.argmax(logits, dim=1)
            nxt = next_tok.detach().cpu().numpy().astype(np.int64)
            gen[t] = nxt
            ctx = np.concatenate([ctx, nxt[None, :]], axis=0)

    K = int(args.codebook_size)
    P = int(args.P)
    V = int(args.V)
    if P * V != K:
        raise ValueError(f"P*V must equal K. Got P={P} V={V} K={K}")

    # Slot-wise basic stats (cheap)
    l1_slots = []
    ent_real = []
    ent_gen = []
    pb_diff = []
    vb_diff = []
    for s in range(20):
        l1_slots.append(_hist_l1(real[:, s], gen[:, s], K=K))
        ent_real.append(_marginal_entropy(real[:, s], K=K))
        ent_gen.append(_marginal_entropy(gen[:, s], K=K))
        # decode joint token -> price_bin, vol_bin
        pr = (real[:, s] // V).astype(np.int64)
        vr = (real[:, s] % V).astype(np.int64)
        pg = (gen[:, s] // V).astype(np.int64)
        vg = (gen[:, s] % V).astype(np.int64)
        pb_diff.append(float(np.abs(pr.mean() - pg.mean())))
        vb_diff.append(float(np.abs(vr.mean() - vg.mean())))

    # Slot-wise LOB-Bench-style Wasserstein and L1_by_group over discrete token ids.
    _resolve_lob_bench_on_path()
    try:
        import metrics as lbm  # type: ignore
        import partitioning as lbp  # type: ignore
    except Exception:
        lbm = None
        lbp = None

    ref_metrics = {}
    ws_slots = []
    l1bg_slots = []
    if lbm is not None and lbp is not None:
        for s in range(20):
            r = real[:, s].astype(np.float64)
            g = gen[:, s].astype(np.float64)
            # Wasserstein via LOB_bench helper expects a dataframe with columns score/type.
            ws_df = pd.DataFrame(
                {
                    "score": np.concatenate([r, g]),
                    "type": (["real"] * len(r)) + (["generated"] * len(g)),
                }
            )
            w = float(lbm.wasserstein(ws_df, bootstrap_ci=False))
            groups_real, groups_gen = lbp.group_by_score(r, [g], discrete=True)
            l1_df = lbp.get_score_table(r, [g], groups_real, groups_gen)
            l1bg = float(lbm.l1_by_group(l1_df, bootstrap_ci=False))
            name = f"slot_{s:02d}_token"
            ref_metrics[name] = {"wasserstein": w, "l1_by_group": l1bg}
            ws_slots.append(w)
            l1bg_slots.append(l1bg)

    lobbench_style_overall = {
        "wasserstein": _lb_aggregate(ws_slots),
        "l1_by_group": _lb_aggregate(l1bg_slots),
        "note": "Aggregated across 20 per-slot discrete-token losses (1Hz snapshot eval).",
    }

    summary = EvalSummary(
        stock=str(args.stock),
        start_time=str(args.start_time),
        context_sec=int(args.context_sec),
        horizon_sec=int(args.horizon_sec),
        K=K,
        seed_mode="real_context",
        decode_P=P,
        decode_V=V,
        token_l1_avg_slots=float(np.mean(l1_slots)),
        token_entropy_real_avg_slots=float(np.mean(ent_real)),
        token_entropy_gen_avg_slots=float(np.mean(ent_gen)),
        price_bin_mean_abs_diff_avg_slots=float(np.mean(pb_diff)),
        vol_bin_mean_abs_diff_avg_slots=float(np.mean(vb_diff)),
    )

    payload = {
        "summary": asdict(summary),
        "reference_comparison": {"metrics": ref_metrics} if ref_metrics else None,
        "lobbench_style_overall": lobbench_style_overall,
        "per_slot": {
            "token_l1": l1_slots,
            "entropy_real": ent_real,
            "entropy_gen": ent_gen,
            "wasserstein_token": ws_slots if ws_slots else None,
            "l1_by_group_token": l1bg_slots if l1bg_slots else None,
            "price_bin_mean_abs_diff": pb_diff,
            "vol_bin_mean_abs_diff": vb_diff,
        },
        "paths": {"checkpoint": args.checkpoint, "data_joblib": args.data_joblib},
    }

    if have_raw and vol_edges is not None:
        V_edges = vol_edges
        V_centers = (V_edges[:-1] + V_edges[1:]) / 2.0
        levels = 10
        df_h = df.iloc[i_seed1:i_real_end].reset_index(drop=True)

        bid1 = df_h["BidPrice1"].to_numpy(np.float64)
        ask1 = df_h["AskPrice1"].to_numpy(np.float64)
        spread = ask1 - bid1
        mid = (ask1 + bid1) / 2.0
        mid_ret = np.diff(np.log(np.maximum(mid, 1e-12)), prepend=np.nan)

        bid_depth10 = np.zeros(len(df_h), dtype=np.float64)
        ask_depth10 = np.zeros(len(df_h), dtype=np.float64)
        for i in range(1, levels + 1):
            bid_depth10 += df_h[f"BidVolume{i}"].to_numpy(np.float64)
            ask_depth10 += df_h[f"AskVolume{i}"].to_numpy(np.float64)
        imb10 = (bid_depth10 - ask_depth10) / np.maximum(bid_depth10 + ask_depth10, 1e-12)

        bid_pA = np.stack([df_h[f"BidPrice{i}_A"].to_numpy(np.float64) for i in range(1, levels + 1)], axis=1)
        ask_pA = np.stack([df_h[f"AskPrice{i}_A"].to_numpy(np.float64) for i in range(1, levels + 1)], axis=1)
        bid_vA = np.stack([df_h[f"BidVolume{i}_A"].to_numpy(np.float64) for i in range(1, levels + 1)], axis=1)
        ask_vA = np.stack([df_h[f"AskVolume{i}_A"].to_numpy(np.float64) for i in range(1, levels + 1)], axis=1)

        gen_tok = gen.astype(np.int64)
        pb = (gen_tok // V).astype(np.int64)
        vb = (gen_tok % V).astype(np.int64)
        dp_ticks = (pb - (P // 2)).astype(np.int64)
        dv = V_centers[vb]

        dp_ask = dp_ticks[:, :levels]
        dp_bid = dp_ticks[:, levels:]
        dv_ask = dv[:, :levels]
        dv_bid = dv[:, levels:]

        ask_pG = ask_pA + dp_ask * float(tick_size)
        bid_pG = bid_pA + dp_bid * float(tick_size)
        ask_vG = np.expm1(np.log1p(np.maximum(ask_vA, 0.0)) + dv_ask)
        bid_vG = np.expm1(np.log1p(np.maximum(bid_vA, 0.0)) + dv_bid)

        bid1_g = bid_pG[:, 0]
        ask1_g = ask_pG[:, 0]
        spread_g = ask1_g - bid1_g
        mid_g = (ask1_g + bid1_g) / 2.0
        mid_ret_g = np.diff(np.log(np.maximum(mid_g, 1e-12)), prepend=np.nan)
        bid_depth10_g = bid_vG.sum(axis=1)
        ask_depth10_g = ask_vG.sum(axis=1)
        imb10_g = (bid_depth10_g - ask_depth10_g) / np.maximum(bid_depth10_g + ask_depth10_g, 1e-12)

        _resolve_lob_bench_on_path()
        import metrics as lbm2  # type: ignore
        import partitioning as lbp2  # type: ignore

        def _compare(rx: np.ndarray, gx: np.ndarray, *, discrete: bool):
            rx = rx[np.isfinite(rx)].astype(np.float64)
            gx = gx[np.isfinite(gx)].astype(np.float64)
            ws_df = pd.DataFrame(
                {"score": np.concatenate([rx, gx]), "type": (["real"] * len(rx)) + (["generated"] * len(gx))}
            )
            w = float(lbm2.wasserstein(ws_df, bootstrap_ci=False))
            if discrete:
                gr, gg = lbp2.group_by_score(rx, [gx], discrete=True)
            else:
                gr, gg = lbp2.group_by_score(rx, [gx], bin_method="fd")
            l1_df = lbp2.get_score_table(rx, [gx], gr, gg)
            l1bg = float(lbm2.l1_by_group(l1_df, bootstrap_ci=False))
            return {"wasserstein": w, "l1_by_group": l1bg, "n_real": int(len(rx)), "n_generated": int(len(gx))}

        snap = {
            "spread": _compare(spread, spread_g, discrete=True),
            "mid_return_1s_log": _compare(mid_ret, mid_ret_g, discrete=False),
            "bid_depth_10": _compare(bid_depth10, bid_depth10_g, discrete=False),
            "ask_depth_10": _compare(ask_depth10, ask_depth10_g, discrete=False),
            "depth_imbalance_10": _compare(imb10, imb10_g, discrete=False),
        }
        payload["snapshot_reference_comparison"] = {"metrics": snap}
        ws_vals = [v["wasserstein"] for v in snap.values()]
        l1_vals = [v["l1_by_group"] for v in snap.values()]
        payload["snapshot_lobbench_style_overall"] = {
            "wasserstein": _lb_aggregate(ws_vals),
            "l1_by_group": _lb_aggregate(l1_vals),
            "metric_names": list(snap.keys()),
        }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("[ok] wrote", out_json)
    print("[token_l1_avg_slots]", summary.token_l1_avg_slots)
    if lobbench_style_overall["wasserstein"]["n"]:
        w = lobbench_style_overall["wasserstein"]
        l1 = lobbench_style_overall["l1_by_group"]
        print(
            "[lobbench6] W_mean={:.6g} W_median={:.6g} W_iqm={:.6g} | L1_mean={:.6g} L1_median={:.6g} L1_iqm={:.6g}".format(
                w["mean"], w["median"], w["iqm"], l1["mean"], l1["median"], l1["iqm"]
            )
        )
    if payload.get("snapshot_lobbench_style_overall", {}).get("wasserstein", {}).get("n", 0):
        sw = payload["snapshot_lobbench_style_overall"]["wasserstein"]
        sl1 = payload["snapshot_lobbench_style_overall"]["l1_by_group"]
        print("[snapshot_metrics]", ",".join(payload["snapshot_lobbench_style_overall"]["metric_names"]))
        print(
            "[snapshot_lobbench6] W_mean={:.6g} W_median={:.6g} W_iqm={:.6g} | L1_mean={:.6g} L1_median={:.6g} L1_iqm={:.6g}".format(
                sw["mean"], sw["median"], sw["iqm"], sl1["mean"], sl1["median"], sl1["iqm"]
            )
        )


if __name__ == "__main__":
    main()

