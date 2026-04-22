#!/usr/bin/env python3
"""
Re-run eval_generated_stream on existing experiment dirs so eval.log + metrics_summary.json
get lobbench_style_overall (and refreshed comparative metrics).

Resolves --real_ref_dir from each exp_dir/generation_notes.json when using --lbmean-root.
Optionally re-evaluates blank-win50 sweep dirs for the LOB-Bench-mean sampling settings.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import List, Tuple


def _latest_dir(pattern: str) -> str:
    hits = glob.glob(pattern)
    if not hits:
        raise FileNotFoundError(f"No match: {pattern}")
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]


def _ref_from_notes(exp_dir: str) -> str:
    notes_path = os.path.join(exp_dir, "generation_notes.json")
    with open(notes_path, "r", encoding="utf-8") as f:
        notes = json.load(f)
    ref = (notes.get("paths") or {}).get("real_ref_dir")
    if not ref or not os.path.isdir(ref):
        raise FileNotFoundError(f"Bad real_ref_dir in {notes_path}: {ref}")
    return ref


def collect_lbmean_pairs(lbmean_root: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(lbmean_root):
        return out
    for sub in sorted(os.listdir(lbmean_root)):
        if not sub.startswith("lbmean_"):
            continue
        root = os.path.join(lbmean_root, sub)
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            exp_dir = os.path.join(root, name)
            if not os.path.isdir(exp_dir):
                continue
            if not os.path.isfile(os.path.join(exp_dir, "generation_notes.json")):
                continue
            out.append((exp_dir, _ref_from_notes(exp_dir)))
    return out


def collect_pretrained_win50_lbmean_pairs(model_variants_root: str) -> List[Tuple[str, str]]:
    """
    Pretrained backbone win50 runs aligned with LOB-Bench-mean sampling:
      - 000617 / 002263 / 002366: best_T13_k0_pretrained_win50
      - 000981: lbmean_T13_k400_pretrained_win50 (after inference job creates it)
    """
    out: List[Tuple[str, str]] = []
    for sub in ("best_T13_k0_pretrained_win50", "lbmean_T13_k400_pretrained_win50"):
        root = os.path.join(model_variants_root, sub)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            exp_dir = os.path.join(root, name)
            if not os.path.isdir(exp_dir):
                continue
            notes_path = os.path.join(exp_dir, "generation_notes.json")
            if not os.path.isfile(notes_path):
                continue
            out.append((exp_dir, _ref_from_notes(exp_dir)))
    return out


def collect_sweep_win50_pairs(sweep_root: str) -> List[Tuple[str, str]]:
    """Blank win50 comparative runs: same LOB-Bench-mean params as lbmean batch."""
    specs = [
        ("000617XSHE", "T13_k0"),
        ("002263XSHE", "T13_k0"),
        ("002366XSHE", "T13_k0"),
        ("000981XSHE", "T13_k400"),
    ]
    out: List[Tuple[str, str]] = []
    for tag, setting in specs:
        pat = os.path.join(sweep_root, setting, f"fixed_start_model_blankgpt2_tokens_openbidanchor_txncomplete_{tag}_*")
        exp_dir = _latest_dir(pat)
        out.append((exp_dir, _ref_from_notes(exp_dir)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/finance_ML/zhanghaohan/stock_language_model")
    ap.add_argument(
        "--lbmean-root",
        default="saved_LOB_stream/pool_0709_0710_eval_0710_model_variants",
        help="Directory containing lbmean_* experiment subfolders.",
    )
    ap.add_argument(
        "--sweep-root",
        default="saved_LOB_stream/pool_0709_0710_eval_0710_sweep",
        help="Sweep root for --include-sweep-win50.",
    )
    ap.add_argument(
        "--include-sweep-win50",
        action="store_true",
        help="Also re-evaluate latest blank-GPT2 win50 runs under T13_k0 / T13_k400 per stock.",
    )
    ap.add_argument(
        "--include-pretrained-win50-lbmean",
        action="store_true",
        help=(
            "Re-eval pretrained win50 dirs: best_T13_k0_pretrained_win50 (3 stocks) and "
            "lbmean_T13_k400_pretrained_win50 (981, if present)."
        ),
    )
    ap.add_argument(
        "--only-pretrained-win50-lbmean",
        action="store_true",
        help="Only re-eval pretrained win50 LOB-Bench-mean dirs (no lbmean blank / no sweep).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    eval_py = os.path.join(root, "scripts", "hist_script", "eval_generated_stream.py")
    if not os.path.isfile(eval_py):
        raise SystemExit(f"Missing {eval_py}")

    pairs: List[Tuple[str, str]] = []
    lb_root = os.path.join(root, args.lbmean_root)
    if args.only_pretrained_win50_lbmean:
        pairs.extend(collect_pretrained_win50_lbmean_pairs(lb_root))
    else:
        pairs.extend(collect_lbmean_pairs(lb_root))
        if args.include_sweep_win50:
            sw_root = os.path.join(root, args.sweep_root)
            pairs.extend(collect_sweep_win50_pairs(sw_root))
        if args.include_pretrained_win50_lbmean:
            pairs.extend(collect_pretrained_win50_lbmean_pairs(lb_root))

    if not pairs:
        raise SystemExit("No experiment directories found.")

    print(f"Planned re-evaluations: {len(pairs)}", flush=True)
    for exp_dir, ref_dir in pairs:
        print(f"  EXP={exp_dir}", flush=True)
        print(f"  REF={ref_dir}", flush=True)
        if args.dry_run:
            continue
        cmd = [
            sys.executable,
            "-u",
            eval_py,
            os.path.abspath(exp_dir),
            "--real_ref_dir",
            os.path.abspath(ref_dir),
        ]
        print(f"  CMD: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=root, check=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
