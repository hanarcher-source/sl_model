# cluster_trackA — diary & agent handoff

Subproject under `stock_language_model/` for **Track A**: regime clustering on **hand-crafted per-window LOB stats** (no real-vs-gen comparison for features) → template captions → new preset anchor lines → train/eval with the **same** fixed-sampling replay pipeline as the main pool **0709→0710** experiments.

**Parent log:** `stock_language_model/DIARY.md` (full experiment history).

---

## Directory layout

| Path | Purpose |
|------|--------|
| `cluster_trackA/DIARY.md` | This file — framework + baselines + Track A proposal. |
| `cluster_trackA/scripts/` | **File copies** of canonical preprocess / train / inference / eval / aggregate scripts (see `scripts/README.md`). |
| `cluster_trackA/logs/` | Slurm or local logs for Track A–specific jobs. |
| `cluster_trackA/checkpoints/` | Optional: pointers or copies of models trained for Track A ablations. |
| `cluster_trackA/data/` | Cluster labels, per-window feature tables, generated `preset_anchors_clustered.txt`, etc. |
| `cluster_trackA/configs/` | YAML/JSON presets for clustering K, bucket definitions, eval manifests. |
| `cluster_trackA/scripts/slurm/` | **Slurm** batch files for long Track A jobs (see `scripts/slurm/README.md`). |

---

## Track A step 1 — train-window clustering (implemented)

**Script:** `cluster_trackA/scripts/build_train_window_clusters_track_a.py`

- Loads latest `final_result_for_merge_realflow_*_{day}_{STOCK}_*.joblib` from the pool preprocess dir.
- Builds **hand features** on each **train** window only (same 60/20/20 index split as `train_blankgpt2_openbidanchor_txncomplete_single_day.py`).
- Sweeps **K** (default 8–48 step 4) with **MiniBatchKMeans** (optional); records **inertia**, **silhouette** (subsampled), **Calinski–Harabasz**, **Davies–Bouldin**, cluster **balance** (min/max fraction, effective cluster count).
- Writes **`recommended_k`** = best silhouette among K with `min_cluster_frac` / `min_cluster_points` feasible; reports **`plateau_k_at_eps`** (parsimony: smallest K within `sil_eps` of best silhouette).

**Slurm (preferred for long runs):** from repo root,

- **One job, three stocks (sequential, one log file):**  
  `sbatch scripts/run_cluster_track_a_pool0709_train_onejob.sh`
- **Three parallel array tasks:**  
  `sbatch scripts/run_cluster_track_a_pool0709_train_array.sh`

Details and env overrides: **`cluster_trackA/scripts/slurm/README.md`**.

**Intra/inter only (saved npz, no re-cluster):**  
`python cluster_trackA/scripts/posthoc_intra_inter_from_npz.py --run-root cluster_trackA/data/cluster_runs/<RUN> --write-json`

---

## End-to-end framework (how pieces connect)

### 1. Data preprocessing (real LOB → joblib / bins)

- **Two-day pool (0709 train + 0710 eval layout):** `cluster_trackA/scripts/preprocess_real_lob_twoday_pool_openbidanchor_txncomplete.py` (copy of `scripts/hist_script/...`).
- **Single day:** `preprocess_real_lob_20250710_openbidanchor_txncomplete.py`.
- Output pattern (typical): under `saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete/` — `final_result_for_merge_realflow_openbidanchor_txncomplete_*_<STOCK>_*.joblib`, bin JSON alongside.

**Track A step 1:** hand-stats **per training window** → K sweep + labels; see **“Track A step 1”** above and `cluster_trackA/data/cluster_runs/…`.

### 2. Models & training

| Model line | Canonical script (also copied in `cluster_trackA/scripts/`) |
|------------|---------------------------------------------------------------|
| Blank GPT-2, open-bid-anchor head (pretrain-style baseline) | `train_blankgpt2_openbidanchor_txncomplete_single_day.py` |
| Blank GPT-2, dynamic anchor | `train_blankgpt2_dynamic_anchor_txncomplete_single_day.py` |
| Sentence preset S2IP (K anchors, cosine align) | `train_blankgpt2_sentence_preset_anchor_s2ip_txncomplete_single_day.py` |

Default pooled sentence runs (outside this folder) live under roots like `training_runs/pool_0709_0710_train0709_pretrained_gpt2_sentence_preset_s2ip_win50/`.

### 3. Inference + replay → stream for eval

- **Sentence / dynamic-anchor checkpoints:** `inference_replay_blankgpt2_dynamic_anchor_txncomplete_fixed_start.py`
- **Open-anchor / pretrain-style blank:** `inference_replay_blankgpt2_openbidanchor_txncomplete_fixed_start.py`

These drive fixed-start generation and token replay so downstream sees a consistent LOB stream.

### 4. Evaluator → `metrics_summary.json`

- **`eval_generated_stream.py`** — writes `metrics_summary.json` inside the experiment directory, including **`reference_comparison`** (real vs generated) and **`lobbench_style_overall`**.

### 5. Six aggregates + order counts (reporting contract)

From each `metrics_summary.json`:

| Field | Meaning |
|-------|--------|
| **`lobbench_style_overall.wasserstein`** | `mean`, `median`, `iqm` — pooled Wasserstein (prefer finite `weighted_wasserstein` per metric when present). |
| **`lobbench_style_overall.l1_by_group`** | `mean`, `median`, `iqm` for L1-by-group losses. |
| **`wasserstein.n` / `l1_by_group.n`** | Often labeled **`W_n`**, **`L1_n`**: count of per-metric terms in each pool — **if `W_n` differs between runs, aggregates are not over the same metric set.** |
| **`reference_comparison.real_reference_rows`** | **`n_real`** |
| **`reference_comparison.generated_rows`** | **`n_gen_rows`** (synthetic LOB rows in the comparison window) |

Recompute or verify aggregates without re-eval:

- **`compute_overall_scores_lobbench_style.py`** — same pooling rules as the JSON block.

**“Six aggregates”** in tables = **W_mean, W_median, W_iqm, L1_mean, L1_median, L1_iqm** (not six separate per-channel W1 scalars unless explicitly said).

### 6. Track A (planned — not implemented in code here yet)

1. Per-window hand-stats (train only) → standardize → **KMeans** (K ≤ anchor budget).
2. One **template sentence** per cluster (discrete buckets from cluster centroids).
3. Write **`cluster_trackA/data/preset_anchors_clustered.txt`**; embed like existing preset pipeline.
4. **Select / prune** clusters by **validation `lobbench_style_overall`**, same decode contract **`T10_k0_win50`** (`temperature=1.0`, `top_k=0`, sample on).

See §9 in `stock_language_model/DIARY.md` for the short proposal text.

---

## Baseline results to keep in view

### Pretrained blank vs sentence K3 — six-pack delta (**T10_k0 win50**, three stocks)

Same six aggregates; **Δ = sentence K3 − pretrained** (negative ⇒ sentence better on that scalar).

| stock | metric | pretrained | K=3 | Δ (K3−pre) |
|-------|--------|------------|-----|------------|
| 000617XSHE | W_mean | 0.429733 | 0.397449 | −0.032283 |
| | W_median | 0.356050 | 0.361496 | +0.005447 |
| | W_IQM | 0.386342 | 0.362537 | −0.023805 |
| | L1_mean | 0.396715 | 0.378481 | −0.018234 |
| | L1_median | 0.371181 | 0.319047 | −0.052134 |
| | L1_IQM | 0.388632 | 0.338571 | −0.050062 |
| 002263XSHE | W_mean | 0.787683 | 0.738345 | −0.049339 |
| | W_median | 0.468090 | 0.472834 | +0.004744 |
| | W_IQM | 0.591480 | 0.505315 | −0.086165 |
| | L1_mean | 0.529026 | 0.530308 | +0.001283 |
| | L1_median | 0.529426 | 0.513051 | −0.016374 |
| | L1_IQM | 0.553586 | 0.540633 | −0.012954 |
| 002721XSHE | W_mean | 0.619750 | 0.604384 | −0.015366 |
| | W_median | 0.547501 | 0.529778 | −0.017724 |
| | W_IQM | 0.582236 | 0.560906 | −0.021331 |
| | L1_mean | 0.402331 | 0.424000 | +0.021669 |
| | L1_median | 0.418656 | 0.432849 | +0.014194 |
| | L1_IQM | 0.387080 | 0.418497 | +0.031417 |

**Artifacts (metrics roots):**

- Pretrained: `saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/pretrained_T10_k0_win50/`
- Sentence K3: `saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/sentence_preset_s2ip_k3/sentence_s2ip_k3_T10_k0_win50/`

### Same three stocks — **`n_real`**, **`n_gen_rows`**, six-pack (+ `W_n`): pretrain vs sentence K3 only

| stock | model | n_real | n_gen_rows | W_n | W_mean | W_median | W_iqm | L1_n | L1_mean | L1_median | L1_iqm |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 000617XSHE | pretrain_baseline | 30961 | 23266 | 20 | 0.429733 | 0.356050 | 0.386342 | 20 | 0.396715 | 0.371181 | 0.388632 |
| 000617XSHE | sentence_k3 | 30961 | 31375 | 20 | 0.397449 | 0.361496 | 0.362537 | 20 | 0.378481 | 0.319047 | 0.338571 |
| 002263XSHE | pretrain_baseline | 16122 | 11659 | 18 | 0.787683 | 0.468090 | 0.591480 | 18 | 0.529026 | 0.529426 | 0.553586 |
| 002263XSHE | sentence_k3 | 16122 | 10622 | 19 | 0.738345 | 0.472834 | 0.505315 | 18 | 0.530308 | 0.513051 | 0.540633 |
| 002721XSHE | pretrain_baseline | 49846 | 38514 | 20 | 0.619750 | 0.547501 | 0.582236 | 20 | 0.402331 | 0.418656 | 0.387080 |
| 002721XSHE | sentence_k3 | 49846 | 42714 | 20 | 0.604384 | 0.529778 | 0.560906 | 20 | 0.424000 | 0.432849 | 0.418497 |

*(Unified-K ep3 row omitted here; see `stock_language_model/DIARY.md` §2026-04-12 for three-way + `W_n` caveat on 617.)*

---

## What a new agent should do first

1. Read **`stock_language_model/DIARY.md`** Quick reference + §2026-04-12.
2. Use **`cluster_trackA/scripts/README.md`** for the copy list and run instructions.
3. Implement Track A feature dump + clustering under **`cluster_trackA/data/`** (new scripts TBD).
4. Keep eval contract **frozen** when comparing captions: **`T10_k0_win50`**, log **`W_n`**.
