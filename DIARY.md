# Experiment diary (stock_language_model)

This file is a running log of notable experiments + key results/settings so we can reproduce runs later.

## Contents

1. **Quick reference** — six aggregates, fixed sampling, replay pairing, `hist_log` note  
2. **2026-04-05 / 06** — txn-complete **split-cancel** (code + headline Wasserstein table + readout)  
3. **2026-04-07** — dynamic vs open-anchor (blank GPT-2), pool 0709→0710  
4. **Cross-cutting** — window length, pretrained vs blank, sweep / general metric  
5. **2026-04-08** — sentence S2IP + fixed-sampling six-pack (blank vs k3 vs k5)  
6. **2026-04-10** — win50 three-way (blank vs K3 uncentered vs K3 centered)  
7. **2026-04-11** — K3 vs sepproj / low-λ; **pretrained vs K3 @ T10_k0** (3 stocks); `k3_regK5_v5match` 1-ep diag note  
8. **2026-04-12** — **unified-K ep3** vs sentence K3 (same fixed sampling); six-pack + counts; **`W_n` mismatch on 617**; three-way table incl. **pretrain baseline + `n_real`**  
9. **Proposal — Track A anchors** — cluster windows on **hand LOB stats** → **template captions** → **val `lobbench_style_overall`** to select/prune (not executed yet)  
10. **`cluster_trackA/` subproject** — layout, **file copies** of canonical scripts, agent handoff: `stock_language_model/cluster_trackA/DIARY.md`

---

## Quick reference

- **Six LOB-bench-style aggregates** (lower is better): **not** “six per-metric W1 distances.” They are **three** pooled **Wasserstein** summaries (**mean, median, IQM**) plus **three** pooled **L1_by_group** summaries (**mean, median, IQM**), each computed over every metric block in `reference_comparison.metrics` that contributes a finite loss (same rule as `compute_overall_scores_lobbench_style._collect_metric_losses`). Cached copy: `metrics_summary.json` → **`lobbench_style_overall`**. Implementation: `scripts/hist_script/compute_overall_scores_lobbench_style.py`.
- **Row counts in summaries:** `reference_comparison.real_reference_rows` = **`n_real`** (reference slice size); `reference_comparison.generated_rows` = synthetic rows in the same comparison window (often labeled **`n_gen_rows`** in tables). **`W_n` / `L1_n`** = how many per-metric terms entered each aggregate; if **`W_n` differs across runs**, mean/median/IQM are averaged over **different sets** of scoring functions (see 2026-04-12 note on 000617).
- **Fixed sampling (apples-to-apples LM eval):** `--sample` on, `--temperature 1.0`, `--top-k 0` — often logged under eval path tags like `T10_k0_win50`.
- **Replay vs clean ref (txn-complete):** compare **paired** dirs (same preprocess / same side vocabulary); split-cancel replay goes with **split-cancel** clean ref, not the old 5-side ref alone.
- **hist_log:** some archived Slurm text lives under `logs/hist_log/`; active wrappers usually write to `logs/<job_basename>.out`.

---

## 2026-04-05 / 2026-04-06 — Txn-complete split-cancel (replay vs clean ref)

### Motivation

- **Before:** txn-complete **5-side** cancel bucket (`Side` 99 → one cancel token); book layer disambiguates cancel vs bid/ask queues.
- **Split cancel (6-side):** use **`OrigSide`** so bid vs ask cancels become distinct tokens; vocab **26×26×12×6 = 48 672** (was **40 560**).

### Code / orchestration (high level)

- **`utility/sim_helper_unified.py`:** `split_cancel_sides` through preprocess + `apply_event_to_book_open_anchor_txn_complete(..., split_cancel_sides=...)`.
- **Preprocess:** `scripts/hist_script/preprocess_real_lob_20250710_openbidanchor_txncomplete.py` with `--split-cancel-sides`.
- **Clean ref + replay:** `generate_lobster_stream_real_openbidanchor_txncomplete_fixed_start.py` and `replay_decoded_real_tokens_openbidanchor_txncomplete_fixed_start.py` with the same flag.
- **Chained post-preprocess driver:** `scripts/hist_script/run_txncomplete_splitcancel_postprocess_pipeline.py` (generate → replay → `eval_generated_stream.py`); Slurm helpers under `scripts/hist_script/run_*splitcancel*`.
- **Dedicated eval logs (pattern):** `logs/eval_against_clean_refs/*_txncomplete_splitcancel_vs_cleanref.log` (vs older `*_txncomplete_vs_cleanref.log` for 5-side).

### Headline Wasserstein check (paired setups; lower = closer)

Old rows: **5-side txn-complete replay vs** clean ref `…openbidanchor_txncomplete_*_20260405_132652`.  
New rows: **split-cancel replay vs split-cancel clean ref** (new preprocess + ref). **Not** a single-factor cancel-only A/B (reference stream also changes).

| Ticker | Metric | Old 5-side vs clean ref | Split-cancel vs split-cancel ref | vs old |
|--------|--------|-------------------------|-----------------------------------|--------|
| 000617 | spread | 0.1956 | **0.1880** | Better |
| 000617 | log inter-arrival | 0.0381 | **0.0324** | Better |
| 000617 | OFI | 0.2454 | 0.2492 | Slightly worse |
| 000617 | orderbook imbalance | 1.2739 | 1.3802 | Worse |
| 002263 | spread | 0.01220 | 0.01217 | ~Flat |
| 002263 | log inter-arrival | 0.01524 | **0.01209** | Better |
| 002263 | OFI / ob_imb | 0.1065 / 0.4595 | 0.1069 / 0.4604 | ~Flat |
| 002366 | spread | 0.1081 | **0.1565** | Worse |
| 002366 | log inter-arrival | 0.0585 | **0.0457** | Better |
| 002366 | OFI / ob_imb | 0.2322 / 0.3261 | 0.2338 / 0.3352 | Slightly worse |
| 000981 | spread / log IAT / OFI / ob_imb | 0.01832 / 0.03311 / 0.2101 / 0.3099 | **Same at printed precision** | No change |

### Readout

- **No uniform win** on these headline metrics: mixed on **000617** and **002366**, small/no change on **002263**, **000981** effectively unchanged on the four reported numbers.
- For a **strict cancel-only** read you would need **fixed** clean ref (or matched preprocess) with cancel encoding toggled; this experiment does not isolate that.

---

## 2026-04-07 — Dynamic market anchor vs open-anchor (blank GPT-2), pooled 0709 train, 0710 eval

### Goal

Compare:

- **Baseline**: blank GPT-2 **no-anchor** head (open-bid-anchor / txn-complete tokenization), trained on pooled preprocess **day 20250709**, evaluated by fixed-start generation+replay+LOB metrics on **day 20250710**.
- **Dynamic market anchor**: blank GPT-2 with **dynamic anchor mixture** (V5-style), trained on the same pooled 0709 slice and evaluated on the same 0710 pipeline.

Both trained with the same schedule to keep comparisons fair:

- **max epochs**: 10
- **early stop patience**: 3
- **window_len**: 50
- **vocab_size**: 40560

### Code (train)

- **Open-anchor trainer (retrain schedule)**:
  - `scripts/train_blankgpt2_openbidanchor_txncomplete_single_day.py`
  - Added CLI `--patience` (default 3) and early stop uses it.
- **Dynamic-anchor trainer (new)**:
  - `scripts/train_blankgpt2_dynamic_anchor_txncomplete_single_day.py`
  - Anchor mechanism: softmax over `anchor_count` learned anchor vectors → weighted anchor token **prepended** to window embeddings.
  - Train objective: **CE + λ·margin_reg** (warmup on reg); **validation/early-stop uses CE only**.

### Data (train/eval)

Pooled preprocess root used for both regimes:

- `saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete/`

Evaluation day artifacts (per stock) are resolved from that folder:

- processed realflow: `final_result_for_merge_realflow_openbidanchor_txncomplete_20250710_<TAG>_*.joblib`
- bin record: `bin_record_realflow_openbidanchor_txncomplete_20250710_<TAG>_*.json`

Clean real reference (per stock):

- `saved_LOB_stream/fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_<TAG>_*`

### Outputs (train checkpoints)

Open-anchor (no-anchor head) retrain root:

- `training_runs/pool_0709_0710_train0709_blank_gpt2_win50_ep10pat3/<TAG>/model_cache/*_best.pt`

Dynamic-anchor retrain root:

- `training_runs/pool_0709_0710_train0709_blank_gpt2_dynamic_anchor_win50/<TAG>/model_cache/*_best.pt`

Example filenames observed on 2026-04-07:

- Open-anchor: `blankGPT2_20250709_txncomplete_<TAG>_win50_20260407_12433*_best.pt`
- Dynamic-anchor: `blankGPT2dyn_20250709_txncomplete_<TAG>_win50_20260407_12433*_best.pt`

### Slurm jobs — training (submitted earlier)

Open-anchor ep10/pat3 retrains:

- 000617: **54059**
- 000981: **54060**
- 002263: **54061**
- 002366: **54062**

Dynamic-anchor retrains:

- 000617: **54063**
- 000981: **54064**
- 002263: **54065**
- 002366: **54066**

### Paper-style “general metric” sampling settings (per stock)

We use the **LOB-Bench aggregate sweep winner** from:

- `logs/eval_pool0709_0710/sweep_best_lobbench_aggregate.json`

Winners (important correction for 000981):

- **000617XSHE**: `T13_k0` → temp=**1.3**, top_k=**0**
- **000981XSHE**: `T13_k400` → temp=**1.3**, top_k=**400**
  - Note: prior z-score pick was `T10_k50`, but the aggregate sweep winner is `T13_k400`.
- **002263XSHE**: `T13_k0` → temp=**1.3**, top_k=**0**
- **002366XSHE**: `T13_k0` → temp=**1.3**, top_k=**0**

### Code (generation stream + replay + eval)

Open-anchor generation+replay+eval entrypoint (existing):

- `scripts/hist_script/inference_replay_blankgpt2_openbidanchor_txncomplete_fixed_start.py`

Dynamic-anchor generation+replay+eval entrypoint (added):

- `scripts/hist_script/inference_replay_blankgpt2_dynamic_anchor_txncomplete_fixed_start.py`

Slurm sweep runners (auto-resolve newest `*_best.pt` in each training root):

- Open-anchor ep10/pat3: `scripts/hist_script/run_inference_blankgpt2_pool0709_0710_eval_0710_openanchor_ep10pat3_sweep_one.sh`
- Dynamic-anchor: `scripts/hist_script/run_inference_blankgpt2_pool0709_0710_eval_0710_dynamic_anchor_sweep_one.sh`

Eval outputs are written under:

- `saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/openanchor_ep10pat3/<SWEEP_TAG>/.../metrics_summary.json`
- `saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/dynamic_anchor/<SWEEP_TAG>/.../metrics_summary.json`

### Slurm jobs — generation+eval (LOB-Bench aggregate winners)

Submitted with settings above (temp/top_k per stock) for both regimes:

- Open-anchor ep10/pat3: **54099**, **54101**, **54103**, **54105**
- Dynamic-anchor: **54100**, **54102**, **54104**, **54106**

Logs:

- `logs/eval_pool0709_0710/run_inference_openanchor_ep10pat3_<SETTING>_win50_<TAG>.out`
- `logs/eval_pool0709_0710/run_inference_dynamic_anchor_<SETTING>_win50_<TAG>.out`

### Notes / caveats

- Dynamic-anchor training uses reg during train, but **val/test CE is evaluated on CE only** (no reg term) for early stopping / best checkpoint.

### Next round proposal (prioritize fixing hurting stocks)

We saw dynamic-anchor helps 002263 but hurts 000617/000981/002366 on the paper-style aggregate metric.
Next round focuses on reducing harmful conditioning and stabilizing routing.

#### Solution A: `dyn_gatebias` + gated margin reg

- **Injection**: no extra token; inject anchor as gated residual bias on embeddings:
  - `emb2 = emb + g * weighted_anchor`
- **Gate**: \(g = \sigma(W_g \cdot query)\)
- **Objective**: CE + warmup * (margin reg), but **margin reg is gated** (multiplied by `g`) so sharp routing is only enforced when anchor is used.

Diagnostics to log per epoch:

- Router entropy + Neff (effective #anchors)
- Top‑1 anchor usage (top‑5 anchors by count)
- Gate mean/std/min/max

#### Solution B: `dyn_attnpool_topk` + entropy/load-balance objective

- **Router**: attention pooling over time to build query (instead of mean pooling)
- **Anchor mixture**: sparse **top‑k** anchors (default k=4)
- **Objective**: CE + (entropy bonus + load-balance KL to uniform) on router probs (scheduled warmup)

Diagnostics to log per epoch:

- Router entropy + Neff
- Top‑1 anchor usage (top‑5)
- Pooling attention entropy (is router attending or uniform?)

Implementation:

- Trainer: `scripts/train_blankgpt2_dynamic_anchor_variants_txncomplete_single_day.py`

---

## Cross-cutting — window length + pretrained vs blank + “paper general metric”

### Window length (blank GPT-2, pooled 0709 train → 0710 eval)

Source table:

- `logs/eval_pool0709_0710/blank_window_lobbench_w_aggregate.csv`

Metric:

- LOB‑Bench-style aggregate Wasserstein (**W_mean**, lower is better), computed from `metrics_summary.json` → `reference_comparison.metrics`.

Observed best window per stock (blank):

- **000617XSHE**: best **win50** (W_mean ≈ 0.4375) vs win100/win200 worse
- **000981XSHE**: best **win50** (W_mean ≈ 0.7755); win200 was close but slightly worse; win100 much worse
- **002263XSHE**: best **win200** (W_mean ≈ 0.6702) vs win50/win100 worse
- **002366XSHE**: best **win100** (W_mean ≈ 0.4163) vs win50 worse and win200 much worse

Takeaway:

- Optimal **window length is stock-dependent** (not “win50 always best”).

### Pretrained vs blank (win50)

Source table:

- `logs/eval_pool0709_0710/blank_vs_pretrained_lobbench_w_mean_win50.csv`

Finding (win50, paper-style per-stock sampling):

- Pretrained was **worse than blank** on all 4 stocks (higher W_mean), with deltas:
  - 000617: +0.0653
  - 002263: +0.0580
  - 002366: +0.0189
  - 000981: +0.0803

### How we introduced the paper’s “general metric” + best sampling params per stock

We implemented the LOB‑Bench paper-style aggregate (“general”) metric as:

- For each `metrics_summary.json`, extract per-metric comparative losses from:
  - `reference_comparison.metrics.<metric_name>.weighted_wasserstein` when present/finite (conditional metrics)
  - else `reference_comparison.metrics.<metric_name>.wasserstein`
- Aggregate across metric configs with:
  - **mean**, **median**, and **IQM** (interquartile mean).

Code:

- `scripts/hist_script/compute_overall_scores_lobbench_style.py`
- Selection over sweep settings:
  - `scripts/hist_script/select_best_sweep_setting_lobbench_aggregate.py`
- Saved winners artifact:
  - `logs/eval_pool0709_0710/sweep_best_lobbench_aggregate.json`

Best sampling setting per stock (by **W_mean** criterion):

- **000617XSHE**: `T13_k0` → temp=1.3, top_k=0
- **000981XSHE**: `T13_k400` → temp=1.3, top_k=400  (important: differs from prior z-score pick `T10_k50`)
- **002263XSHE**: `T13_k0` → temp=1.3, top_k=0
- **002366XSHE**: `T13_k0` → temp=1.3, top_k=0

### Training budget sanity-check: ~3 epochs vs ep10/pat3 (open-anchor / no-anchor)

Question:

- Does increasing training to **max_epoch=10** with **patience=3** improve the paper-style aggregate metric vs the earlier shorter run (~max 3 epochs)?

Result (paper-style aggregate Wasserstein W_mean; lower is better), evaluated apples-to-apples with the same per-stock sampling winners:

- **000617XSHE (T13_k0)**: **0.437511 → 0.437511** (no change)
- **000981XSHE (T13_k400)**: **0.775456 → 0.775456** (no change)
- **002366XSHE (T13_k0)**: **0.442770 → 0.442770** (no change)
- **002263XSHE (T13_k0)**: **0.685905 → 0.786979** (worse)

Takeaway:

- Increasing epoch/patience **did not help much** in this setup (3/4 unchanged, 1/4 worse).

---

## 2026-04-08 — Sentence preset anchors (S2IP-style) + fixed sampling regime

### Fixed sampling policy (“rolling weighted dice”)

Use multinomial sampling from the model distribution (same policy for every model/stock):

- `--sample` **on**
- `--temperature 1.0`
- `--top-k 0`

### Fixed-sampling eval results (6 aggregates; lower is better)

Aggregates are over `reference_comparison.metrics` (Wasserstein and L1_by_group; each summarized by mean/median/IQM).

```
Fixed sampling: sample=on, temp=1.0, topk=0
stock      variant       W_mean   W_median  W_IQM    L1_mean  L1_median L1_IQM
000617XSHE blank         0.3750   0.3349    0.3381   0.3729   0.3698    0.3509
000617XSHE sentence_k3   0.3974   0.3615    0.3625   0.3785   0.3190    0.3386
000617XSHE sentence_k5   0.4374   0.3689    0.3914   0.4054   0.3482    0.3640

000981XSHE blank         0.8910   0.8709    0.9280   0.6078   0.6929    0.6459
000981XSHE sentence_k3   0.8087   0.6981    0.7735   0.5481   0.5290    0.5300
000981XSHE sentence_k5   0.8581   0.8024    0.8620   0.5705   0.5852    0.5827

002263XSHE blank         0.8492   0.7335    0.6545   0.5825   0.6932    0.6090
002263XSHE sentence_k3   0.7383   0.4728    0.5053   0.5303   0.5131    0.5406
002263XSHE sentence_k5   0.8073   0.6542    0.6068   0.5359   0.5409    0.5640

002366XSHE blank         0.5672   0.4477    0.4460   0.3743   0.3386    0.3405
002366XSHE sentence_k3   0.6819   0.6691    0.6507   0.5278   0.5301    0.5563
002366XSHE sentence_k5   0.6430   0.5699    0.5836   0.5041   0.5322    0.5253
```

Takeaways:

- Under fixed sampling, **K=3 > K=5** for the sentence model across all 4 stocks.
- Sentence_k3 is clearly better on 000981 and 002263, worse on 002366, and mixed on 000617.

### Next round: sentence-anchor A/B training variants (submit 8 jobs)

Train on pooled 0709 (win50, epochs=3), K=3:

- **Variant A**: frozen anchors + **centered** anchor bank + **separate** projections (q_proj/a_proj)
- **Variant B**: **trainable** anchors + centered + separate projections

Slurm job IDs:

- Variant A: **54178–54181** (000617/000981/002263/002366)
- Variant B: **54182–54185** (000617/000981/002263/002366)

### Note: next improvements to try (selection collapse)

If sentence-anchor retrieval collapses (same few anchors dominate), the alignment-only objective can saturate without improving replay metrics.
Next plan is to **re-run the clean baseline (no Variant A/B toggles)** but with **fuller router diagnostics**, to confirm whether collapse/weak-separation is real and when it appears. Only after we can *see* the failure mode in logs do we add anti-collapse penalties.

#### Phase 1 — Full router diagnostics in logs (no behavior change)

Add a standardized `router_health` block (print + optionally write JSONL) every N steps and once per epoch.

Diagnostics dimensions (what to log):

- **Router distribution / usage**
  - top‑1 anchor id per sample (and batch histogram; log top‑5 anchors by count)
  - top‑1 unique count per epoch (how many anchors ever win)
  - diagnostic softmax over scores (even if training uses hard top‑K prepend): mean entropy, and **Neff** (effective # anchors)
  - optional: HHI/Gini over usage as a single collapse scalar

- **Separation / confidence**
  - gap stats: \(s_1-s_2\), and \(s_K-s_{K+1}\) (mean/median/p10)
  - diagnostic “top‑K probability mass” under softmax(scores/τ): \(\sum_{i\in topK} p_i\)
  - raw score scale (mean/std of cosine scores; tells whether scores are flat)

- **Anchor bank geometry** (important if anchors are trainable, still useful if frozen)
  - anchor norm distribution (mean/std/min/max)
  - off-diagonal cosine stats (mean/max pairwise cos) to detect near-duplicates / anisotropy

- **Is prefix actually used? (sanity probe)**
  - once per epoch: evaluate a tiny batch with normal prefix vs shuffled prefix and log ΔCE (if ~0, prefix may be ignored)

- **If anchors are trainable**
  - anchor drift \(\|A_t-A_{t0}\|\) (mean/max)
  - grad norms on anchor bank and projection layers

Goal of Phase 1:

- Determine whether “bad” runs correlate with **collapse** (low Neff / low unique anchors / high HHI) or with **weak separation** (small \(s_K-s_{K+1}\)), vs “prefix ignored” (ΔCE ~0).

#### Phase 2 — Add explicit anti-collapse / separation penalties (small + targeted)

Once Phase 1 confirms the failure mode, add one or two of:

- **Separation (SEP) margin** (closest to `v5_1st_stock.py`):
  - enforce \(s^{(K)} - s^{(K+1)} \ge m\) via ReLU penalty
  - \(L = CE + \lambda_{align} L_{align} + \lambda_{sep} L_{sep}\)

- **Load-balance penalty** (MoE-style) on top‑1 usage:
  - encourage higher entropy / non-degenerate usage across anchors within a batch/epoch

- **Anchor repulsion** (only if anchors trainable):
  - penalize high pairwise cosine between anchors (avoid duplicates)

#### Phase 3 — Tuning protocol (keep eval fair)

- Keep evaluation decoding fixed (“rolling weighted dice”):
  - `--sample --temperature 1.0 --top-k 0`
- Run short sweeps (≤3 epochs) over only 1–2 knobs first:
  - \(\lambda_{sep}\) and/or margin \(m\)
  - \(\lambda_{lb}\) (load-balance)
- Use Phase 1 diagnostics to select ranges:
  - low Neff / collapse → increase \(\lambda_{lb}\)
  - small \(s_K-s_{K+1}\) → increase \(\lambda_{sep}\) or margin \(m\)
  - ΔCE ~0 (prefix unused) → revisit injection strength / align weight last

We will return to Phase 1–3 after finishing the window-length ablation evals.

---

## 2026-04-10 — Win50 three-way table: blank vs sentence K3 uncentered vs sentence K3 centered (0710 eval, fixed sampling)

### Protocol (unchanged from fixed-sampling regime)

- **Decode:** `--sample` on, `--temperature 1.0`, `--top-k 0` (same for all rows).
- **Train:** pooled preprocess day **20250709**, **`window_len=50`**, **`epochs_max=3`** (blank uses `best.pt` by val CE within that cap; sentence models likewise).
- **Eval:** day **20250710** fixed-start generation → replay → LOB comparative metrics.
- **Six aggregates:** mean / median / IQM over all comparative Wasserstein and L1-by-group metrics in `reference_comparison.metrics` (same pooling as `scripts/hist_script/compute_overall_scores_lobbench_style.py`). **Lower is better.**

### Next options inspired by S2IP-LLM paper (to try later)

Paper (S2IP-LLM / S2IIP): derive anchors from **pretrained token embeddings**, retrieve **top-K** by cosine to TS embedding, **prepend** anchors as prefix prompts, and add an **alignment reward** term in the objective; freeze most of GPT-2 and fine-tune only small subsets (positional embeddings + layer norms + mapping heads).

Planned options for our LOB setup (from most paper-faithful to more expressive):

1. **Anchors from GPT-2 token embeddings (not sentences)**: build anchor bank from the GPT-2 word/token embedding table (or a reduced subset), retrieve top‑K by cosine to our LOB query, and prepend as prefix prompts.
2. **Learnable mapping \(f(\cdot)\) on token-embedding anchors**: add a small linear/MLP mapping from selected token embeddings into the joint/router space; train with our CE − λ·align objective.
3. **Paper-style freezing**: keep most GPT‑2 frozen and only fine-tune positional embeddings + layer norms (plus router/proj/head), to isolate prompt effects and reduce instability.
4. **Query-dependent readout (cross-attention) + gate**: LOB query attends over unpooled sentence/token representations and a gate blends anchor branch with a blank-like bypass. (Not in the paper; a “beyond S2IP” extension to address pooling/interaction limits.)

### Checkpoint / eval artifact provenance

| Row | Training root | Notes |
|-----|---------------|--------|
| **blank** | `training_runs/pool_0709_0710_train0709_blank_gpt2/<TAG>/model_cache/*_win50_20260406_132212_best.pt` | **Not** the ep10/pat3 blank retrain; this is **max 3 epochs**. |
| **sentence K3 uncentered** | `training_runs/pool_0709_0710_train0709_sentence_preset_s2ip_win50_k3/<TAG>/...` | Full 3-epoch runs (`run_train_pool0709_sentence_s2ip_k3_ep3_*.sh`). Same trainer recipe as phase1diag **without** router-diag-only logging. **Do not** use unfinished `..._phase1diag_*` jobs for this row. |
| **sentence K3 centered** | `training_runs/pool_0709_0710_train0709_sentence_k3_baseline_win50_ep3_center_phase1diag/<TAG>/...` | `--center-anchors`; completed 3 epochs in logs. |

**`metrics_summary.json` paths used for the table below**

- Blank: `saved_LOB_stream/pool_0709_0710_eval_0710/fixed_start_model_blankgpt2_tokens_openbidanchor_txncomplete_<TAG>_20260406_182708|182710/`
- K3 uncentered: `saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/sentence_preset_s2ip_k3/sentence_s2ip_k3_T10_k0_win50/..._20260408_165408|165402/`
- K3 centered: `saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/sentence_k3_center_win50/sentence_k3_center_win50_T10_k0_ep3/..._20260410_124013/`

Machine-readable copy (regenerate by re-running the aggregation over those files):

- `saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/comparison_blank_k3unc_k3ctr_win50_T10_k0.json`

### Results (six aggregates)

```
stock      variant                    W_mean   W_median  W_IQM    L1_mean  L1_median L1_IQM
000617XSHE blank                      0.3750   0.3349    0.3381   0.3729   0.3698    0.3509
000617XSHE sentence K3 uncentered    0.3974   0.3615    0.3625   0.3785   0.3190    0.3386
000617XSHE sentence K3 centered      0.4709   0.4512    0.4466   0.4010   0.3587    0.3696

000981XSHE blank                      0.8910   0.8709    0.9280   0.6078   0.6929    0.6459
000981XSHE sentence K3 uncentered    0.8087   0.6981    0.7735   0.5481   0.5290    0.5300
000981XSHE sentence K3 centered      0.8655   0.8372    0.8563   0.5759   0.7137    0.6111

002263XSHE blank                      0.8492   0.7335    0.6545   0.5825   0.6932    0.6090
002263XSHE sentence K3 uncentered    0.7383   0.4728    0.5053   0.5303   0.5131    0.5406
002263XSHE sentence K3 centered      0.7865   0.5523    0.5737   0.5211   0.5606    0.5511

002366XSHE blank                      0.5672   0.4477    0.4460   0.3743   0.3386    0.3405
002366XSHE sentence K3 uncentered    0.6819   0.6691    0.6507   0.5278   0.5301    0.5563
002366XSHE sentence K3 centered      0.6937   0.6877    0.6510   0.5564   0.5830    0.5689
```

### Short read

- **Uncentered K3** still beats blank on **981** and **263** on these six scores; **366** sentence is worse than blank; **617** mixed.
- **Centering** improves router geometry in diagnostics but **does not** improve these six aggregates vs uncentered on this table (617 W much worse centered; 981/263/366 mostly worse or mixed on W/L1 aggregates).

---

## 2026-04-11 — Sentence K3 vs sep proj / low-align ablations; pretrained vs K3 (T10_k0)

### Sentence K3 vs `sepproj` vs low `λ_align` (1e-2), fixed T10_k0 win50

Six-pack from each run’s `metrics_summary.json` (same pooling as `compute_overall_scores_lobbench_style.py`). **Lower is better.** Baseline = original sentence K3 evals.

| stock | variant | W_mean | W_med | W_iqm | L1_mean | L1_med | L1_iqm | gen_rows |
|--------|---------|--------|-------|-------|---------|--------|--------|----------|
| **000617** | baseline k3 | **0.397** | **0.362** | **0.363** | **0.379** | **0.319** | **0.339** | 31375 |
| | separate proj | 0.449 | 0.395 | 0.420 | 0.414 | 0.395 | 0.381 | 26498 |
| | low λ_align 1e-2 | 0.436 | 0.395 | 0.411 | 0.381 | 0.324 | 0.353 | 32257 |
| **002263** | baseline k3 | **0.738** | **0.473** | **0.505** | **0.530** | **0.513** | **0.541** | 10622 |
| | separate proj | 0.748 | 0.485 | 0.546 | 0.547 | 0.534 | 0.569 | 10914 |
| | low λ_align 1e-2 | 0.755 | 0.469 | 0.523 | 0.552 | 0.543 | 0.569 | 10982 |
| **002721** | baseline k3 | 0.604 | 0.530 | 0.561 | 0.424 | 0.433 | 0.419 | 42714 |
| | separate proj | 0.600 | 0.558 | 0.569 | **0.408** | **0.401** | **0.386** | 40153 |
| | low λ_align 1e-2 | **0.521** | **0.467** | **0.493** | **0.406** | **0.407** | **0.395** | 43759 |

**Paths**

- Baseline: `saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/sentence_preset_s2ip_k3/sentence_s2ip_k3_T10_k0_win50/`
- Sep proj: `.../sentence_preset_s2ip_k3_sepproj/sentence_s2ip_k3_sepproj_T10_k0_win50/..._20260411_140221/`
- Low align: `.../sentence_preset_s2ip_k3_lowalign1e2/sentence_s2ip_k3_lowalign1e2_T10_k0_win50/..._20260411_140221/`

**Takeaways:** On **000617** and **002263**, baseline K3 wins all six vs both ablations. On **002721**, **low λ_align (1e-2) wins all six** vs baseline; sep proj beats baseline on L1 aggregates but not on W_med / W_iqm. **`generated_rows` differs** across runs (sampling + dynamics)—treat small gaps as noisy.

### Router diagnostics (same session, qualitative)

- **Diagnostic softmax** over 128 anchors stayed **near-uniform** (`neff_softmax ≈ 128`, entropy ≈ ln(128)) for sep proj and low-align runs.
- **Separate projections:** on **000617**, **hard top-1 collapse** worsened (`top1_unique` ~48 → ~7, **HHI** up) vs low-align keeping more distinct top-1 IDs (~60s).
- **Cosine gaps:** `gap_1_2` shrank over training (weaker separation); **`align_mean`↑** with **top-K cosines nearly tied** — high alignment with **flat** full-bank scores.

### Pretrained GPT-2 vs sentence K3 (blank + sentence recipe), **T10_k0**, three stocks only

Same six aggregates, **Δ = K3 − pretrained** (negative ⇒ sentence K3 better).

**Artifacts**

- Pretrained: `saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/pretrained_T10_k0_win50/`
- Sentence K3: `.../sentence_preset_s2ip_k3/sentence_s2ip_k3_T10_k0_win50/`

| stock | metric | pretrained | K=3 | Δ (K3−pre) |
|--------|--------|------------|-----|------------|
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

**Counts (finite per-metric terms):** mostly **20 W + 20 L1** each side; **002263** K=3 has **19** W vs **18** for pretrained (minor set mismatch).

**Readout:** **617** — K3 wins 5/6. **263** — K3 wins 4/6. **721** — K3 wins all **W** aggregates; pretrained wins all **L1** aggregates (W vs L1 tradeoff).

### One-epoch diagnostic: `k3_regK5_v5match` (routing K=3, V5 align/sep at K_reg=5)

- **Slurm / trainer:** `scripts/run_train_pool0709_k3_regK5_v5match_1ep_diag_*.sh` → `scripts/train_blankgpt2_sentence_preset_anchor_s2ip_txncomplete_single_day_variant_margin_sep_k3_regk5_v5match.py`.
- **Submitted (example IDs):** 54338–54340 (617 / 263 / 721).
- **End CE (stdout):** val/test best ≈ **6.56 / 7.37** (617), **7.19 / 6.98** (263), **10.62 / 11.49** (721).
- **`sep_logs.margin_k_mean` (K=5 vs 6):** rose over the epoch (~**0.01 → 0.12–0.18** across names) — hinge on **5 vs 6** opens as trained.
- **`router.gap_k_k1` (K=3 vs 4):** stayed **small** (~**0.008–0.017**) — **prefix** boundary not forced by this objective.
- **Takeaway:** mixed **K_reg** vs routing **K** shows **wide 5–6 margin** but **not** the same story on **3–4**; for “does K=3 separation improve?” compare runs where **align+sep use the same K as the prefix cut**.

---

## 2026-04-12 — Unified-K (3-epoch, pretrained-together) vs sentence K3; counts; `W_n` trap

### What we compared

- **Decode / eval protocol:** fixed sampling **`T10_k0_win50`** (same as prior sentence K3 rows): `--sample` on, **`temperature=1.0`**, **`top_k=0`**.
- **Sentence K3 (per-stock):** `sentence_preset_s2ip_k3/sentence_s2ip_k3_T10_k0_win50/` (dynamic-anchor txn-complete checkpoints; eval dirs dated **2026-04-08** / **2026-04-10** as in §2026-04-10 / §2026-04-11).
- **Unified-K ep3 (“pretrained together”):** `sentence_preset_s2ip_k3_v5style_unifiedK_ep3/sentence_s2ip_k3_v5style_unifiedK_T10_k0_win50/` (eval dirs **`20260412_100909`**).
- **Pretrain baseline (blank GPT-2, pooled pretrain checkpoint):** `pretrained_T10_k0_win50/` — **`openbidanchor`** paths in dirname vs **`dynamic_anchor`** for sentence rows; **`n_real` still matches** per stock vs sentence/unified runs below (same reference row count).

### Terminology fix (reporting)

When we say **“six aggregates,”** we mean **`lobbench_style_overall`** (or equivalent recomputation): **W_mean, W_median, W_iqm, L1_mean, L1_median, L1_iqm** — plus optional **`W_n` / `L1_n`** counting how many `reference_comparison.metrics` entries had finite scores. We are **not** reporting six separate scalar channels (e.g. spread / OFI / …) unless explicitly labeled as per-metric.

### Why **`W_n` = 20** on **000617** sentence K3 but **`W_n` = 22** on unifiedK ep3

Aggregation prefers **`weighted_wasserstein`** when finite, else top-level **`wasserstein`** (`compute_overall_scores_lobbench_style._collect_metric_losses`). On **000617**, **two conditional blocks** differ between runs:

- **`ask_volume | spread`**
- **`spread | time`**

For **sentence K3**, both have **`weighted_wasserstein: null`** (some `per_group` entries null → no finite pooled W), **and** no top-level `wasserstein`, so they are **omitted** → **20** terms.

For **unifiedK ep3**, both have **finite `weighted_wasserstein`**, so they **enter** the pool → **22** terms.

**Implication:** headline **W_mean / median / IQM** on 617 are **not** strictly apples-to-apples unless you re-aggregate on a **fixed intersection** of metric names or re-run eval so conditional weighted W is defined for both.

### Three stocks — **pretrain baseline**, sentence K3, unifiedK ep3 (`n_real`, **`generated_rows`**, six-pack + `W_n` / `L1_n`)

**`n_real`** = `reference_comparison.real_reference_rows`. **`n_gen_rows`** = `reference_comparison.generated_rows`.

#### 000617XSHE (`n_real` = 30961)

| model | n_real | n_gen_rows | W_n | W_mean | W_median | W_iqm | L1_n | L1_mean | L1_median | L1_iqm |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pretrain_baseline | 30961 | 23266 | 20 | 0.429733 | 0.356050 | 0.386342 | 20 | 0.396715 | 0.371181 | 0.388632 |
| sentence_k3 | 30961 | 31375 | 20 | 0.397449 | 0.361496 | 0.362537 | 20 | 0.378481 | 0.319047 | 0.338571 |
| unifiedK_ep3 | 30961 | 27712 | 22 | 0.396400 | 0.346412 | 0.349201 | 20 | 0.367772 | 0.317865 | 0.342683 |

#### 002263XSHE (`n_real` = 16122)

| model | n_real | n_gen_rows | W_n | W_mean | W_median | W_iqm | L1_n | L1_mean | L1_median | L1_iqm |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pretrain_baseline | 16122 | 11659 | 18 | 0.787683 | 0.468090 | 0.591480 | 18 | 0.529026 | 0.529426 | 0.553586 |
| sentence_k3 | 16122 | 10622 | 19 | 0.738345 | 0.472834 | 0.505315 | 18 | 0.530308 | 0.513051 | 0.540633 |
| unifiedK_ep3 | 16122 | 11018 | 19 | 0.769799 | 0.512722 | 0.564223 | 18 | 0.549275 | 0.538011 | 0.589375 |

#### 002721XSHE (`n_real` = 49846)

| model | n_real | n_gen_rows | W_n | W_mean | W_median | W_iqm | L1_n | L1_mean | L1_median | L1_iqm |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pretrain_baseline | 49846 | 38514 | 20 | 0.619750 | 0.547501 | 0.582236 | 20 | 0.402331 | 0.418656 | 0.387080 |
| sentence_k3 | 49846 | 42714 | 20 | 0.604384 | 0.529778 | 0.560906 | 20 | 0.424000 | 0.432849 | 0.418497 |
| unifiedK_ep3 | 49846 | 41398 | 20 | 0.622995 | 0.660348 | 0.604688 | 20 | 0.426127 | 0.401942 | 0.424294 |

### Short read (this slice only)

- **617:** unifiedK ep3 is **best on most W/L1 aggregates** vs sentence K3 **and** pretrain, but **inflate `W_n` caveat** above.
- **263:** sentence K3 **beats** unifiedK ep3 on **all six** here; pretrain is worst on W aggregates.
- **721:** sentence K3 **best W**; pretrain **best L1** on mean/median; unifiedK mixed (better W_mean than pretrain, worse W_median/W_iqm than sentence K3).

### `metrics_summary.json` paths (exact dirs used for the numbers above)

- **Pretrain:** `saved_LOB_stream/pool_0709_0710_eval_0710_model_variants/pretrained_T10_k0_win50/fixed_start_model_blankgpt2_tokens_openbidanchor_txncomplete_<TAG>_20260410_195210` (002721 uses **`..._220603`**).
- **Sentence K3:** `.../sentence_preset_s2ip_k3/sentence_s2ip_k3_T10_k0_win50/fixed_start_model_blankgpt2_tokens_dynamic_anchor_txncomplete_<TAG>_20260408_165408|165402` and **`002721XSHE_20260410_235917`**.
- **UnifiedK ep3:** `.../sentence_preset_s2ip_k3_v5style_unifiedK_ep3/sentence_s2ip_k3_v5style_unifiedK_T10_k0_win50/fixed_start_model_blankgpt2_tokens_dynamic_anchor_txncomplete_<TAG>_20260412_100909/`.

---

## Proposal (not yet run): Track A — data‑derived sentence anchors for replay aggregates

**Goal:** improve **`lobbench_style_overall`** (replay vs real ref), not CE as the sole objective.

**Idea:** cluster training **windows** in a **hand‑crafted LOB feature space** (spread / imbalance / activity / simple depth or return / session bin — 10–40 scalars from preprocess or decoded state; winsorize + standardize on train; optional PCA). **KMeans** (or minibatch) with **K ≤ anchor_count**; reject degenerate splits (one giant cluster).

**Captions:** one **deterministic template sentence** per cluster from **discrete buckets** (quartiles/tertiles on train) of a few readable dims — not free‑form prose in v0.

**Anchors:** write lines to a new preset file; embed **exactly like today** (preset txt → GPT‑2 last‑layer mean hidden).

**Selection:** **time‑split val**; **greedy add / optional prune** of clusters using **marginal gain on val replay aggregate** (same eval contract as prod: fixed **`T10_k0_win50`**, watch **`W_n`**). Val CE only as a **cheap pre‑filter** if needed.

**Why Track A:** benchmark losses pool over **observable LOB marginals**; clustering in that space targets **regimes** those metrics already care about, vs clustering in LM hidden space (looser link to Wasserstein/L1 terms).

