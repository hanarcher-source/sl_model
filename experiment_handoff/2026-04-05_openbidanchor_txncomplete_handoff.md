# Experiment Handoff Log — 2026-04-05

This note is a dated handoff for the open-bid-anchor replay and clean-reference reconstruction work under `stock_language_model`. It is intended to let another agent resume work without re-deriving directory purpose, script roles, or the main conclusions.

## Current High-Level Status

As of 2026-04-05, the active comparison setup is:

- Old open-anchor baseline replay
- Old open-anchor replay with exact real price and exact real quantity (`exactPQ`)
- Open-anchor replay with explicit txn-complete event semantics (`txncomplete`)
- Clean regenerated realflow reference built from txn-complete processed rows

The old broken realflow reference path has been archived and should not be used as the benchmark.

## Folder Map

### Active top-level folders

- `saved_LOB_stream/`
  - active experiment outputs
  - active clean reference outputs
  - processed realflow inputs
- `logs/eval_against_clean_refs/`
  - completed SLURM eval logs for replay-vs-clean-reference comparisons
- `scripts/hist_script/`
  - active preprocess, replay, clean-reference generation, and SLURM launcher scripts
- `experiment_handoff/`
  - this handoff documentation folder

### Archived folders

- `saved_LOB_stream/archive_bugged_reconstruction/`
  - old broken reference exports and one incomplete superseded clean-generator foreground run
- `saved_LOB_stream/archive_intermediate_experiments/`
  - sampling sweeps, non-open-anchor outputs, exact-price-only outputs, duplicate or superseded outputs
- `logs/archive_bugged_reconstruction/`
  - logs tied to the broken reference path
- `logs/archive_intermediate_experiments/`
  - intermediary replay, preprocess, generation, sweep, and historical log files
- `scripts/hist_script/archive_bugged_reconstruction/`
  - old broken reference generator script and launcher
- `scripts/hist_script/archive_intermediate_experiments/`
  - scripts and launchers not needed for the active workflow

## Active Output Directories

### Old open-anchor binning baseline

Purpose:
- baseline replay using the old open-anchor representation without txn-complete semantics

Directories:
- `saved_LOB_stream/fixed_start_decoded_real_tokens_openbidanchor_000617XSHE_20260404_212227`
- `saved_LOB_stream/fixed_start_decoded_real_tokens_openbidanchor_000981XSHE_20260404_212230`
- `saved_LOB_stream/fixed_start_decoded_real_tokens_openbidanchor_002263XSHE_20260404_212230`
- `saved_LOB_stream/fixed_start_decoded_real_tokens_openbidanchor_002366XSHE_20260404_212230`

### Exact price + exact quantity replay (`exactPQ`)

Purpose:
- replay old event semantics
- keep decoded event regime
- override decoded price with exact real `Price`
- override decoded qty with exact real `OrderQty`

Directories:
- `saved_LOB_stream/fixed_start_decoded_real_tokens_openbidanchor_exactprice_exactqty_000617XSHE_20260405_123424`
- `saved_LOB_stream/fixed_start_decoded_real_tokens_openbidanchor_exactprice_exactqty_000981XSHE_20260405_123424`
- `saved_LOB_stream/fixed_start_decoded_real_tokens_openbidanchor_exactprice_exactqty_002263XSHE_20260405_123424`
- `saved_LOB_stream/fixed_start_decoded_real_tokens_openbidanchor_exactprice_exactqty_002366XSHE_20260405_123424`

### Txn-complete replay

Purpose:
- replay with explicit txn-complete event semantics
- still decode within the saved binning regime for price and qty
- does **not** override decoded price/qty with exact real row values

Directories:
- `saved_LOB_stream/fixed_start_decoded_real_tokens_openbidanchor_txncomplete_000617XSHE_20260404_222446`
- `saved_LOB_stream/fixed_start_decoded_real_tokens_openbidanchor_txncomplete_000981XSHE_20260404_222446`
- `saved_LOB_stream/fixed_start_decoded_real_tokens_openbidanchor_txncomplete_002263XSHE_20260404_222446`
- `saved_LOB_stream/fixed_start_decoded_real_tokens_openbidanchor_txncomplete_002366XSHE_20260404_222446`

### Clean realflow reference

Purpose:
- regenerated standalone realflow reference using corrected txn-complete semantics
- this is the current benchmark for replay evaluation

Directories:
- `saved_LOB_stream/fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_000617XSHE_20260405_132652`
- `saved_LOB_stream/fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_000981XSHE_20260405_132652`
- `saved_LOB_stream/fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_002263XSHE_20260405_132652`
- `saved_LOB_stream/fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_002366XSHE_20260405_132652`

### Processed realflow inputs

Purpose:
- upstream processed data tables and bin records used by the active replays and clean reference generator

Directory:
- `saved_LOB_stream/processed_real_flow/`

## Active Log Directory

### Clean-reference eval logs

Purpose:
- authoritative replay-vs-clean-reference evaluation logs for `exactPQ` and `txncomplete`

Directory:
- `logs/eval_against_clean_refs/`

Files:
- `000617XSHE_exactpq_vs_cleanref.log`
- `000617XSHE_txncomplete_vs_cleanref.log`
- `000981XSHE_exactpq_vs_cleanref.log`
- `000981XSHE_txncomplete_vs_cleanref.log`
- `002263XSHE_exactpq_vs_cleanref.log`
- `002263XSHE_txncomplete_vs_cleanref.log`
- `002366XSHE_exactpq_vs_cleanref.log`
- `002366XSHE_txncomplete_vs_cleanref.log`

All of these SLURM reruns completed successfully and end with `Evaluation complete`.

## Active Scripts and Their Roles

### Evaluation

- `scripts/hist_script/eval_generated_stream.py`
  - computes generated-only metrics and reference-comparison metrics
  - patched on 2026-04-05 so it can robustly locate `LOB_bench/` from this workspace layout

### Preprocessing

- `scripts/hist_script/preprocess_real_lob_20250710_openbidanchor.py`
  - old open-anchor preprocessing path without txn-complete semantics
- `scripts/hist_script/preprocess_real_lob_20250710_openbidanchor_txncomplete.py`
  - txn-complete preprocessing path
  - produces processed rows and bin records for the txn-complete representation

### Replay

- `scripts/hist_script/replay_decoded_real_tokens_openbidanchor_fixed_start.py`
  - old open-anchor baseline replay
- `scripts/hist_script/replay_decoded_real_tokens_openbidanchor_exactprice_fixed_start.py`
  - exact-price replay variant
  - with `--use-exact-qty`, this becomes the `exactPQ` branch
- `scripts/hist_script/replay_decoded_real_tokens_openbidanchor_txncomplete_fixed_start.py`
  - txn-complete replay branch
  - explicit execution semantics
  - still decodes price and qty from bins

### Clean reference generation

- `scripts/hist_script/generate_lobster_stream_real_openbidanchor_txncomplete_fixed_start.py`
  - regenerated realflow reference using corrected txn-complete semantics
  - current canonical reference generator

### SLURM launchers worth keeping

- clean-reference eval launchers:
  - `run_eval_generated_stream_cleanref_exactpq_*.sh`
  - `run_eval_generated_stream_cleanref_exactpq_parallel.sh`
  - `run_eval_generated_stream_cleanref_txncomplete_*.sh`
  - `run_eval_generated_stream_cleanref_txncomplete_parallel.sh`
- replay launchers for kept branches:
  - `run_replay_decoded_real_tokens_openbidanchor_*.sh`
  - `run_replay_decoded_real_tokens_openbidanchor_parallel.sh`
  - `run_replay_decoded_real_tokens_openbidanchor_exactprice_exactqty_*.sh`
  - `run_replay_decoded_real_tokens_openbidanchor_exactprice_exactqty_parallel.sh`
  - `run_replay_decoded_real_tokens_openbidanchor_txncomplete_*.sh`
  - `run_replay_decoded_real_tokens_openbidanchor_txncomplete_parallel.sh`
- clean reference generation launchers:
  - `run_generate_lobster_stream_real_openbidanchor_txncomplete_*.sh`
  - `run_generate_lobster_stream_real_openbidanchor_txncomplete_parallel.sh`

## What Was Bugged

The broken reference path was the old realflow generator that produced directories now archived under `saved_LOB_stream/archive_bugged_reconstruction/`.

Core bug:
- aggressive flow was effectively double-represented
- an aggressive interaction could appear once as a visible post and again as a removal or execution
- this created invalid crossed or locked intermediate book states in the exported reference streams

Consequence:
- earlier spread comparison metrics against those old reference exports were contaminated and not trustworthy

This bug is fixed in the clean generator by:
- rejecting passive crossing posts
- applying txn-complete removals directly to resting liquidity
- exporting the resulting book states only after corrected semantics are applied

## General Pipeline

### Old open-anchor branch

1. preprocess real flow into open-anchor representation
2. create bin record and tokenized rows
3. replay tokenized rows by decoding from bins
4. export LOBSTER-like stream
5. evaluate against reference

### ExactPQ branch

1. use old open-anchor processed rows and bin record
2. decode event timing and event type from tokens
3. override decoded price with exact row `Price`
4. override decoded qty with exact row `OrderQty`
5. export replay stream
6. evaluate against clean reference

### Txn-complete branch

1. preprocess real flow into txn-complete representation
2. create txn-complete bin record and tokenized rows
3. decode token into explicit event semantics including txn-complete sides
4. replay decoded event against current synthetic book state
5. export replay stream
6. evaluate against clean reference

### Clean reference branch

1. use txn-complete processed realflow rows
2. rebuild book with corrected passive-post and txn-complete semantics
3. export standalone realflow LOBSTER-like stream
4. use this stream as the canonical benchmark

## Important Representation Facts

### Old open-anchor binning regime

The active old open-anchor and txn-complete experiments used the old fixed-open-best-bid anchor binning regime:

- fixed 09:31 best-bid anchor
- 26 price bins
- 26 qty bins
- within-bin empirical resampling from saved per-bin value distributions

This is **not** the newer hypothetical custom regime with exact singleton center bins and aggregated edge bins. That newer regime was discussed conceptually but not implemented in the active runs.

### ExactPQ vs txn-complete is not a one-factor ablation

`exactPQ`:
- old event semantics
- exact real price
- exact real qty

`txncomplete`:
- explicit txn-complete event semantics
- decoded price from bins
- decoded qty from bins

Therefore, when `txncomplete` beats `exactPQ`, it is doing so despite still carrying binning error in price and qty.

## What We Found So Far

### 1. The clean reconstructed reference now looks believable

Compared with the old broken reference path:

- pathological crossed-book behavior is gone
- spread distributions are plausible
- zero-remove rates among actual remove attempts are low enough to be acceptable

Zero-remove rates among cancel/execute attempts:
- 000617: 3.080%
- 000981: 1.535%
- 002263: 3.013%
- 002366: 1.693%

### 2. The old benchmark artifact was real

The earlier spread failure was heavily contaminated by the broken reference generator. After switching to the clean reference, the comparison became much healthier and more interpretable.

### 3. Event semantics matter more than price/qty binning

This is the main result so far.

Under the clean reference benchmark, `txncomplete` is much better than `exactPQ` on the most structural metrics, especially spread and inter-arrival timing, even though `txncomplete` still decodes price and qty from bins.

Selected Wasserstein comparison (`exactPQ` vs `txncomplete`):

#### Spread
- 000617: 0.3541 vs 0.1956
- 000981: 0.0227 vs 0.0183
- 002263: 0.1849 vs 0.0122
- 002366: 0.3121 vs 0.1081

#### Log inter-arrival time
- 000617: 0.3935 vs 0.0381
- 000981: 0.2950 vs 0.0331
- 002263: 0.4738 vs 0.0152
- 002366: 0.4295 vs 0.0585

#### OFI
- 000617: 0.4957 vs 0.2454
- 000981: 0.1200 vs 0.2101
- 002263: 0.6347 vs 0.1065
- 002366: 0.2653 vs 0.2322

#### Volume per minute
- 000617: 0.3710 vs 0.0595
- 000981: 0.0259 vs 0.0182
- 002263: 0.5371 vs 0.0343
- 002366: 0.3140 vs 0.1611

Interpretation:
- price and qty binning still matter
- but the larger distortion in the old regime came from not modeling txn-complete semantics explicitly

### 4. ExactPQ still helps some size metrics

`exactPQ` can still be better on some raw size/touch metrics for specific names, because it re-injects exact real price and qty into the old regime. But overall structural realism is now stronger under txn-complete.

### 5. 981 is no longer a spread problem

Under the cleaned reference, 981 looks fine on spread. The remaining concern for 981 is more about price dynamics and returns, not top-of-book spread realism.

## Current Bottom Line

The current best conclusion is:

- the txn-complete representation is much more acceptable and realistic as a reconstruction of real LOB flow than the old non-txn-complete regime
- the main improvement appears to come from event semantics rather than from removing price/qty binning alone
- the clean realflow reference is now the correct benchmark to use

## Recommended Next Experiment

The next natural branch is:

- txn-complete + exact real price + exact real qty

Reason:
- if txn-complete already beats `exactPQ` while still decoding price and qty from bins, then combining txn-complete semantics with exact price/qty should be the cleanest next ablation and may be the strongest replay branch overall

## Resume Guidance For Another Agent

If resuming from here:

1. Use `logs/eval_against_clean_refs/` as the authoritative replay-vs-reference eval logs.
2. Use `saved_LOB_stream/fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_*_20260405_132652` as the canonical clean realflow references.
3. Do **not** use anything under `archive_bugged_reconstruction/` as the benchmark.
4. Keep the old open-anchor baseline, `exactPQ`, and `txncomplete` branches as the main comparison set.
5. If implementing the next ablation, create a new branch rather than overwriting existing replay scripts.
