# `stock_language_model` Manual

This document is a handoff-oriented infrastructure manual for the whole
`/finance_ML/zhanghaohan/stock_language_model` project.

It is written for a new agent or researcher who knows nothing about the codebase
and needs to quickly understand:

1. what the project does,
2. where the important files live,
3. how preprocessing / training / inference / evaluation are wired,
4. how Slurm jobs are supposed to be launched,
5. what the two main modeling tracks are right now.

## 1. Project Purpose

This repo contains a research framework for **generative modeling of limit order
book dynamics**.

There are currently two major tracks:

1. **Order-flow track**
   - model the market as a tokenized event/message stream
   - replay generated events into a simulated order book
   - compare generated streams/books against real references

2. **1Hz book-state track**
   - model the market as a sequence of 1-second snapshots
   - encode 10 ask + 10 bid levels as discrete tokens
   - predict the next 1Hz snapshot using a temporal backbone + parallel heads
   - evaluate generated 1Hz snapshots against real 1Hz snapshots

These two tracks share the same broad research loop:

`raw data -> preprocess -> train -> generate -> evaluate -> compare metrics`

## 2. Top-Level Directory Layout

The most important directories under `stock_language_model/` are:

- `scripts/`
  Main training / preprocess / inference / evaluation scripts, plus Slurm launchers.

- `scripts/hist_script/`
  Older but still important evaluation and replay scripts for the order-flow track.

- `utility/`
  Shared helper logic, especially book replay / decoding / simulation functions.

- `saved_LOB_stream/processed_real_flow/`
  Preprocessed tokenized order-flow datasets.

- `saved_LOB_stream/processed_book_state/`
  Preprocessed 1Hz snapshot datasets.

- `training_runs/`
  Model checkpoints, metadata JSONs, and outputs from training jobs.

- `logs/`
  Slurm stdout/stderr logs.

- `cluster_trackA/`
  Track A anchor-clustering subproject. This is a meaningful subproject, not just junk.

- `DIARY.md`
  Ongoing historical experiment log. This is long and detailed, but useful.

## 3. External Data / Environment

### Raw data location

Raw source data lives outside this repo under:

- `/finance_ML/zhanghaohan/LOB_data/<DAY>/`

Important raw files include:

- `mdl_6_28_0.csv`
  Snapshot-style 10-level LOB table. Used for the 1Hz book-state track.

- `mdl_6_33_0.csv`
  Order post / order-flow-related raw input in the older preprocessors.

- `mdl_6_36_0.csv`
  Trade / execution-related raw input.

### Python environment

The project usually runs under:

- conda bootstrap:
  - `/finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh`
- main env:
  - `/finance_ML/zhanghaohan/mycondaenv`

Some ad hoc local inspection was also done with:

- `/finance_ML/zhanghaohan/env_py310/bin/python`

For Slurm jobs, the standard pattern in this repo is:

```bash
source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv
```

## 4. Operational Rule for Slurm Wrappers

This repo uses two kinds of shell scripts:

1. **actual Slurm job scripts**
   - these contain `#SBATCH ...`
   - these should be submitted with `sbatch`

2. **wrapper submit scripts**
   - these call `sbatch` on one or more real job scripts
   - these should be run with `bash`, **not** `sbatch`

This matters a lot.

If you accidentally `sbatch` a wrapper, Slurm will try to schedule the wrapper as
its own batch job, which leads to confusing pending jobs and no useful work.

Example:

- correct:
  ```bash
  bash scripts/submit_bookstate_parallelheads_anchor5m_0709.sh
  ```

- wrong:
  ```bash
  sbatch scripts/submit_bookstate_parallelheads_anchor5m_0709.sh
  ```

## 5. Track A: Order-Flow Pipeline

This is the older and more mature pipeline.

### 5.1 Raw input -> tokenized real flow

The key preprocessors live under:

- `scripts/hist_script/preprocess_real_lob_20250710_openbidanchor_txncomplete.py`
- `scripts/hist_script/preprocess_real_lob_twoday_pool_openbidanchor_txncomplete.py`

These create tokenized order/event streams with:

- price binning,
- quantity binning,
- interval binning,
- side/event vocab,
- anchor metadata.

Typical outputs:

- `final_result_for_merge_realflow_openbidanchor_txncomplete_<DAY>_<TAG>_*.joblib`
- `bin_record_realflow_openbidanchor_txncomplete_<DAY>_<TAG>_*.json`

Typical output directory:

- `saved_LOB_stream/processed_real_flow/pool_0709_0710_openbidanchor_txncomplete/`

### 5.1.1 Order-flow event schema: 5-side vs 6-side

This distinction is important and should be made explicit for any new agent.

#### 5-side `openbidanchor_txncomplete`

This is the standard `txncomplete` schema used in many existing training and
inference runs.

Side/event mapping:

- `49`: bid-side post
- `50`: ask-side post
- `99`: cancel, **without distinguishing bid vs ask**
- `129`: transaction-complete event on one side
- `130`: transaction-complete event on the other side

Interpretation:

- in the **5-side** schema, **cancel does not distinguish side**
- but **transaction-complete events do still distinguish side**

So yes: the 5-side version means cancel is merged into one class, while
transaction-complete remains side-aware.

This is why the 5-side vocab is:

- `26 * 26 * 12 * 5 = 40560`

#### 6-side `openbidanchor_txncomplete_splitcancel`

This is the newer schema when we want more faithful cancel semantics.

Side/event mapping:

- `49`: bid-side post
- `50`: ask-side post
- `97`: bid-side cancel
- `98`: ask-side cancel
- `129`: transaction-complete event on one side
- `130`: transaction-complete event on the other side

Interpretation:

- in the **6-side split-cancel** schema, **cancel also distinguishes side**
- transaction-complete events remain side-aware as before

This is why the 6-side vocab is:

- `26 * 26 * 12 * 6 = 48672`

Practical takeaway:

- use **5-side `txncomplete`** when continuing older apples-to-apples model runs
- use **6-side `txncomplete_splitcancel`** when the experiment specifically wants
  side-aware cancel handling end to end

### 5.2 Core order-flow trainers

Important training entrypoints include:

- `scripts/train_blankgpt2_openbidanchor_txncomplete_single_day.py`
  - blank/random-init GPT-style baseline on order-flow tokens

- `scripts/train_blankgpt2_dynamic_anchor_txncomplete_single_day.py`
  - dynamic-anchor order-flow model

- `scripts/train_blankgpt2_dynamic_anchor_variants_txncomplete_single_day.py`
  - dynamic-anchor ablations / variants

- `scripts/train_blankgpt2_sentence_preset_anchor_s2ip_txncomplete_single_day.py`
  - sentence-anchor S2IP-style trainer

- `scripts/train_blankgpt2_sentence_preset_anchor_s2ip_gated_crossattn_txncomplete_single_day.py`
  - sentence-anchor + gated cross-attention

- `scripts/train_blankgpt2_sentence_preset_anchor_s2ip_vocabcode_from_clusters_txncomplete_single_day.py`
  - Track A cluster-based vocab-code anchor trainer

### 5.3 Common order-flow Slurm launchers

There are many stock-specific launchers in `scripts/`. Examples:

- `run_train_pool0709_sentence_s2ip_trackA_clustered_ep3_000617XSHE.sh`
- `run_train_pool0709_sentence_s2ip_gca_trackA_clustered_ep3_topk1_routerdiag_000617XSHE.sh`
- `run_train_pool0709_vocabcode_cluster_ep3_topk1_routerdiag_000617XSHE.sh`

Common wrapper examples:

- `submit_train_pool0709_sentence_s2ip_trackA_clustered_ep3_parallel.sh`
- `submit_train_pool0709_sentence_s2ip_gca_trackA_clustered_ep3_topk1_routerdiag_parallel.sh`
- `submit_train_pool0709_vocabcode_cluster_ep3_topk1_routerdiag_parallel.sh`

### 5.4 Order-flow inference / replay / eval

Important inference/eval entrypoints:

- `scripts/hist_script/inference_replay_blankgpt2_openbidanchor_txncomplete_fixed_start.py`
  - generate order tokens from a checkpoint, replay them through the book, emit LOBSTER-style files, then evaluate

- `scripts/hist_script/replay_decoded_real_tokens_openbidanchor_txncomplete_fixed_start.py`
  - replay real decoded token streams through the simulator

- `scripts/hist_script/eval_generated_stream.py`
  - main comparative evaluation script for replayed message/book outputs

### 5.4.1 Current order-generation regime

If the task is **order generation** rather than 1Hz book-state generation, the
current agreed regime is:

- preprocessing family: `openbidanchor_txncomplete`
- raw inputs merged from:
  - `mdl_6_33_0.csv`
  - `mdl_6_28_0.csv`
  - `mdl_6_36_0.csv`
- anchor time: fixed `09:31:00`
- bin counts:
  - price bins: `26`
  - quantity bins: `26`
  - interval bins: `12`
- preferred preprocessing for consistent cross-day comparison:
  - `preprocess_real_lob_twoday_pool_openbidanchor_txncomplete.py`
  - fit price/qty/interval bins once on pooled days, then apply to each day

Current decoding regime for newer order-generation comparisons:

- token-level decoding:
  - `--sample`
  - `--temperature 1.0`
  - `--top-k 0`
- field-level decode from token to event:
  - sampled from the stored bin distributions in `bin_record`

So the order-generation pipeline is stochastic at two levels:

- sample the next token
- sample the realized price/qty/interval value within that token bin

### 5.4.2 Critical workflow: compare a new order-generation model against baselines

This is the most important operational workflow for the order-flow track.

A new agent should think of the comparison loop as having **three separate runs**
that must all use the **same evaluation window** and the **same schema**:

1. **clean real reference generation**
2. **direct replay baseline**
3. **model generation + replay**

If any of these use a different:

- stock,
- trade day,
- start time,
- lookahead duration,
- 5-side vs 6-side schema,
- preprocessing / binning regime,

then the comparison is no longer apples-to-apples.

#### A. Required inputs for a valid comparison

Before running any comparison, make sure all of the following are fixed:

- stock, e.g. `000617_XSHE`
- eval day, usually `20250710`
- fixed-start time, usually `10:00:00`
- lookahead, usually `10` minutes
- schema:
  - `openbidanchor_txncomplete` (5-side), or
  - `openbidanchor_txncomplete_splitcancel` (6-side)
- preprocess/bins:
  - prefer pooled two-day fitting if continuing the newer regime
- decode regime:
  - token sampling on
  - `temperature=1.0`
  - `top_k=0`

#### B. Step 1: preprocess the real flow for the eval day

Create the processed real-flow joblib and matching `bin_record`.

Use one of:

- `scripts/hist_script/preprocess_real_lob_20250710_openbidanchor_txncomplete.py`
- `scripts/hist_script/preprocess_real_lob_twoday_pool_openbidanchor_txncomplete.py`

Outputs you need:

- `final_result_for_merge_realflow_<...>.joblib`
- `bin_record_realflow_<...>.json`

These are the canonical tokenized inputs for both:

- direct replay baseline
- model replay eval

#### C. Step 2: create the clean real reference directory

This step creates the “real-world target” that both baseline replay and model
generation should be evaluated against.

Main script:

- `scripts/hist_script/generate_lobster_stream_real_openbidanchor_txncomplete_fixed_start.py`

What it does:

- initializes the book from the real snapshot at the chosen start time
- replays the **real** fixed-start window into LOBSTER-style outputs
- writes a reference directory like:
  - `saved_LOB_stream/fixed_start_realflow_generate_lobster_openbidanchor_txncomplete_<TAG>_<timestamp>/`

This directory is then passed as:

- `--real-ref-dir`

to both baseline replay and model replay eval.

#### D. Step 3: run the direct replay baseline

This is the baseline **without model generation**.

Main script:

- `scripts/hist_script/replay_decoded_real_tokens_openbidanchor_txncomplete_fixed_start.py`

Important args:

- `--stock`
- `--processed-real-flow-path`
- `--bin-record-path`
- `--real-ref-dir`
- `--lob-snap-path`
- `--trade-date-str`
- `--start-time`
- `--sim-lookahead-minutes`
- `--split-cancel-sides` if and only if using the 6-side schema

What it does:

- takes the real `order_token`s,
- decodes them stochastically through the same decode logic,
- replays them through the book simulator,
- evaluates against the clean real reference,
- writes a run directory containing:
  - `metrics_summary.json`
  - `run.log`
  - LOBSTER-style outputs

Why this baseline is critical:

- it measures the loss from
  - tokenization,
  - within-bin decode sampling,
  - and replay mechanics
- without adding model-generation error

This is the baseline every new model should be compared against.

#### E. Step 4: run model generation + replay

Main script:

- `scripts/hist_script/inference_replay_blankgpt2_openbidanchor_txncomplete_fixed_start.py`

Despite the name, this script is the general fixed-start model replay/eval entrypoint
for multiple variants. It can infer the model type from checkpoint state, or you can
set it explicitly with `--model-variant`.

Important args:

- `--stock`
- `--checkpoint`
- `--processed-real-flow-path`
- `--bin-record-path`
- `--real-ref-dir`
- `--baseline-metrics-json`
- `--window-len`
- `--vocab-size`
- `--sample`
- `--temperature 1.0`
- `--top-k 0`
- optional:
  - `--model-variant auto`
  - `--model-variant sentence_preset_s2ip`
  - `--model-variant dynamic_anchor`
  - etc.

What it does:

1. loads the checkpoint,
2. seeds with the first `window_len` real tokens,
3. autoregressively generates the rest,
4. decodes generated tokens into events,
5. replays them into the simulator,
6. evaluates against the same clean real reference,
7. optionally writes a direct baseline comparison JSON if `--baseline-metrics-json`
   is provided.

#### F. Step 5: compare against baseline correctly

The correct comparison contract is:

- same preprocess output family
- same `bin_record`
- same clean `real_ref_dir`
- same fixed-start window
- same schema (5-side vs 6-side)
- same decode regime

Artifacts to inspect:

- direct baseline:
  - `metrics_summary.json`
- model run:
  - `metrics_summary.json`
  - `comparison_vs_direct_token_replay.json` if `--baseline-metrics-json` was provided

When quoting results, use:

- `lobbench_style_overall`
  - `W_mean`
  - `W_median`
  - `W_iqm`
  - `L1_mean`
  - `L1_median`
  - `L1_iqm`

Also check:

- `W_n`
- `L1_n`

because if the number of finite metric terms differs, the aggregates are not
strictly averaged over the same metric pool.

#### G. Minimal recipe for a new model

If a new agent has trained a new checkpoint and wants to compare it to existing
baselines, the operational recipe is:

1. preprocess eval-day flow with the agreed schema and bins
2. generate the clean real reference directory
3. run direct replay baseline and save its `metrics_summary.json`
4. run model inference replay using:
   - the same processed flow
   - the same `bin_record`
   - the same `real_ref_dir`
   - the same decode regime
5. compare:
   - direct replay vs real reference
   - model replay vs real reference
   - model replay vs direct replay

#### H. Common failure modes in this workflow

- comparing a model run to the wrong `real_ref_dir`
- mixing 5-side preprocess with 6-side replay flags
- using a `bin_record` from a different preprocessing regime
- comparing metrics where `W_n` / `L1_n` differ without noting it
- changing decode settings between models and calling it a model comparison
- forgetting that direct replay is already stochastic because within-bin decode is sampled

#### I. What should be logged every time

For every new order-generation eval, record all of the following:

- stock
- train day(s)
- eval day
- schema:
  - 5-side `txncomplete`, or
  - 6-side `txncomplete_splitcancel`
- preprocess script used
- exact processed-flow path
- exact `bin_record` path
- exact clean `real_ref_dir`
- checkpoint path
- model variant
- window length
- decode regime:
  - sample on/off
  - temperature
  - top-k
- generated row count
- `W_n`, `L1_n`
- the six aggregate metrics

If these are not written down, later comparisons become hard to trust.

### 5.5 Order-flow outputs

Training outputs:

- `training_runs/.../<TAG>/model_cache/*_best.pt`
- `training_runs/.../<TAG>/*_meta.json`

Eval outputs:

- `saved_LOB_stream/.../metrics_summary.json`
- `saved_LOB_stream/.../eval.log`
- plots under `plots/`

### 5.6 Order-flow metrics

The order-flow/replay eval can compute many comparative metrics, including:

- spread
- orderbook imbalance
- ask/bid volume
- depth-related features
- conditional variants like `spread | time`
- LOB-Bench-style 6 aggregates:
  - `W_mean`, `W_median`, `W_iqm`
  - `L1_mean`, `L1_median`, `L1_iqm`

These aggregates are implemented in:

- `scripts/hist_script/eval_generated_stream.py`
- `scripts/hist_script/compute_overall_scores_lobbench_style.py`

## 6. Track B: 1Hz Book-State Pipeline

This is the newer pipeline.

### 6.1 Goal

Represent a 10-level bid/ask book snapshot every 1 second as discrete tokens, then
train a temporal model to predict the next snapshot.

### 6.2 Preprocessing entrypoint

Main script:

- `scripts/preprocess_bookstate_mdl628_anchor5m_bins_20250709.py`

What it does:

1. reads `LOB_data/<DAY>/mdl_6_28_0.csv`
2. floors updates to seconds
3. keeps the last snapshot per second
4. densifies to a full 1Hz grid by forward-filling
5. defines 5-minute anchor buckets
6. computes tokenized relative encodings for:
   - 10 ask levels
   - 10 bid levels

Current encoding:

- price delta bins:
  - signed ticks relative to bucket anchor
  - `P = 41` for `[-20, +20]`

- volume delta bins:
  - signed log-volume delta
  - `V = 31`

- joint token:
  - `K = P * V = 1271`

Outputs:

- joblib with:
  - `SecurityID`
  - `TradeDate`
  - `TransactDT_SEC`
  - `BucketStart`
  - `book_token_00 ... book_token_19`
  - optionally raw and anchor columns if `--keep-raw` is used

- meta JSON with:
  - `vol_edges`
  - `tick_size`
  - vocab/bin config

### 6.3 1Hz preprocess Slurm launchers

Files:

- `scripts/run_preprocess_bookstate_mdl628_anchor5m_20250709.sh`
  - preprocess 0709

- `scripts/run_preprocess_bookstate_mdl628_anchor5m_20250710_apply0709bins.sh`
  - preprocess 0710 while reusing 0709-fitted volume edges

Wrapper:

- `scripts/submit_bookstate_parallelheads_anchor5m_0709.sh`
  - submit preprocess 0709 + dependent training jobs
  - run with `bash`

### 6.4 1Hz trainer

Main trainer:

- `scripts/train_bookstate_parallelheads_anchor5m.py`

Current model:

- temporal GPT2-style backbone over per-second snapshots
- each snapshot contains 20 slots
- output is predicted via **20 parallel heads**
- loss is average CE across the 20 heads

Trainer supports:

- **blank/random** backbone
- **pretrained GPT-2** backbone

Important flags:

- `--data-joblib`
- `--stock`
- `--codebook-size`
- `--context-sec`
- `--stride-sec`
- `--init-pretrained-backbone`
- `--pretrained-backbone-name`
- `--pretrained-local-only`

### 6.5 1Hz training Slurm launchers

Blank trainers:

- `scripts/run_train_bookstate_parallelheads_anchor5m_000617XSHE.sh`
- `scripts/run_train_bookstate_parallelheads_anchor5m_002263XSHE.sh`
- `scripts/run_train_bookstate_parallelheads_anchor5m_002721XSHE.sh`

Pretrained trainers:

- `scripts/run_train_bookstate_parallelheads_anchor5m_pretrained_000617XSHE.sh`
- `scripts/run_train_bookstate_parallelheads_anchor5m_pretrained_002263XSHE.sh`
- `scripts/run_train_bookstate_parallelheads_anchor5m_pretrained_002721XSHE.sh`

Typical blank output root:

- `training_runs/pool_0709_bookstate_parallelheads_anchor5m/`

Typical pretrained output root:

- `training_runs/pool_0709_bookstate_parallelheads_anchor5m_pretrained/`

### 6.6 1Hz fixed-start inference / eval

Main script:

- `scripts/inference_eval_bookstate_parallelheads_1s_fixed_start.py`

What it does:

1. loads checkpoint + preprocessed 1Hz joblib
2. uses real 1Hz context as seed
3. generates the next horizon autoregressively
4. compares generated snapshots vs real snapshots

Current important args:

- `--stock`
- `--checkpoint`
- `--data-joblib`
- `--meta-json`
- `--start-time`
- `--context-sec`
- `--horizon-sec`
- `--sample`
- `--temperature`
- `--run-tag`

### 6.7 1Hz eval Slurm launchers

Core fixed-start evals:

- `run_inference_eval_bookstate_parallelheads_1s_fixed_start_0710_000617XSHE.sh`
- `run_inference_eval_bookstate_parallelheads_1s_fixed_start_0710_002263XSHE.sh`
- `run_inference_eval_bookstate_parallelheads_1s_fixed_start_0710_002721XSHE.sh`

Wrapper:

- `submit_bookstate_eval_1s_fixed_start_0710_start1000_T300.sh`
  - run with `bash`

Temperature-specific eval launchers were also added for targeted experiments, e.g.:

- `run_eval_bookstate_1s_0710_temp0p7_000617XSHE.sh`
- `run_eval_bookstate_1s_0710_temp0p7_002721XSHE.sh`
- `run_temp_sweep_eval_bookstate_1s_0710_002263XSHE.sh`
- `run_eval_bookstate_1s_0710_temp0p7_pretrained_000617XSHE.sh`
- `run_eval_bookstate_1s_0710_temp0p7_pretrained_002263XSHE.sh`
- `run_eval_bookstate_1s_0710_temp0p7_pretrained_002721XSHE.sh`

### 6.8 1Hz metrics

There are now two eval layers for 1Hz:

#### A. Token-space metrics

These evaluate the generated slot-token distributions directly.

#### B. Snapshot-relevant decoded metrics

These are the meaningful high-level metrics for snapshot-only comparison:

- `spread`
- `mid_return_1s_log`
- `bid_depth_10`
- `ask_depth_10`
- `depth_imbalance_10`

The evaluator also computes 6 aggregate summary numbers over this snapshot metric set:

- `W_mean`, `W_median`, `W_iqm`
- `L1_mean`, `L1_median`, `L1_iqm`

These appear in output JSON under:

- `snapshot_reference_comparison.metrics`
- `snapshot_lobbench_style_overall`

## 7. Important Output Locations

### Order-flow processed data

- `saved_LOB_stream/processed_real_flow/...`

### 1Hz book-state processed data

- 0709:
  - `saved_LOB_stream/processed_book_state/pool_0709_bookstate_anchor5m_mdl628/`

- 0710 using 0709 bins:
  - `saved_LOB_stream/processed_book_state/pool_0710_bookstate_anchor5m_mdl628_apply0709bins/`

### Training checkpoints

- blank 1Hz:
  - `training_runs/pool_0709_bookstate_parallelheads_anchor5m/<TAG>/`

- pretrained 1Hz:
  - `training_runs/pool_0709_bookstate_parallelheads_anchor5m_pretrained/<TAG>/`

### 1Hz eval outputs

- standard 0710 fixed-start eval:
  - `saved_LOB_stream/pool_0710_bookstate_eval_1s/bsph_T300_ctx60_start1000/`

- temp sweep / temp-specific dirs:
  - `saved_LOB_stream/pool_0710_bookstate_eval_1s/bsph_sweep_T300_ctx60_start1000/`
  - `saved_LOB_stream/pool_0710_bookstate_eval_1s/bsph_temp_singles_T300_ctx60_start1000/`
  - `saved_LOB_stream/pool_0710_bookstate_eval_1s/bsph_temp0p7_blank_vs_pretrained/`

### Logs

- all Slurm logs:
  - `logs/`

Examples:

- `logs/run_train_bookstate_parallelheads_anchor5m_002263XSHE_<jobid>.out`
- `logs/run_eval_bookstate_1s_0710_temp0p7_pretrained_002263XSHE_<jobid>.out`

## 8. Typical Usage Recipes

### Recipe A: train blank 1Hz book-state model on 0709

1. preprocess 0709:
   ```bash
   bash scripts/submit_bookstate_parallelheads_anchor5m_0709.sh
   ```
   or submit pieces manually:
   ```bash
   sbatch scripts/run_preprocess_bookstate_mdl628_anchor5m_20250709.sh
   sbatch scripts/run_train_bookstate_parallelheads_anchor5m_000617XSHE.sh
   ...
   ```

2. checkpoints will appear under:
   - `training_runs/pool_0709_bookstate_parallelheads_anchor5m/`

### Recipe B: evaluate 0710 at 10:00 for 5 minutes

1. preprocess 0710 using 0709 volume edges:
   ```bash
   bash scripts/submit_bookstate_eval_1s_fixed_start_0710_start1000_T300.sh
   ```

2. eval JSONs will appear under:
   - `saved_LOB_stream/pool_0710_bookstate_eval_1s/bsph_T300_ctx60_start1000/`

### Recipe C: run a temperature sweep

Example implemented for `002263_XSHE`:

```bash
sbatch scripts/run_temp_sweep_eval_bookstate_1s_0710_002263XSHE.sh
```

### Recipe D: train pretrained 1Hz model

Submit:

```bash
sbatch scripts/run_train_bookstate_parallelheads_anchor5m_pretrained_000617XSHE.sh
sbatch scripts/run_train_bookstate_parallelheads_anchor5m_pretrained_002263XSHE.sh
sbatch scripts/run_train_bookstate_parallelheads_anchor5m_pretrained_002721XSHE.sh
```

### Recipe E: compare blank vs pretrained at a fixed temperature

At the moment this was done with dedicated scripts for `temp=0.7`, 0710, start 10:00:

```bash
sbatch scripts/run_eval_bookstate_1s_0710_temp0p7_pretrained_000617XSHE.sh
...
```

## 9. Agreed Parameters And Current Defaults

This section is the most important one for a new agent who wants to continue
running experiments in the same regime we have already agreed on.

### 9.1 Current default stock universe

The current standard 3-stock set is:

- `000617_XSHE`
- `002263_XSHE`
- `002721_XSHE`

### 9.2 Current default 1Hz book-state preprocessing regime

These are the settings we have consistently used for the current book-state track:

- raw file: `LOB_data/<DAY>/mdl_6_28_0.csv`
- time resolution: **1 second**
- densification: **yes**, forward-fill missing seconds
- anchor scheme: **5-minute anchor buckets**
- anchor parameter: `--anchor-minutes 5`
- max price offset: `--max-tick 20`
- volume bins: `--vol-bins 31`
- price bins: `P = 41` from `[-20, +20]`
- joint codebook size: `K = 41 * 31 = 1271`
- chunksize: `--chunksize 500000`

Important protocol:

- fit volume edges on **0709**
- reuse the same edges on **0710**
- for 0710 eval preprocessing, use `--vol-edges-from-meta`
- for 0710 eval preprocessing, also use `--keep-raw`

This is the current agreed leakage-safe setup.

### 9.3 Current default 1Hz training regime

Current standard training settings for the parallel-head book-state model:

- model: `train_bookstate_parallelheads_anchor5m.py`
- snapshot slots: **20** heads
  - 10 ask + 10 bid
- context length: `--context-sec 60`
- stride: `--stride-sec 5`
- epochs: `--epochs 3`
- patience: `--patience 3`
- batch size: `--batch-size 256`
- learning rate: `--lr 2e-4`
- AMP: `--amp`
- GPU per task: **1**

Interpretation:

- one training sample uses a **60-second prompt**
- target is the **next 1-second snapshot**
- rolling windows are created every **5 seconds**

### 9.4 Backbone variants currently supported

We have used two versions:

#### Blank baseline

- no `--init-pretrained-backbone`
- output root:
  - `training_runs/pool_0709_bookstate_parallelheads_anchor5m/`

#### Pretrained GPT-2 backbone

- `--init-pretrained-backbone`
- `--pretrained-backbone-name gpt2`
- `--pretrained-local-only`
- output root:
  - `training_runs/pool_0709_bookstate_parallelheads_anchor5m_pretrained/`

### 9.5 Current default 1Hz evaluation regime

This is the standard fixed-start evaluation protocol we have been using:

- eval day: **20250710**
- start time: `10:00:00`
- context length: `60` seconds
- generation horizon: `300` seconds
- decoding mode: **sampling**, not argmax
- evaluator script:
  - `scripts/inference_eval_bookstate_parallelheads_1s_fixed_start.py`

Core eval flags:

- `--start-time 10:00:00`
- `--context-sec 60`
- `--horizon-sec 300`
- `--codebook-size 1271`
- `--P 41`
- `--V 31`
- `--sample`

### 9.6 Newest agreed decoding regime

The **newest agreed decoding regime** for the 1Hz book-state model is:

- decode by **sampling**
- use **temperature = 0.7** as the current preferred default
- evaluate on **0710**, starting **10:00:00**
- use **300 generated snapshots** after a **60-second real context**

Why this is the default now:

- we explicitly swept temperatures
- among the tested values, `0.7` gave the best result in the targeted comparison
- this is the regime used for the recent blank-vs-pretrained comparison

So unless there is a specific reason to explore otherwise, a new agent should start from:

```bash
--sample --temperature 0.7 --start-time 10:00:00 --context-sec 60 --horizon-sec 300
```

### 9.7 Current default snapshot metric set

For the 1Hz book-state track, the currently agreed meaningful decoded metrics are:

- `spread`
- `mid_return_1s_log`
- `bid_depth_10`
- `ask_depth_10`
- `depth_imbalance_10`

And the standard summary aggregates are:

- `W_mean`
- `W_median`
- `W_iqm`
- `L1_mean`
- `L1_median`
- `L1_iqm`

These are the main numbers a new agent should quote for high-level comparison.

### 9.8 Practical recommendation for new experiments

If the goal is a new **comparable** 1Hz experiment, keep the following fixed unless
the experiment is explicitly about changing them:

- same 3 stocks
- same 1Hz densification
- same 5-minute anchors
- same `P=41`, `V=31`, `K=1271`
- same train context `60s`
- same train stride `5s`
- same eval start `10:00:00`
- same eval horizon `300s`
- same sampling decode
- same default temperature `0.7`
- same decoded metric set and same 6 aggregates

If you change any of these, record it clearly because results will no longer be
directly apples-to-apples.

## 10. What To Read First If You Are New

Recommended order for a new agent:

1. this file: `MANUAL.md`
2. `RESULTS_HANDOVER.md`
3. `DIARY.md`
4. for book-state work:
   - `scripts/preprocess_bookstate_mdl628_anchor5m_bins_20250709.py`
   - `scripts/train_bookstate_parallelheads_anchor5m.py`
   - `scripts/inference_eval_bookstate_parallelheads_1s_fixed_start.py`
5. for order-flow work:
   - `scripts/train_blankgpt2_openbidanchor_txncomplete_single_day.py`
   - `scripts/train_blankgpt2_sentence_preset_anchor_s2ip_*.py`
   - `scripts/hist_script/inference_replay_blankgpt2_openbidanchor_txncomplete_fixed_start.py`
   - `scripts/hist_script/eval_generated_stream.py`

## 11. Common Pitfalls

- Do **not** `sbatch` wrapper scripts that are meant to be run with `bash`.
- For 0710 out-of-sample 1Hz evaluation, do **not** refit volume bin edges on 0710.
  Use `--vol-edges-from-meta` from the 0709 preprocessing meta.
- Snapshot metrics are **not** message-flow metrics. Do not interpret cancel/fill/inter-arrival
  metrics on 1Hz snapshot-only outputs.
- The 1Hz evaluator currently compares:
  - token-space metrics, and
  - decoded snapshot metrics.
  Make sure you know which one you are quoting.

## 12. Current Status

As of the latest work:

- order-flow pipeline is mature and heavily experimented
- 1Hz book-state pipeline is functional end-to-end
- blank and pretrained 1Hz variants are both implemented
- snapshot metric evaluation and aggregate reporting are working
- temperature sweeps are supported for 1Hz generation

For actual experiment conclusions, see:

- `RESULTS_HANDOVER.md`

