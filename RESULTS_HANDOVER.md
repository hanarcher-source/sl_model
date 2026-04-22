# Results Handover

This document is a compact handoff summary of the main results and lessons learned
so far across the whole `stock_language_model` project.

It is meant to help a new agent quickly answer:

1. what we already ran,
2. what worked,
3. what did not,
4. what the current best findings are,
5. what is still open.

For infrastructure and file locations, see:

- `MANUAL.md`

For the historical running log, see:

- `DIARY.md`

## 1. Big Picture

The project has explored two main modeling regimes:

1. **Order-flow / token-stream generation**
2. **1Hz book-state generation**

The order-flow regime is older and more extensive.
The 1Hz book-state regime is newer but now fully functional.

## 2. Main Research Questions So Far

### Order-flow side

- Can GPT-style next-token models generate realistic order flows?
- Do anchors help?
- If anchors help, should they be:
  - dynamic,
  - sentence-based,
  - clustered,
  - vocab-code-like?
- Is a pretrained backbone actually better than blank/random init?

### Book-state side

- Can we model the market directly as 1Hz snapshots?
- Is one-token-per-whole-book feasible? (answer: no)
- Does a per-slot parallel-head design work? (answer: yes)
- What evaluation metrics make sense for snapshots?
- Does pretrained help for snapshot modeling?
- What decoding temperature works best?

## 3. What Was Achieved

### 3.1 Order-flow infrastructure is mature

We now have:

- raw-to-token preprocessors,
- multiple model families,
- replay infrastructure,
- comparative LOB-Bench-style evaluation,
- Slurm launchers for reproducible training/eval.

### 3.2 Anchor research became a real subproject

Anchor variants explored include:

- open-anchor / no-anchor baseline,
- dynamic anchor,
- sentence preset anchors,
- S2IP-style alignment,
- gated cross-attention,
- Track A clustered anchors,
- cluster-to-GPT2-vocab-code anchors.

### 3.3 1Hz snapshot pipeline exists end-to-end

This now includes:

- preprocessing from `mdl_6_28_0.csv`,
- dense 1Hz reindex + forward fill,
- 5-minute anchor-relative tokenization,
- temporal GPT-style model with 20 parallel heads,
- token-space eval,
- decoded snapshot-metric eval,
- temperature sweep support,
- blank vs pretrained comparison.

## 4. Key Representation Findings

### 4.1 Whole-book single token is not feasible

Trying to represent an entire 20-slot book state as one giant discrete token leads
to a combinatorial explosion. This was ruled out conceptually.

### 4.2 Per-slot factorization is the practical design

The workable design is:

- 20 slots (10 ask + 10 bid)
- one joint token per slot
- temporal dependence handled by the backbone
- parallel heads predict the next snapshot

This is the basis of the current 1Hz book-state model.

### 4.3 Sentence anchors are not naturally “semantic” for this domain

The sentence-anchor geometry suggested strong colinearity / low diversity after projection.
This means generic sentence anchors are not automatically meaningful in microstructure space.

### 4.4 Clustered anchors are more promising

Track A clustering is more domain-grounded and appears to be a more meaningful anchor direction
than naive sentence semantics.

## 5. Order-Flow Results: Main Lessons

### 5.1 Blank GPT-style baselines are strong

The blank/random-init baseline is hard to beat consistently.

This remained true in several comparisons:

- longer training budget did not necessarily improve it,
- pretrained often did not help,
- some anchor variants only helped on certain stocks.

### 5.2 Pretrained is not automatically better

In the order-flow regime, several comparisons showed pretrained was often worse than blank on
the main aggregate metrics.

### 5.3 Sentence-anchor K=3 was better than K=5

In fixed-sampling comparisons, sentence K=3 consistently beat K=5 across stocks.

### 5.4 Performance is stock-dependent

Different stocks clearly prefer different settings. There is no universal winner.

### 5.5 002721 / some harder stocks remain difficult

Some stocks repeatedly underperform or show unstable behavior across variants.

## 6. Non-TrackA Variant Summary

This section is specifically about the **non-TrackA** order-generation variants.

The main question here is:

- outside Track A clustering / vocab-code anchors, which model family looked most
  promising relative to the blank and pretrained baselines?

### 6.1 Best overall non-TrackA direction

Judging from the diary tables and the comparison slices we actually ran, the
**most promising overall non-TrackA variant** is:

- **plain sentence-preset S2IP `K=3`, uncentered**

Why:

- it shows repeatable gains over blank on multiple stocks,
- it is consistently better than sentence `K=5`,
- it is more robust than the centered / separate-proj variants,
- and it is the strongest clean non-TrackA branch before moving to Track A.

This is **not** the same as saying it wins on every stock and all six aggregates.
It does not. But it is the best overall direction among the non-TrackA options we tested.

### 6.2 Sentence `K=3` vs blank baseline

Fixed-sampling six-pack:

- decode: `sample=on`, `temperature=1.0`, `top_k=0`
- train: pooled preprocess day `20250709`, `window_len=50`, `epochs=3`
- eval: fixed-start `20250710`

Aggregate metrics:

#### 000617_XSHE

- blank:
  - `W_mean=0.3750`
  - `W_median=0.3349`
  - `W_iqm=0.3381`
  - `L1_mean=0.3729`
  - `L1_median=0.3698`
  - `L1_iqm=0.3509`
- sentence `K=3` uncentered:
  - `W_mean=0.3974`
  - `W_median=0.3615`
  - `W_iqm=0.3625`
  - `L1_mean=0.3785`
  - `L1_median=0.3190`
  - `L1_iqm=0.3386`

Readout:

- `000617` is **mixed**
- blank is better on the three W aggregates and on `L1_mean`
- sentence `K=3` is better on `L1_median` and `L1_iqm`

#### 000981_XSHE

- blank:
  - `W_mean=0.8910`
  - `W_median=0.8709`
  - `W_iqm=0.9280`
  - `L1_mean=0.6078`
  - `L1_median=0.6929`
  - `L1_iqm=0.6459`
- sentence `K=3` uncentered:
  - `W_mean=0.8087`
  - `W_median=0.6981`
  - `W_iqm=0.7735`
  - `L1_mean=0.5481`
  - `L1_median=0.5290`
  - `L1_iqm=0.5300`

Readout:

- `000981` is a **clean all-6 win** for sentence `K=3` vs blank

#### 002263_XSHE

- blank:
  - `W_mean=0.8492`
  - `W_median=0.7335`
  - `W_iqm=0.6545`
  - `L1_mean=0.5825`
  - `L1_median=0.6932`
  - `L1_iqm=0.6090`
- sentence `K=3` uncentered:
  - `W_mean=0.7383`
  - `W_median=0.4728`
  - `W_iqm=0.5053`
  - `L1_mean=0.5303`
  - `L1_median=0.5131`
  - `L1_iqm=0.5406`

Readout:

- `002263` is also a **clean all-6 win** for sentence `K=3` vs blank

#### 002366_XSHE

- blank:
  - `W_mean=0.5672`
  - `W_median=0.4477`
  - `W_iqm=0.4460`
  - `L1_mean=0.3743`
  - `L1_median=0.3386`
  - `L1_iqm=0.3405`
- sentence `K=3` uncentered:
  - `W_mean=0.6819`
  - `W_median=0.6691`
  - `W_iqm=0.6507`
  - `L1_mean=0.5278`
  - `L1_median=0.5301`
  - `L1_iqm=0.5563`

Readout:

- `002366` is a **clear loss on all 6** for sentence `K=3` vs blank

Overall read:

- sentence `K=3` beat blank on all six for `000981` and `002263`
- sentence `K=3` was mixed on `000617`
- sentence `K=3` lost clearly on `002366`

### 6.3 Sentence `K=3` vs pretrained baseline

Three-stock comparison at the same fixed-sampling regime (`T10_k0_win50`):

#### 000617_XSHE

- pretrained baseline:
  - `W_mean=0.429733`
  - `W_median=0.356050`
  - `W_iqm=0.386342`
  - `L1_mean=0.396715`
  - `L1_median=0.371181`
  - `L1_iqm=0.388632`
- sentence `K=3`:
  - `W_mean=0.397449`
  - `W_median=0.361496`
  - `W_iqm=0.362537`
  - `L1_mean=0.378481`
  - `L1_median=0.319047`
  - `L1_iqm=0.338571`

Readout:

- sentence `K=3` wins **5/6**
- it only loses `W_median`

#### 002263_XSHE

- pretrained baseline:
  - `W_mean=0.787683`
  - `W_median=0.468090`
  - `W_iqm=0.591480`
  - `L1_mean=0.529026`
  - `L1_median=0.529426`
  - `L1_iqm=0.553586`
- sentence `K=3`:
  - `W_mean=0.738345`
  - `W_median=0.472834`
  - `W_iqm=0.505315`
  - `L1_mean=0.530308`
  - `L1_median=0.513051`
  - `L1_iqm=0.540633`

Readout:

- sentence `K=3` wins **4/6**
- it loses `W_median` and `L1_mean`
- there is also a minor metric-pool mismatch here:
  - sentence `K=3` has `W_n=19`
  - pretrained has `W_n=18`

#### 002721_XSHE

- pretrained baseline:
  - `W_mean=0.619750`
  - `W_median=0.547501`
  - `W_iqm=0.582236`
  - `L1_mean=0.402331`
  - `L1_median=0.418656`
  - `L1_iqm=0.387080`
- sentence `K=3`:
  - `W_mean=0.604384`
  - `W_median=0.529778`
  - `W_iqm=0.560906`
  - `L1_mean=0.424000`
  - `L1_median=0.432849`
  - `L1_iqm=0.418497`

Readout:

- sentence `K=3` wins **all three W aggregates**
- pretrained wins **all three L1 aggregates**
- so this is a **W vs L1 tradeoff**, not a clean winner

Overall read:

- sentence `K=3` is usually stronger than pretrained on Wasserstein-side metrics
- but it does **not** dominate pretrained on all six for every stock

### 6.4 Centered `K=3` was not promising overall

The centered version improved some router geometry diagnostics, but did **not**
improve the six aggregate metrics versus the uncentered baseline.

Exact centered results:

#### 000617_XSHE

- sentence `K=3` centered:
  - `W_mean=0.4709`
  - `W_median=0.4512`
  - `W_iqm=0.4466`
  - `L1_mean=0.4010`
  - `L1_median=0.3587`
  - `L1_iqm=0.3696`

#### 000981_XSHE

- sentence `K=3` centered:
  - `W_mean=0.8655`
  - `W_median=0.8372`
  - `W_iqm=0.8563`
  - `L1_mean=0.5759`
  - `L1_median=0.7137`
  - `L1_iqm=0.6111`

#### 002263_XSHE

- sentence `K=3` centered:
  - `W_mean=0.7865`
  - `W_median=0.5523`
  - `W_iqm=0.5737`
  - `L1_mean=0.5211`
  - `L1_median=0.5606`
  - `L1_iqm=0.5511`

#### 002366_XSHE

- sentence `K=3` centered:
  - `W_mean=0.6937`
  - `W_median=0.6877`
  - `W_iqm=0.6510`
  - `L1_mean=0.5564`
  - `L1_median=0.5830`
  - `L1_iqm=0.5689`

Readout:

- centered `K=3` is **not** the promising non-TrackA direction
- uncentered `K=3` is better almost everywhere in the six-pack comparison

### 6.5 `sepproj` and low-`lambda_align` ablations

These were useful ablations, but not the best general replacement for baseline `K=3`.

#### 000617_XSHE

- baseline `K=3`:
  - `W_mean=0.397`
  - `W_median=0.362`
  - `W_iqm=0.363`
  - `L1_mean=0.379`
  - `L1_median=0.319`
  - `L1_iqm=0.339`
- separate projections:
  - `W_mean=0.449`
  - `W_median=0.395`
  - `W_iqm=0.420`
  - `L1_mean=0.414`
  - `L1_median=0.395`
  - `L1_iqm=0.381`
- low `lambda_align = 1e-2`:
  - `W_mean=0.436`
  - `W_median=0.395`
  - `W_iqm=0.411`
  - `L1_mean=0.381`
  - `L1_median=0.324`
  - `L1_iqm=0.353`

Readout:

- baseline `K=3` wins all six vs both ablations

#### 002263_XSHE

- baseline `K=3`:
  - `W_mean=0.738`
  - `W_median=0.473`
  - `W_iqm=0.505`
  - `L1_mean=0.530`
  - `L1_median=0.513`
  - `L1_iqm=0.541`
- separate projections:
  - `W_mean=0.748`
  - `W_median=0.485`
  - `W_iqm=0.546`
  - `L1_mean=0.547`
  - `L1_median=0.534`
  - `L1_iqm=0.569`
- low `lambda_align = 1e-2`:
  - `W_mean=0.755`
  - `W_median=0.469`
  - `W_iqm=0.523`
  - `L1_mean=0.552`
  - `L1_median=0.543`
  - `L1_iqm=0.569`

Readout:

- baseline `K=3` again wins all six vs both ablations

#### 002721_XSHE

- baseline `K=3`:
  - `W_mean=0.604`
  - `W_median=0.530`
  - `W_iqm=0.561`
  - `L1_mean=0.424`
  - `L1_median=0.433`
  - `L1_iqm=0.419`
- separate projections:
  - `W_mean=0.600`
  - `W_median=0.558`
  - `W_iqm=0.569`
  - `L1_mean=0.408`
  - `L1_median=0.401`
  - `L1_iqm=0.386`
- low `lambda_align = 1e-2`:
  - `W_mean=0.521`
  - `W_median=0.467`
  - `W_iqm=0.493`
  - `L1_mean=0.406`
  - `L1_median=0.407`
  - `L1_iqm=0.395`

Readout:

- `002721` is the exception
- low `lambda_align = 1e-2` beats baseline `K=3` on **all six**
- `sepproj` improves the L1 aggregates but does not beat baseline on the full six-pack

Overall read:

- low-`lambda_align` is a **stock-specific promising branch**
- especially for `002721`
- but it is **not** the strongest overall non-TrackA direction

### 6.6 `unifiedK_ep3` is the strongest challenger, but not the clearest winner

This is the strongest challenger to plain `K=3`, but I would **not** call it the
best overall non-TrackA direction yet.

#### 000617_XSHE

- pretrained baseline:
  - `W_n=20`
  - `W_mean=0.429733`
  - `W_median=0.356050`
  - `W_iqm=0.386342`
  - `L1_n=20`
  - `L1_mean=0.396715`
  - `L1_median=0.371181`
  - `L1_iqm=0.388632`
- sentence `K=3`:
  - `W_n=20`
  - `W_mean=0.397449`
  - `W_median=0.361496`
  - `W_iqm=0.362537`
  - `L1_n=20`
  - `L1_mean=0.378481`
  - `L1_median=0.319047`
  - `L1_iqm=0.338571`
- `unifiedK_ep3`:
  - `W_n=22`
  - `W_mean=0.396400`
  - `W_median=0.346412`
  - `W_iqm=0.349201`
  - `L1_n=20`
  - `L1_mean=0.367772`
  - `L1_median=0.317865`
  - `L1_iqm=0.342683`

Readout:

- on `000617`, `unifiedK_ep3` looks best on most aggregates
- but there is a **`W_n` mismatch caveat**
- its W aggregates use **22** finite terms, while sentence `K=3` uses **20**
- so the W-side comparison here is **not perfectly apples-to-apples**

#### 002263_XSHE

- pretrained baseline:
  - `W_n=18`
  - `W_mean=0.787683`
  - `W_median=0.468090`
  - `W_iqm=0.591480`
  - `L1_n=18`
  - `L1_mean=0.529026`
  - `L1_median=0.529426`
  - `L1_iqm=0.553586`
- sentence `K=3`:
  - `W_n=19`
  - `W_mean=0.738345`
  - `W_median=0.472834`
  - `W_iqm=0.505315`
  - `L1_n=18`
  - `L1_mean=0.530308`
  - `L1_median=0.513051`
  - `L1_iqm=0.540633`
- `unifiedK_ep3`:
  - `W_n=19`
  - `W_mean=0.769799`
  - `W_median=0.512722`
  - `W_iqm=0.564223`
  - `L1_n=18`
  - `L1_mean=0.549275`
  - `L1_median=0.538011`
  - `L1_iqm=0.589375`

Readout:

- on `002263`, plain sentence `K=3` beats `unifiedK_ep3` on **all six**

#### 002721_XSHE

- pretrained baseline:
  - `W_n=20`
  - `W_mean=0.619750`
  - `W_median=0.547501`
  - `W_iqm=0.582236`
  - `L1_n=20`
  - `L1_mean=0.402331`
  - `L1_median=0.418656`
  - `L1_iqm=0.387080`
- sentence `K=3`:
  - `W_n=20`
  - `W_mean=0.604384`
  - `W_median=0.529778`
  - `W_iqm=0.560906`
  - `L1_n=20`
  - `L1_mean=0.424000`
  - `L1_median=0.432849`
  - `L1_iqm=0.418497`
- `unifiedK_ep3`:
  - `W_n=20`
  - `W_mean=0.622995`
  - `W_median=0.660348`
  - `W_iqm=0.604688`
  - `L1_n=20`
  - `L1_mean=0.426127`
  - `L1_median=0.401942`
  - `L1_iqm=0.424294`

Readout:

- on `002721`, `unifiedK_ep3` is **mixed**
- it is not clearly best
- sentence `K=3` remains best on the W aggregates
- pretrained is still better on some L1-side aggregates

Overall read:

- `unifiedK_ep3` is interesting and worth remembering
- but it is **not yet the cleanest “most promising overall” answer**
- the best overall non-TrackA direction remains **plain sentence `K=3` uncentered**

## 7. Track A / Cluster-Based Results

Track A cluster-based anchor work is an important positive direction.

Recent results (corrected apples-to-apples comparison against pretrain):

- **000617**: vocab-code cluster anchor variant beat pretrain on **5/6** aggregates
- **002263**: vocab-code cluster anchor variant beat pretrain on **3/6**
- **002721**: vocab-code cluster anchor variant lost on all six in the corrected table

This means:

- cluster-based anchors are promising,
- but not robustly dominant yet,
- and comparisons must always be checked carefully for exact metric-pool consistency.

## 8. Order-Flow Direct-Replay Baseline

These results are important because they are the **direct replay baseline without
model generation**.

That means:

- tokens are the ground-truth preprocessed `order_token`s,
- there is **no autoregressive model prediction** involved,
- replay still includes the normal decode + book-update pipeline,
- so these numbers measure the distortion from:
  - binning,
  - stochastic decode within bins,
  - and replay mechanics.

These are therefore a useful lower-bound / sanity-check baseline for the
order-flow regime.

### 8.1 Original 5-side `txncomplete` direct-replay baseline

Schema:

- `openbidanchor_txncomplete`
- **5-side**
- cancel is **not** split by side
- transaction-complete remains side-aware

Saved run family:

- `fixed_start_decoded_real_tokens_openbidanchor_txncomplete_*_20260404_222446`

Aggregates:

#### 000617_XSHE

- `W_mean=0.447424`
- `W_median=0.245374`
- `W_iqm=0.273835`
- `L1_mean=0.381619`
- `L1_median=0.361693`
- `L1_iqm=0.356623`

#### 000981_XSHE

- `W_mean=0.294888`
- `W_median=0.025712`
- `W_iqm=0.139458`
- `L1_mean=0.267204`
- `L1_median=0.078888`
- `L1_iqm=0.152347`

#### 002263_XSHE

- `W_mean=0.217085`
- `W_median=0.068102`
- `W_iqm=0.110222`
- `L1_mean=0.248606`
- `L1_median=0.087997`
- `L1_iqm=0.152746`

#### 002366_XSHE

- `W_mean=0.231084`
- `W_median=0.192082`
- `W_iqm=0.191128`
- `L1_mean=0.216442`
- `L1_median=0.209922`
- `L1_iqm=0.205859`

### 8.2 Later pooled-bin 5-side direct-replay reruns

We also reran direct replay later under the pooled-bin regime for a subset of stocks.

These should **not** be mixed blindly with the earlier table; they come from a
different preprocessing setup:

- pooled day fitting for bin edges
- eval-on-0710 rerun setup

Saved run family:

- `pool_0709_0710_eval_0710/fixed_start_decoded_real_tokens_openbidanchor_txncomplete_*_20260406_174836`

#### 000617_XSHE

- `W_mean=0.315636`
- `W_median=0.212553`
- `W_iqm=0.213626`
- `L1_mean=0.265835`
- `L1_median=0.232060`
- `L1_iqm=0.232965`

#### 002263_XSHE

- `W_mean=0.295313`
- `W_median=0.106409`
- `W_iqm=0.171405`
- `L1_mean=0.266047`
- `L1_median=0.164296`
- `L1_iqm=0.195923`

### 8.3 Interpretation

These direct-replay baselines are important reference points:

- if a model is much worse than direct replay, the gap is mostly model-generation error
- if direct replay is already imperfect, that imperfection comes from the tokenization /
  decode / replay pipeline itself
- when comparing new order-generation experiments, always check whether the baseline
  being used is:
  - the original 5-side direct replay, or
  - the later pooled-bin 5-side direct replay

## 9. 1Hz Book-State Results

## 9.1 Preprocessing

Important protocol decision:

- Fit `vol_edges` on **0709 only**
- Apply the same edges to **0710**

This is correct and avoids leakage.

### 9.2 Blank 1Hz parallel-head model trained successfully

Trained for:

- `000617_XSHE`
- `002263_XSHE`
- `002721_XSHE`

The model is:

- temporal GPT-style backbone
- 20 parallel heads
- CE averaged across the 20 heads

### 9.3 1Hz eval setup

The current standard 1Hz eval we used is:

- day: **0710**
- start time: **10:00:00**
- context: **60 seconds**
- horizon: **300 seconds** (5 minutes)
- real reference: 1Hz snapshot feed from `mdl_6_28_0.csv`
- generated decoding: autoregressive token generation at 1Hz

### 9.4 Snapshot-only metrics that make sense

We explicitly separated snapshot-valid metrics from message-flow metrics.

Current snapshot-relevant metrics are:

- `spread`
- `mid_return_1s_log`
- `bid_depth_10`
- `ask_depth_10`
- `depth_imbalance_10`

These are the current meaningful high-level metrics for 1Hz snapshot comparison.

### 9.5 Temperature matters

For `002263_XSHE`, a small temperature sweep on 0710 @ 10:00 showed:

- `temp = 0.7` was clearly better than `1.0`, `1.3`, `1.7`
  on the **snapshot aggregate Wasserstein mean**.

This is an important operational finding:

- lower temperature improved realism for the 1Hz book-state model in this setup.

### 9.6 Temp=0.7 full snapshot results for blank model

At **temp=0.7**, **0710 10:00–10:05**, the snapshot aggregate results were:

#### 000617_XSHE

- `spread`: W=0.153911, L1=0.11
- `mid_return_1s_log`: W=0.366451, L1=0.00334448
- `bid_depth_10`: W=1.00911, L1=0.776667
- `ask_depth_10`: W=0.716212, L1=0.58
- `depth_imbalance_10`: W=0.985992, L1=0.743333

Aggregates:

- **W_mean=0.646335**
- **W_median=0.716212**
- **W_iqm=0.689552**
- **L1_mean=0.442669**
- **L1_median=0.58**
- **L1_iqm=0.477778**

#### 002263_XSHE

- `spread`: W=0.177746, L1=0.4
- `mid_return_1s_log`: W=0.175092, L1=0.00334448
- `bid_depth_10`: W=0.214137, L1=0.746667
- `ask_depth_10`: W=0.440888, L1=0.736667
- `depth_imbalance_10`: W=1.03557, L1=0.7

Aggregates:

- **W_mean=0.408687**
- **W_median=0.214137**
- **W_iqm=0.27759**
- **L1_mean=0.517336**
- **L1_median=0.7**
- **L1_iqm=0.612222**

#### 002721_XSHE

- `spread`: W=0.358037, L1=0.38
- `mid_return_1s_log`: W=0.284647, L1=0.00334448
- `bid_depth_10`: W=1.27921, L1=0.896667
- `ask_depth_10`: W=0.633756, L1=0.76
- `depth_imbalance_10`: W=1.48354, L1=0.92

Aggregates:

- **W_mean=0.807837**
- **W_median=0.633756**
- **W_iqm=0.757**
- **L1_mean=0.592002**
- **L1_median=0.76**
- **L1_iqm=0.678889**

## 10. 1Hz Blank vs Pretrained (temp = 0.7)

We implemented a true pretrained-backbone version for the 1Hz book-state model and
compared it to the blank model on the same eval:

- **0710**
- **start 10:00**
- **context 60**
- **horizon 300**
- **temperature 0.7**

### 10.1 000617_XSHE

Pretrained was better overall.

Blank:

- W_mean=0.646335, W_median=0.716212, W_iqm=0.689552
- L1_mean=0.442669, L1_median=0.58, L1_iqm=0.477778

Pretrained:

- **W_mean=0.437662, W_median=0.464551, W_iqm=0.455146**
- **L1_mean=0.314669, L1_median=0.376667, L1_iqm=0.345556**

### 10.2 002263_XSHE

Blank was clearly better overall.

Blank:

- **W_mean=0.408687, W_median=0.214137, W_iqm=0.27759**
- **L1_mean=0.517336, L1_median=0.7, L1_iqm=0.612222**

Pretrained:

- W_mean=0.814801, W_median=0.816767, W_iqm=0.817923
- L1_mean=0.562002, L1_median=0.67, L1_iqm=0.681111

### 10.3 002721_XSHE

Blank was better overall.

Blank:

- **W_mean=0.807837, W_median=0.633756, W_iqm=0.757**
- **L1_mean=0.592002, L1_median=0.76, L1_iqm=0.678889**

Pretrained:

- W_mean=0.914262, W_median=0.931181, W_iqm=0.942631
- L1_mean=0.723153, L1_median=0.729097, L1_iqm=0.718588

### 10.4 Conclusion from blank vs pretrained at 1Hz

Same broad lesson as in other parts of the project:

- **pretrained is not uniformly better**

At least in this current book-state setup:

- `000617` preferred pretrained
- `002263` preferred blank
- `002721` preferred blank

## 11. What Worked

### Worked well

- strong baseline blank GPT-style models
- robust preprocessing / training / eval infra
- Track A clustered anchor direction
- 1Hz book-state factorized token design
- leakage-aware 0709->0710 binning
- lower-temperature sampling for snapshot generation

## 12. What Did Not Work / Did Not Generalize

- giant one-token whole-book encoding
- naive sentence semantics as a strong universal anchor
- assuming pretrained would always help
- assuming one hyperparameter setting works across all stocks

## 13. Current Best Operational Defaults

These are not final truths, but they are reasonable current defaults:

### For 1Hz book-state

- use the dense 1Hz preprocess
- use 5-minute anchors
- fit bins on 0709 only
- evaluate on 0710 with reused 0709 bins
- use the 3-stock set:
  - `000617_XSHE`
  - `002263_XSHE`
  - `002721_XSHE`
- use `P=41`, `V=31`, `K=1271`
- train with:
  - `context-sec=60`
  - `stride-sec=5`
  - `epochs=3`
  - `patience=3`
  - `batch-size=256`
  - `lr=2e-4`
  - `amp`
- use snapshot metrics:
  - spread
  - 1s mid return
  - bid depth 10
  - ask depth 10
  - depth imbalance 10
- evaluate with:
  - `start-time=10:00:00`
  - `context-sec=60`
  - `horizon-sec=300`
  - `sample`
- for decoding, `temp=0.7` is the current preferred default

### Newest agreed regime for apples-to-apples 1Hz runs

If a new agent wants to run the current standard comparison, use exactly this:

```bash
preprocess: 1Hz dense snapshots, 5-minute anchors, P=41, V=31, K=1271
train:      context=60s, stride=5s, epochs=3, batch=256, lr=2e-4, amp
eval:       day=0710, start=10:00:00, context=60s, horizon=300s
decode:     sample, temperature=0.7
metrics:    spread, mid_return_1s_log, bid_depth_10, ask_depth_10, depth_imbalance_10
summary:    W_mean/W_median/W_iqm and L1_mean/L1_median/L1_iqm
```

### For order-flow track

- always compare using the same metric pool
- check `W_n` / `L1_n` when comparing aggregate scores
- do not assume semantic anchors are helping unless replay metrics say so

## 14. Open Questions / Next Experiments

The main open questions are:

1. Should the future focus be on:
   - order-flow modeling,
   - book-state modeling,
   - or a hybrid?

2. For 1Hz book-state, should we expand the metric set to include:
   - L1 bid/ask depth,
   - spread change,
   - volatility windows,
   - shape/entropy of depth profile?

3. Should temperature be swept across all stocks and both blank / pretrained regimes?

4. Should snapshot generation be compared against:
   - raw snapshot feed only,
   - or also 1Hz snapshots reconstructed from replayed order-flow?

5. Should Track A cluster ideas be transferred more directly into the 1Hz book-state model?

## 15. Practical Recommendation for a New Agent

If a new agent has to pick up the project now:

1. read `MANUAL.md`
2. read this file
3. read `DIARY.md` selectively
4. decide which track to work on:
   - order-flow
   - 1Hz book-state
5. use the existing Slurm launchers rather than inventing new submission patterns
6. keep new results recorded in this file and/or `DIARY.md`

