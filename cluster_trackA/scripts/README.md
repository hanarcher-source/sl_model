# `cluster_trackA/scripts` — canonical **copies**

These files are **`cp -a` snapshots** of the main repo trainers / preprocessors / evaluators (not symlinks). Upstream lives under `stock_language_model/scripts/` and `stock_language_model/scripts/hist_script/`.

## Files (copy → original)

| Copy in this folder | Original path |
|---------------------|---------------|
| `compute_overall_scores_lobbench_style.py` | `scripts/hist_script/compute_overall_scores_lobbench_style.py` |
| `eval_generated_stream.py` | `scripts/hist_script/eval_generated_stream.py` |
| `preprocess_real_lob_twoday_pool_openbidanchor_txncomplete.py` | `scripts/hist_script/...` |
| `preprocess_real_lob_20250710_openbidanchor_txncomplete.py` | `scripts/hist_script/...` |
| `inference_replay_blankgpt2_dynamic_anchor_txncomplete_fixed_start.py` | `scripts/hist_script/...` |
| `inference_replay_blankgpt2_openbidanchor_txncomplete_fixed_start.py` | `scripts/hist_script/...` |
| `train_blankgpt2_openbidanchor_txncomplete_single_day.py` | `scripts/...` |
| `train_blankgpt2_dynamic_anchor_txncomplete_single_day.py` | `scripts/...` |
| `train_blankgpt2_sentence_preset_anchor_s2ip_txncomplete_single_day.py` | `scripts/...` |

## How to run

- **Preferred:** from repository root `stock_language_model/`, call the **canonical** originals so you always track upstream fixes:
  - `python scripts/hist_script/eval_generated_stream.py ...`
- **Using copies here:** run with cwd = `stock_language_model` and put the repo root on `PYTHONPATH` so imports like `utility.*` resolve (inference / eval scripts often need this):
  - `cd /finance_ML/zhanghaohan/stock_language_model && PYTHONPATH=. python cluster_trackA/scripts/eval_generated_stream.py /path/to/exp_dir`

The two **train** scripts `train_blankgpt2_*sentence*...` and `train_blankgpt2_openbidanchor...` import each other as **siblings** in the same directory; with both copied here, they can be launched from `cluster_trackA/scripts/` if cwd and data paths in argparse match your layout.

## Refresh copies

When upstream changes and you want this folder to match:

```bash
SLM=/finance_ML/zhanghaohan/stock_language_model
DST=$SLM/cluster_trackA/scripts
for f in \
  hist_script/compute_overall_scores_lobbench_style.py \
  hist_script/eval_generated_stream.py \
  hist_script/preprocess_real_lob_twoday_pool_openbidanchor_txncomplete.py \
  hist_script/preprocess_real_lob_20250710_openbidanchor_txncomplete.py \
  hist_script/inference_replay_blankgpt2_dynamic_anchor_txncomplete_fixed_start.py \
  hist_script/inference_replay_blankgpt2_openbidanchor_txncomplete_fixed_start.py \
  train_blankgpt2_openbidanchor_txncomplete_single_day.py \
  train_blankgpt2_dynamic_anchor_txncomplete_single_day.py \
  train_blankgpt2_sentence_preset_anchor_s2ip_txncomplete_single_day.py
do cp -a "$SLM/scripts/$f" "$DST/"
done
```
