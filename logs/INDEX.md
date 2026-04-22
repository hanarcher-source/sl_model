# Logs index

This folder is organized into subfolders by experiment/task.

## Folders

- `preprocess/`: preprocess + pooled-bin preprocess runs
- `training/`: blank GPT2 training runs
- `eval_pool0709_0710/`: pooled-bin 0709+0710 experiment eval (direct replay + model inference)
- `eval_splitcancel/`: txncomplete split-cancel pipeline logs
- `eval_other/`: other eval / inference runs not in pooled-bin experiment
- `analysis/`: one-off analysis scripts (match-rate, etc.)
- `eval_against_clean_refs/`: curated “cleanref” comparisons (kept as-is)
- `archive_*`: older archived experiments (kept as-is)

## Quick tips

- If you’re looking for the latest pooled-bin 0710 comparison:
  - direct replay logs: `eval_pool0709_0710/run_replay_decoded_real_tokens_openbidanchor_txncomplete_pool0709_0710_eval_0710_*`
  - model inference logs: `eval_pool0709_0710/run_inference_blankgpt2_pool0709_0710_eval_0710_*`

