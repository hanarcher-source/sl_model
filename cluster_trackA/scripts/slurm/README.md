# Slurm — Track A clustering

Canonical batch scripts live next to your other jobs (same `#SBATCH` pattern: **`batch`**, **`--gres=gpu:1`**, **`--reservation=finai`**, **`logs/...%j.out`**).

From `stock_language_model/`:

## One job, three stocks (sequential)

```bash
sbatch scripts/run_cluster_track_a_pool0709_train_onejob.sh
```

- Log: `logs/run_cluster_track_a_pool0709_train_onejob_<jobid>.out`
- Parse: `grep '^###' logs/run_cluster_track_a_pool0709_train_onejob_*.out`
- Data: `cluster_trackA/data/cluster_runs/pool0709_train_<jobid>/<STOCK>/`

## Array (3 parallel tasks)

```bash
sbatch scripts/run_cluster_track_a_pool0709_train_array.sh
```

- Logs: `logs/run_cluster_track_a_pool0709_train_array_<arrayid>_<taskid>.out`

## Env overrides (either driver)

```bash
STRIDE=3 K_MAX=40 RUN_NAME=pilot_001 sbatch scripts/run_cluster_track_a_pool0709_train_onejob.sh
```

The older copies under `cluster_trackA/scripts/slurm/*.slurm` are deprecated; use **`scripts/run_cluster_track_a_pool0709_*.sh`** only.
