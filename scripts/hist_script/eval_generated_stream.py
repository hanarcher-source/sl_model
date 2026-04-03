"""
eval_generated_stream.py
========================
Compute stylized-fact metrics on a single generated LOBSTER stream.

All computation uses only the (message, orderbook) pair produced by the
generator — no real reference data is required.

Usage
-----
    python eval_generated_stream.py                          # auto-picks latest exp dir
    python eval_generated_stream.py /path/to/exp_dir        # explicit exp dir

Outputs (written inside <exp_dir>/)
------------------------------------
    plots/
        mid_price.png
        spread.png
        returns_dist.png
        autocorr.png
        inter_arrival_time.png
        l1_volume.png
        volume_per_minute.png
        orderbook_imbalance.png
        orderflow_imbalance.png
        event_type_counts.png
        time_to_fill.png          (only if matched fill pairs found)
        time_to_cancel.png        (only if matched cancel pairs found)
    metrics_summary.json
"""

import sys
import os
import json
import glob
import argparse
import warnings

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ── resolve LOB_bench on sys.path ─────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOB_BENCH = os.path.abspath(os.path.join(_HERE, '..', 'LOB_bench'))
if _LOB_BENCH not in sys.path:
    sys.path.insert(0, _LOB_BENCH)

import eval as ev          # noqa: E402  (after path setup)
import data_loading as dl  # noqa: E402
import metrics as lbm      # noqa: E402
import partitioning as lbp # noqa: E402


# Metrics from lob_bench-main/run_bench.py that are comparative by design.
# These require real reference sequences/book snapshots and are not attempted here.
LOB_BENCH_REAL_REFERENCE_METRICS = [
    'spread',
    'orderbook_imbalance',
    'log_inter_arrival_time',
    'log_time_to_cancel',
    'ask_volume_touch',
    'bid_volume_touch',
    'ask_volume',
    'bid_volume',
    'limit_ask_order_depth',
    'limit_bid_order_depth',
    'ask_cancellation_depth',
    'bid_cancellation_depth',
    'limit_ask_order_levels',
    'limit_bid_order_levels',
    'ask_cancellation_levels',
    'bid_cancellation_levels',
    'vol_per_min',
    'ofi',
    'ofi_up',
    'ofi_stay',
    'ofi_down',
    'ask_volume | spread',
    'spread | time',
    'spread | volatility',
]

# Requires true order-level ID linkage between add/cancel/fill messages.
# Our generated output uses synthetic IDs, so these are not attempted.
REQUIRES_TRUE_ORDER_ID_LINKAGE = [
    'time_to_first_fill',
    'time_to_cancel',
]


def _clean_numeric(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _compare_unconditional_distribution(real_scores, gen_scores, discrete=False):
    real_arr = _clean_numeric(real_scores)
    gen_arr = _clean_numeric(gen_scores)
    if len(real_arr) == 0 or len(gen_arr) == 0:
        raise ValueError("empty real or generated sample after cleaning")

    ws_df = pd.DataFrame({
        'score': np.concatenate([real_arr, gen_arr]),
        'type': (['real'] * len(real_arr)) + (['generated'] * len(gen_arr)),
    })
    wasserstein = float(lbm.wasserstein(ws_df, bootstrap_ci=False))

    if discrete:
        groups_real, groups_gen = lbp.group_by_score(real_arr, [gen_arr], discrete=True)
    else:
        groups_real, groups_gen = lbp.group_by_score(real_arr, [gen_arr], bin_method='fd')
    l1_df = lbp.get_score_table(real_arr, [gen_arr], groups_real, groups_gen)
    l1_by_group = float(lbm.l1_by_group(l1_df, bootstrap_ci=False))

    return {
        'n_real': int(len(real_arr)),
        'n_generated': int(len(gen_arr)),
        'wasserstein': wasserstein,
        'l1_by_group': l1_by_group,
    }


def _compare_conditional_distribution(
    eval_real,
    cond_real,
    eval_gen,
    cond_gen,
    cond_discrete=False,
    cond_thresholds=None,
):
    eval_real = np.asarray(eval_real, dtype=float)
    cond_real = np.asarray(cond_real, dtype=float)
    eval_gen = np.asarray(eval_gen, dtype=float)
    cond_gen = np.asarray(cond_gen, dtype=float)

    mask_real = np.isfinite(eval_real) & np.isfinite(cond_real)
    mask_gen = np.isfinite(eval_gen) & np.isfinite(cond_gen)
    eval_real = eval_real[mask_real]
    cond_real = cond_real[mask_real]
    eval_gen = eval_gen[mask_gen]
    cond_gen = cond_gen[mask_gen]

    if len(eval_real) == 0 or len(eval_gen) == 0:
        raise ValueError("empty real or generated conditional sample after cleaning")

    if cond_thresholds is not None:
        groups_real, groups_gen = lbp.group_by_score(
            cond_real,
            [cond_gen],
            thresholds=cond_thresholds,
        )
    elif cond_discrete:
        groups_real, groups_gen = lbp.group_by_score(
            cond_real,
            [cond_gen],
            discrete=True,
        )
    else:
        groups_real, groups_gen = lbp.group_by_score(
            cond_real,
            [cond_gen],
            quantiles=np.arange(0.1, 1.0, 0.1),
        )

    grouped_df = lbp.get_score_table(eval_real, [eval_gen], groups_real, groups_gen)
    weighted_ws = 0.0
    total_weight = 0
    per_group = {}

    for group_id, group_df in grouped_df.groupby('group'):
        n_group = int(len(group_df))
        if n_group == 0:
            continue
        ws_group = float(lbm.wasserstein(group_df[['score', 'type']], bootstrap_ci=False))
        weighted_ws += ws_group * n_group
        total_weight += n_group
        per_group[str(int(group_id))] = {
            'n': n_group,
            'wasserstein': ws_group,
        }

    if total_weight == 0:
        raise ValueError("no non-empty conditional groups")

    return {
        'n_real': int(len(eval_real)),
        'n_generated': int(len(eval_gen)),
        'weighted_wasserstein': float(weighted_ws / total_weight),
        'num_condition_groups': int(len(per_group)),
        'per_group': per_group,
    }


def _compute_reference_comparison_metrics(messages, book, ref_messages, ref_book, logger=None):
    def _log_time_to_cancel(msg_df):
        vals = ev.time_to_cancel(msg_df).dt.total_seconds().replace({0: 1e-9}).values.astype(float)
        return np.log(vals)

    uncond_specs = {
        'spread': {
            'real_fn': lambda m, b: ev.spread(m, b).values,
            'gen_fn': lambda m, b: ev.spread(m, b).values,
            'discrete': True,
        },
        'orderbook_imbalance': {
            'real_fn': lambda m, b: ev.orderbook_imbalance(m, b).values,
            'gen_fn': lambda m, b: ev.orderbook_imbalance(m, b).values,
            'discrete': False,
        },
        'log_inter_arrival_time': {
            'real_fn': lambda m, b: np.log(ev.inter_arrival_time(m).replace({0: 1e-9}).values.astype(float)),
            'gen_fn': lambda m, b: np.log(ev.inter_arrival_time(m).replace({0: 1e-9}).values.astype(float)),
            'discrete': False,
        },
        'log_time_to_cancel': {
            'real_fn': lambda m, b: _log_time_to_cancel(m),
            'gen_fn': lambda m, b: _log_time_to_cancel(m),
            'discrete': False,
        },
        'ask_volume_touch': {
            'real_fn': lambda m, b: ev.l1_volume(m, b).ask_vol.values,
            'gen_fn': lambda m, b: ev.l1_volume(m, b).ask_vol.values,
            'discrete': False,
        },
        'bid_volume_touch': {
            'real_fn': lambda m, b: ev.l1_volume(m, b).bid_vol.values,
            'gen_fn': lambda m, b: ev.l1_volume(m, b).bid_vol.values,
            'discrete': False,
        },
        'ask_volume': {
            'real_fn': lambda m, b: ev.total_volume(m, b, 10).ask_vol_10.values,
            'gen_fn': lambda m, b: ev.total_volume(m, b, 10).ask_vol_10.values,
            'discrete': False,
        },
        'bid_volume': {
            'real_fn': lambda m, b: ev.total_volume(m, b, 10).bid_vol_10.values,
            'gen_fn': lambda m, b: ev.total_volume(m, b, 10).bid_vol_10.values,
            'discrete': False,
        },
        'limit_ask_order_depth': {
            'real_fn': lambda m, b: ev.limit_order_depth(m, b)[0].values,
            'gen_fn': lambda m, b: ev.limit_order_depth(m, b)[0].values,
            'discrete': False,
        },
        'limit_bid_order_depth': {
            'real_fn': lambda m, b: ev.limit_order_depth(m, b)[1].values,
            'gen_fn': lambda m, b: ev.limit_order_depth(m, b)[1].values,
            'discrete': False,
        },
        'ask_cancellation_depth': {
            'real_fn': lambda m, b: ev.cancellation_depth(m, b)[0].values,
            'gen_fn': lambda m, b: ev.cancellation_depth(m, b)[0].values,
            'discrete': False,
        },
        'bid_cancellation_depth': {
            'real_fn': lambda m, b: ev.cancellation_depth(m, b)[1].values,
            'gen_fn': lambda m, b: ev.cancellation_depth(m, b)[1].values,
            'discrete': False,
        },
        'limit_ask_order_levels': {
            'real_fn': lambda m, b: ev.limit_order_levels(m, b)[0].values,
            'gen_fn': lambda m, b: ev.limit_order_levels(m, b)[0].values,
            'discrete': True,
        },
        'limit_bid_order_levels': {
            'real_fn': lambda m, b: ev.limit_order_levels(m, b)[1].values,
            'gen_fn': lambda m, b: ev.limit_order_levels(m, b)[1].values,
            'discrete': True,
        },
        'ask_cancellation_levels': {
            'real_fn': lambda m, b: ev.cancel_order_levels(m, b)[0].values,
            'gen_fn': lambda m, b: ev.cancel_order_levels(m, b)[0].values,
            'discrete': True,
        },
        'bid_cancellation_levels': {
            'real_fn': lambda m, b: ev.cancel_order_levels(m, b)[1].values,
            'gen_fn': lambda m, b: ev.cancel_order_levels(m, b)[1].values,
            'discrete': True,
        },
        'vol_per_min': {
            'real_fn': lambda m, b: ev.volume_per_minute(m, b).values,
            'gen_fn': lambda m, b: ev.volume_per_minute(m, b).values,
            'discrete': False,
        },
        'ofi': {
            'real_fn': lambda m, b: ev.orderflow_imbalance(m, b).values,
            'gen_fn': lambda m, b: ev.orderflow_imbalance(m, b).values,
            'discrete': False,
        },
        'ofi_up': {
            'real_fn': lambda m, b: ev.orderflow_imbalance_cond_tick(m, b, 1).values,
            'gen_fn': lambda m, b: ev.orderflow_imbalance_cond_tick(m, b, 1).values,
            'discrete': False,
        },
        'ofi_stay': {
            'real_fn': lambda m, b: ev.orderflow_imbalance_cond_tick(m, b, 0).values,
            'gen_fn': lambda m, b: ev.orderflow_imbalance_cond_tick(m, b, 0).values,
            'discrete': False,
        },
        'ofi_down': {
            'real_fn': lambda m, b: ev.orderflow_imbalance_cond_tick(m, b, -1).values,
            'gen_fn': lambda m, b: ev.orderflow_imbalance_cond_tick(m, b, -1).values,
            'discrete': False,
        },
    }

    cond_specs = {
        'ask_volume | spread': {
            'eval_real_fn': lambda m, b: ev.l1_volume(m, b).ask_vol.values,
            'eval_gen_fn': lambda m, b: ev.l1_volume(m, b).ask_vol.values,
            'cond_real_fn': lambda m, b: ev.spread(m, b).values,
            'cond_gen_fn': lambda m, b: ev.spread(m, b).values,
            'cond_discrete': True,
            'cond_thresholds': None,
        },
        'spread | time': {
            'eval_real_fn': lambda m, b: ev.spread(m, b).values,
            'eval_gen_fn': lambda m, b: ev.spread(m, b).values,
            'cond_real_fn': lambda m, b: ev.time_of_day(m).values,
            'cond_gen_fn': lambda m, b: ev.time_of_day(m).values,
            'cond_discrete': False,
            'cond_thresholds': np.linspace(0, 24 * 60 * 60, 24),
        },
        'spread | volatility': {
            'eval_real_fn': lambda m, b: ev.spread(m, b).values,
            'eval_gen_fn': lambda m, b: ev.spread(m, b).values,
            'cond_real_fn': lambda m, b: np.repeat(float(ev.volatility(m, b, '0.1s')), len(m)),
            'cond_gen_fn': lambda m, b: np.repeat(float(ev.volatility(m, b, '0.1s')), len(m)),
            'cond_discrete': True,
            'cond_thresholds': None,
        },
    }

    results = {}
    failed = {}

    for name, spec in uncond_specs.items():
        try:
            real_scores = spec['real_fn'](ref_messages, ref_book)
            gen_scores = spec['gen_fn'](messages, book)
            results[name] = _compare_unconditional_distribution(
                real_scores,
                gen_scores,
                discrete=bool(spec.get('discrete', False)),
            )
        except Exception as exc:
            failed[name] = str(exc)
            if logger:
                logger.warning("Reference metric %s skipped: %s", name, exc)

    for name, spec in cond_specs.items():
        try:
            results[name] = _compare_conditional_distribution(
                spec['eval_real_fn'](ref_messages, ref_book),
                spec['cond_real_fn'](ref_messages, ref_book),
                spec['eval_gen_fn'](messages, book),
                spec['cond_gen_fn'](messages, book),
                cond_discrete=bool(spec.get('cond_discrete', False)),
                cond_thresholds=spec.get('cond_thresholds', None),
            )
        except Exception as exc:
            failed[name] = str(exc)
            if logger:
                logger.warning("Reference conditional metric %s skipped: %s", name, exc)

    return results, failed


# ── logging ───────────────────────────────────────────────────────────────────

def _setup_logger(log_file: str):
    """Configure logger with timestamped output to both console and file."""
    logger = logging.getLogger('eval_stream')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers (in case of re-initialization)
    logger.handlers.clear()
    
    # Format: [YYYY-MM-DD HH:MM:SS] [LEVEL] message
    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    return logger


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_serializable(obj):
    """Recursively convert numpy / pandas scalars to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    # pandas NA / NaN scalar
    try:
        if pd.isnull(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def _safe(fn, *args, label=None, default=None, logger=None, **kwargs):
    """Call fn; return *default* on any exception (with a warning)."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        name = label or getattr(fn, '__name__', str(fn))
        msg = f"{name} skipped: {exc}"
        if logger:
            logger.warning(msg)
        else:
            print(f"   [warn] {msg}")
        return default


def _save_fig(fig, path, logger=None):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    msg = f"saved {os.path.basename(path)}"
    if logger:
        logger.info(msg)
    else:
        print(f"  {msg}")


def _log_all_metrics_summary(metrics: dict, logger) -> None:
    """Print all computed metrics in a compact one-line-per-metric form."""
    logger.info("\n── all computed metrics summary ─────────────────────")

    computed = metrics.get('capability_report', {}).get('computed_generated_only_metrics', [])
    for name in computed:
        if name not in metrics:
            continue
        payload = json.dumps(_to_serializable(metrics[name]), sort_keys=True)
        logger.info("  %s: %s", name, payload)

    ref_metrics = metrics.get('reference_comparison', {}).get('metrics', {})
    if ref_metrics:
        logger.info("  [real-vs-generated comparative metrics]")
        for name in sorted(ref_metrics.keys()):
            payload = json.dumps(_to_serializable(ref_metrics[name]), sort_keys=True)
            logger.info("  %s: %s", name, payload)


# ── data loading ──────────────────────────────────────────────────────────────

def load_lobster_pair(exp_dir: str, logger=None):
    """Return (messages_df, book_df, notes_dict) for an experiment directory."""
    notes_path = os.path.join(exp_dir, 'generation_notes.json')
    with open(notes_path) as fh:
        notes = json.load(fh)

    msg_path  = notes['lobster_message_csv']
    book_path = notes['lobster_orderbook_csv']

    # extract ISO date from filename e.g. 000617XSHE_2025-07-10_36000000_...
    fname = os.path.basename(msg_path)
    date  = fname.split('_')[1]         # '2025-07-10'

    if logger:
        logger.info(f"Loading LOBSTER data from {exp_dir}")
    messages = dl.load_message_df(msg_path, parse_time=True)
    book     = dl.load_book_df(book_path)
    dl.add_date_to_time(messages, date)

    # align integer index so book rows match messages rows
    book.index = messages.index

    assert len(messages) == len(book), (
        f"Row count mismatch: {len(messages)} messages vs {len(book)} book rows"
    )
    n_levels = book.shape[1] // 4
    msg = f"Loaded {len(messages):,} rows | {n_levels} book levels | date {date}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return messages, book, notes


# ── metric computation ────────────────────────────────────────────────────────

def compute_metrics(
    messages: pd.DataFrame,
    book: pd.DataFrame,
    logger=None,
    ref_messages: pd.DataFrame = None,
    ref_book: pd.DataFrame = None,
) -> dict:
    out = {}
    computed_now = []

    if logger:
        logger.debug("Starting metric computation...")

    # ---- event type distribution ----------------------------------------
    vc = messages['event_type'].value_counts().sort_index()
    total = len(messages)
    out['event_type_counts']    = {int(k): int(v) for k, v in vc.items()}
    out['event_type_fractions'] = {int(k): round(v / total, 4) for k, v in vc.items()}
    computed_now.extend(['event_type_counts', 'event_type_fractions'])

    # direction distribution
    vd = messages['direction'].value_counts().sort_index()
    out['direction_counts']    = {int(k): int(v) for k, v in vd.items()}
    out['direction_fractions'] = {int(k): round(v / total, 4) for k, v in vd.items()}
    computed_now.extend(['direction_counts', 'direction_fractions'])

    # ---- inter-arrival time (ms) ----------------------------------------
    # Exclude fill/execution rows (event_type 4, 5): they share the same
    # timestamp as the triggering order and are NOT independent model arrivals.
    msg_arrivals = messages[~messages['event_type'].isin([4, 5])]
    iat = ev.inter_arrival_time(msg_arrivals)
    iat_nonneg = iat[iat >= 0]          # include genuine 0-ms simultaneous arrivals
    out['iat_ms'] = {
        'n':              int(len(iat_nonneg)),
        'n_exec_excluded': int((messages['event_type'].isin([4, 5])).sum()),
        'mean':           float(iat_nonneg.mean()),
        'median':         float(iat_nonneg.median()),
        'std':            float(iat_nonneg.std()),
        'p95':            float(iat_nonneg.quantile(0.95)),
    }
    computed_now.append('iat_ms')
    if logger:
        logger.debug("  ✓ inter-arrival time")

    # ---- spread (LOBSTER int price units -> 0.01 tick units) ------------
    sprd      = ev.spread(messages, book)
    sprd_pos  = sprd[sprd > 0]
    out['spread_ticks'] = {
        'tick_size_cny': 0.01,
        'tick_size_price_int': 100,
        'n':              int(len(sprd_pos)),
        'mean':           float((sprd_pos / 100.0).mean())   if len(sprd_pos) else None,
        'median':         float((sprd_pos / 100.0).median()) if len(sprd_pos) else None,
        'std':            float((sprd_pos / 100.0).std())    if len(sprd_pos) else None,
        'pct_zero_spread': float((sprd <= 0).mean()),
    }
    computed_now.append('spread_ticks')
    if logger:
        logger.debug("  ✓ spread")

    # ---- mid-price (LOBSTER int price units -> CNY and 0.01 ticks) ------
    mid = ev.mid_price(messages, book)
    out['mid_price'] = {
        'start_price_int':  float(mid.iloc[0]),
        'end_price_int':    float(mid.iloc[-1]),
        'mean_price_int':   float(mid.mean()),
        'std_price_int':    float(mid.std()),
        'start_cny':        float(mid.iloc[0] / 10000.0),
        'end_cny':          float(mid.iloc[-1] / 10000.0),
        'mean_cny':         float((mid / 10000.0).mean()),
        'std_cny':          float((mid / 10000.0).std()),
        'mean_ticks_0p01':  float((mid / 100.0).mean()),
        'std_ticks_0p01':   float((mid / 100.0).std()),
        'total_return_bps': float((mid.iloc[-1] / mid.iloc[0] - 1) * 10_000),
    }
    computed_now.append('mid_price')
    if logger:
        logger.debug("  ✓ mid-price")

    # ---- 1-min log returns -------------------------------------------
    ret1 = _safe(ev.mid_returns, messages, book, '1min', label='mid_returns_1min', logger=logger)
    if ret1 is not None:
        r = ret1.dropna()
        max_lags = max(1, min(10, len(r) - 2))
        out['returns_1min'] = {
            'n':        int(len(r)),
            'mean':     float(r.mean()),
            'std':      float(r.std()),
            'skew':     float(r.skew()),
            'kurtosis': float(r.kurtosis()),
        }
        acf_vals = _safe(
            ev.autocorr, r, max_lags, None,
            label='autocorr', logger=logger,
        )
        if acf_vals is not None:
            out['acf_returns_1min'] = list(acf_vals)
            computed_now.append('acf_returns_1min')
        computed_now.append('returns_1min')
        if logger:
            logger.debug("  ✓ 1-min returns & autocorr")
    else:
        out['returns_1min'] = None

    # ---- volatility (std of 1-min log returns) -------------------------
    vol = _safe(ev.volatility, messages, book, '1min', label='volatility_1min', logger=logger)
    out['volatility_1min'] = float(vol) if vol is not None else None
    computed_now.append('volatility_1min')

    # ---- L1 volume (best bid / ask) ------------------------------------
    l1vol = _safe(ev.l1_volume, messages, book, label='l1_volume', logger=logger)
    if l1vol is not None:
        out['l1_volume'] = {
            'ask_mean':   float(l1vol['ask_vol'].mean()),
            'bid_mean':   float(l1vol['bid_vol'].mean()),
            'ask_median': float(l1vol['ask_vol'].median()),
            'bid_median': float(l1vol['bid_vol'].median()),
        }
        computed_now.append('l1_volume')
        if logger:
            logger.debug("  ✓ L1 volume")

    # ---- aggregated volume (up to 5 levels) ----------------------------
    n_levels = book.shape[1] // 4
    n_agg    = min(5, n_levels)
    tvol = _safe(ev.total_volume, messages, book, n_agg, label='total_volume', logger=logger)
    if tvol is not None:
        out[f'total_volume_{n_agg}lvl'] = {
            'ask_mean': float(tvol[f'ask_vol_{n_agg}'].mean()),
            'bid_mean': float(tvol[f'bid_vol_{n_agg}'].mean()),
        }
        computed_now.append(f'total_volume_{n_agg}lvl')

    # ---- executed volume per minute ------------------------------------
    vpm = _safe(ev.volume_per_minute, messages, book, label='volume_per_minute', logger=logger)
    if vpm is not None and len(vpm) > 0:
        out['volume_per_minute'] = {
            'n':     int(len(vpm)),
            'mean':  float(vpm.mean()),
            'total': float(vpm.sum()),
        }
        computed_now.append('volume_per_minute')

    # ---- orderbook imbalance (L1) -------------------------------------
    obi = _safe(ev.orderbook_imbalance, messages, book, label='ob_imbalance', logger=logger)
    if obi is not None:
        obi_c = obi.dropna()
        out['ob_imbalance'] = {
            'mean':   float(obi_c.mean()),
            'std':    float(obi_c.std()),
            'median': float(obi_c.median()),
        }
        computed_now.append('ob_imbalance')
        if logger:
            logger.debug("  ✓ orderbook imbalance")

    # ---- orderflow imbalance (rolling 100) ----------------------------
    ofi = _safe(ev.orderflow_imbalance, messages, book, 100, label='orderflow_imbalance', logger=logger)
    if ofi is not None and len(ofi) > 0:
        out['orderflow_imbalance'] = {
            'n':    int(len(ofi)),
            'mean': float(ofi.mean()),
            'std':  float(ofi.std()),
        }
        computed_now.append('orderflow_imbalance')
        if logger:
            logger.debug("  ✓ orderflow imbalance")

    reference_results = {}
    reference_failed = {}
    if ref_messages is not None and ref_book is not None:
        reference_results, reference_failed = _compute_reference_comparison_metrics(
            messages,
            book,
            ref_messages,
            ref_book,
            logger,
        )
        out['reference_comparison'] = {
            'metrics': reference_results,
            'failed_metrics': reference_failed,
            'real_reference_rows': int(len(ref_messages)),
            'generated_rows': int(len(messages)),
        }

    out['capability_report'] = {
        'computed_generated_only_metrics': sorted(set(computed_now)),
        'computed_reference_comparison_metrics': sorted(reference_results.keys()),
        'not_attempted_requires_real_reference_lob_data': (
            [] if (ref_messages is not None and ref_book is not None) else LOB_BENCH_REAL_REFERENCE_METRICS
        ),
        'failed_reference_comparison_metrics': reference_failed,
        'not_attempted_requires_true_order_id_linkage': REQUIRES_TRUE_ORDER_ID_LINKAGE,
        'policy': (
            'Generated-only metrics are always computed. Comparative benchmark metrics are computed '
            'when real reference LOBSTER data is provided via --real_ref_dir.'
        ),
    }

    if logger:
        logger.info(
            "Computed now (generated-only): %d metric groups",
            len(out['capability_report']['computed_generated_only_metrics'])
        )
        if ref_messages is None or ref_book is None:
            logger.info(
                "Skipped (needs real reference LOB data): %d benchmark metric configs",
                len(LOB_BENCH_REAL_REFERENCE_METRICS)
            )
        else:
            logger.info(
                "Computed (with real reference LOB data): %d benchmark metric configs",
                len(reference_results)
            )
            if reference_failed:
                logger.info(
                    "Failed (with real reference LOB data): %d benchmark metric configs",
                    len(reference_failed)
                )
        logger.info(
            "Skipped (needs true order-ID linkage): %d metrics",
            len(REQUIRES_TRUE_ORDER_ID_LINKAGE)
        )
    return out


# ── plotting ──────────────────────────────────────────────────────────────────

def make_plots(messages: pd.DataFrame, book: pd.DataFrame,
               plots_dir: str, notes: dict, logger=None) -> None:

    fname   = os.path.basename(notes['lobster_message_csv'])
    ticker  = notes.get('stock', 'UNKNOWN').replace('_', '')
    date    = fname.split('_')[1]
    hdr     = f"{ticker}  {date}"

    if logger:
        logger.debug("Starting plot generation...")

    # ── 1. Mid-price time series ──────────────────────────────────────────
    mid = ev.mid_price(messages, book)
    mid_cny = mid / 10_000.0    # integer ticks → CNY
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(mid_cny.index, mid_cny.values, lw=0.8, color='steelblue')
    ax.set_title(f'{hdr} — Mid-price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (CNY)')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    fig.autofmt_xdate()
    plt.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, 'mid_price.png'), logger)

    # ── 2. Spread time series + histogram ────────────────────────────────
    sprd = ev.spread(messages, book)
    sprd_cny = sprd / 10_000.0
    p99 = sprd_cny.quantile(0.99)
    sprd_clp = sprd_cny[sprd_cny.between(0, p99)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(sprd_cny.index, sprd_cny.values, lw=0.4, alpha=0.7, color='darkorange')
    axes[0].set_title('Spread over time (CNY)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Spread (CNY)')
    fig.autofmt_xdate()

    axes[1].hist(sprd_clp.values, bins=60, density=True,
                 edgecolor='k', lw=0.3, alpha=0.75, color='darkorange')
    axes[1].set_title('Spread distribution (≤p99)')
    axes[1].set_xlabel('Spread (CNY)')
    axes[1].set_ylabel('Density')

    fig.suptitle(f'{hdr} — Spread', fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, 'spread.png'), logger)

    # ── 3. 1-min returns distribution ─────────────────────────────────────
    ret1 = _safe(ev.mid_returns, messages, book, '1min', label='mid_returns_plot')
    if ret1 is not None:
        r = ret1.dropna()
        if len(r) >= 3:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(r.values, bins=max(10, len(r)),
                    density=True, edgecolor='k', lw=0.4, alpha=0.75, color='mediumseagreen')
            mean_r, std_r = r.mean(), r.std()
            ax.axvline(mean_r, color='red', linestyle='--', lw=1.2, label=f'mean={mean_r:.4f}')
            ax.set_title(f'{hdr} — 1-min log-returns distribution')
            ax.set_xlabel('Log return')
            ax.set_ylabel('Density')
            ax.legend(fontsize=9)
            plt.tight_layout()
            _save_fig(fig, os.path.join(plots_dir, 'returns_dist.png'), logger)

    # ── 4. ACF of 1-min returns ───────────────────────────────────────────
        if len(r) >= 5:
            try:
                import statsmodels.api as sm2
                max_lags = max(1, min(10, len(r) - 2))
                fig, ax = plt.subplots(figsize=(8, 4))
                sm2.graphics.tsa.plot_acf(r.values, lags=max_lags, ax=ax, zero=False)
                ax.set_title(f'{hdr} — ACF of 1-min log-returns')
                ax.set_xlabel('Lag')
                plt.tight_layout()
                _save_fig(fig, os.path.join(plots_dir, 'autocorr.png'), logger)
            except Exception as exc:
                if logger:
                    logger.warning(f"ACF plot skipped: {exc}")
                else:
                    print(f"  [warn] ACF plot skipped: {exc}")

    # ── 5. Inter-arrival time histogram ──────────────────────────────────
    # Exclude fill/execution rows (event_type 4, 5) — same fix as compute_metrics.
    msg_arrivals_plot = messages[~messages['event_type'].isin([4, 5])]
    iat      = ev.inter_arrival_time(msg_arrivals_plot)
    iat_pos  = iat[iat >= 0]
    iat_clp  = iat_pos[iat_pos < iat_pos.quantile(0.99)]
    fig, ax  = plt.subplots(figsize=(7, 4))
    ax.hist(iat_clp.values, bins=60, density=True,
            edgecolor='k', lw=0.3, alpha=0.75, color='mediumpurple')
    ax.set_title(f'{hdr} — Inter-arrival time (ms, ≤p99)')
    ax.set_xlabel('IAT (ms)')
    ax.set_ylabel('Density')
    plt.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, 'inter_arrival_time.png'), logger)

    # ── 6. L1 ask/bid volume over time ───────────────────────────────────
    l1vol = _safe(ev.l1_volume, messages, book, label='l1_volume_plot')
    if l1vol is not None:
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(l1vol.index, l1vol['ask_vol'], lw=0.5, alpha=0.75,
                label='Ask L1', color='crimson')
        ax.plot(l1vol.index, l1vol['bid_vol'], lw=0.5, alpha=0.75,
                label='Bid L1', color='royalblue')
        ax.legend(fontsize=9)
        ax.set_title(f'{hdr} — L1 Ask / Bid Volume')
        ax.set_xlabel('Time')
        ax.set_ylabel('Shares')
        fig.autofmt_xdate()
        plt.tight_layout()
        _save_fig(fig, os.path.join(plots_dir, 'l1_volume.png'), logger)

    # ── 7. Volume per minute (executed) ──────────────────────────────────
    vpm = _safe(ev.volume_per_minute, messages, book, label='volume_per_minute_plot')
    if vpm is not None and len(vpm) > 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(len(vpm)), vpm.values, width=0.8,
               color='teal', edgecolor='k', lw=0.4)
        ax.set_title(f'{hdr} — Executed volume per minute')
        ax.set_xlabel('Minute index')
        ax.set_ylabel('Shares / min')
        plt.tight_layout()
        _save_fig(fig, os.path.join(plots_dir, 'volume_per_minute.png'), logger)

    # ── 8. Orderbook imbalance ─────────────────────────────────────────────
    obi = _safe(ev.orderbook_imbalance, messages, book, label='ob_imbalance_plot')
    if obi is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(obi.index, obi.values, lw=0.4, alpha=0.7, color='saddlebrown')
        axes[0].axhline(0, color='k', lw=0.8, linestyle='--')
        axes[0].set_title('OB Imbalance over time')
        axes[0].set_xlabel('Time')
        fig.autofmt_xdate()

        obi_c = obi.dropna()
        axes[1].hist(obi_c.values, bins=60, density=True,
                     edgecolor='k', lw=0.3, alpha=0.75, color='saddlebrown')
        axes[1].axvline(0, color='k', lw=0.8, linestyle='--')
        axes[1].set_title('OB Imbalance distribution')
        axes[1].set_xlabel('Imbalance  (bid−ask)/(bid+ask)')
        axes[1].set_ylabel('Density')

        fig.suptitle(f'{hdr} — Orderbook Imbalance (L1)', fontweight='bold')
        plt.tight_layout()
        _save_fig(fig, os.path.join(plots_dir, 'orderbook_imbalance.png'), logger)

    # ── 9. Orderflow imbalance (rolling 100) ─────────────────────────────
    ofi = _safe(ev.orderflow_imbalance, messages, book, 100,
                label='orderflow_imbalance_plot')
    if ofi is not None and len(ofi) > 0:
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(range(len(ofi)), ofi.values, lw=0.5, alpha=0.8, color='darkolivegreen')
        ax.axhline(0, color='k', lw=0.8, linestyle='--')
        ax.set_title(f'{hdr} — Orderflow Imbalance (rolling 100-event window)')
        ax.set_xlabel('Event index')
        ax.set_ylabel('OFI')
        plt.tight_layout()
        _save_fig(fig, os.path.join(plots_dir, 'orderflow_imbalance.png'), logger)

    # ── 10. Event type counts ─────────────────────────────────────────────
    vc = messages['event_type'].value_counts().sort_index()
    _ETYPE_LABEL = {
        1: 'New Limit\n(1)',
        2: 'Cancel\n(2)',
        3: 'Modify\n(3)',
        4: 'Execute\n(4)',
        5: 'Hidden\n(5)',
        7: 'Halt\n(7)',
    }
    labels = [_ETYPE_LABEL.get(int(t), f'Type {t}') for t in vc.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vc.values, edgecolor='k', lw=0.5,
                  color=['#4C72B0', '#DD8452', '#55A868', '#C44E52',
                          '#8172B3', '#937860'][:len(vc)])
    ax.bar_label(bars, fmt='%d', padding=3, fontsize=9)
    ax.set_title(f'{hdr} — Event type distribution')
    ax.set_ylabel('Count')
    plt.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, 'event_type_counts.png'), logger)

    if logger:
        logger.info(
            "Skipped time-to-fill/time-to-cancel plots: requires true order-ID linkage (not available in synthetic IDs)."
        )

    # ── summary panel: mid-price + spread + OBI stacked ──────────────────
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
        axes[0].plot(mid_cny.index, mid_cny.values, lw=0.8, color='steelblue')
        axes[0].set_ylabel('Mid (CNY)')
        axes[0].set_title(f'{hdr} — Overview', fontweight='bold')
        fig.autofmt_xdate()

        axes[1].plot(sprd_cny.index, sprd_cny.values, lw=0.4, alpha=0.7,
                     color='darkorange')
        axes[1].set_ylabel('Spread (CNY)')

        if obi is not None:
            axes[2].fill_between(range(len(obi)), obi.values, 0,
                                 where=obi.values >= 0, color='royalblue', alpha=0.5,
                                 label='Bid-heavy')
            axes[2].fill_between(range(len(obi)), obi.values, 0,
                                 where=obi.values < 0,  color='crimson', alpha=0.5,
                                 label='Ask-heavy')
            axes[2].axhline(0, color='k', lw=0.8)
            axes[2].legend(fontsize=8)
        else:
            axes[2].set_visible(False)
        axes[2].set_ylabel('OBI')
        axes[2].set_xlabel('Event index')

        plt.tight_layout()
        _save_fig(fig, os.path.join(plots_dir, 'overview_panel.png'), logger)
    except Exception as exc:
        if logger:
            logger.warning(f"overview panel skipped: {exc}")
        else:
            print(f"  [warn] overview panel skipped: {exc}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a generated LOBSTER stream (no reference data required).'
    )
    parser.add_argument(
        'exp_dir', nargs='?', default=None,
        help='Path to experiment directory (default: latest fixed_start_* in saved_LOB_stream/)'
    )
    parser.add_argument(
        '--real_ref_dir',
        default=None,
        help='Path to real reference LOBSTER experiment directory for comparative metrics.',
    )
    args = parser.parse_args()

    if args.exp_dir:
        exp_dir = os.path.abspath(args.exp_dir)
    else:
        base = os.path.abspath(os.path.join(_HERE, '..', 'saved_LOB_stream'))
        candidates = sorted(glob.glob(os.path.join(base, 'fixed_start_*')))
        if not candidates:
            raise FileNotFoundError(
                f"No fixed_start_* experiment directories found under {base}"
            )
        exp_dir = candidates[-1]

    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Setup logger
    log_file = os.path.join(exp_dir, 'eval.log')
    logger = _setup_logger(log_file)

    logger.info("=" * 70)
    logger.info("eval_generated_stream.py — stylized fact evaluation")
    logger.info("=" * 70)
    logger.info(f"Evaluating: {exp_dir}")

    logger.info("\n── loading data ───────────────────────────────────────")
    messages, book, notes = load_lobster_pair(exp_dir, logger)

    ref_messages = None
    ref_book = None
    if args.real_ref_dir:
        real_ref_dir = os.path.abspath(args.real_ref_dir)
        logger.info(f"Loading real reference LOBSTER data from {real_ref_dir}")
        ref_messages, ref_book, _ = load_lobster_pair(real_ref_dir, logger)

    logger.info("\n── computing metrics ──────────────────────────────────")
    metrics = compute_metrics(messages, book, logger, ref_messages=ref_messages, ref_book=ref_book)

    logger.info("\n── capability split (strict) ─────────────────────────")
    for name in metrics['capability_report']['computed_generated_only_metrics']:
        logger.info("  [computed] %s", name)
    for name in metrics['capability_report']['computed_reference_comparison_metrics']:
        logger.info("  [computed: real-vs-generated] %s", name)
    for name in metrics['capability_report']['not_attempted_requires_real_reference_lob_data']:
        logger.info("  [skipped: needs real reference LOB data] %s", name)
    for name, reason in metrics['capability_report'].get('failed_reference_comparison_metrics', {}).items():
        logger.info("  [failed: real-vs-generated] %s | %s", name, reason)
    for name in metrics['capability_report']['not_attempted_requires_true_order_id_linkage']:
        logger.info("  [skipped: needs true order-ID linkage] %s", name)

    logger.info("\n── generating plots ───────────────────────────────────")
    make_plots(messages, book, plots_dir, notes, logger)

    summary_path = os.path.join(exp_dir, 'metrics_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(_to_serializable(metrics), fh, indent=2)
    logger.info(f"Wrote metrics summary: {summary_path}")

    # print a concise text summary
    logger.info("\n── key metrics ────────────────────────────────────────")
    if metrics.get('spread_ticks'):
        s = metrics['spread_ticks']
        logger.info(
            f"  spread        mean={s['mean']:.2f} ticks(0.01)  median={s['median']:.2f} ticks(0.01)"
        )
    if metrics.get('volatility_1min') is not None:
        logger.info(f"  vol (1min)    {metrics['volatility_1min']:.6f}")
    if metrics.get('returns_1min'):
        r = metrics['returns_1min']
        logger.info(f"  returns       skew={r['skew']:.3f}  kurt={r['kurtosis']:.3f}")
    if metrics.get('iat_ms'):
        i = metrics['iat_ms']
        logger.info(f"  IAT           mean={i['mean']:.1f}ms  median={i['median']:.1f}ms")
    if metrics.get('ob_imbalance'):
        o = metrics['ob_imbalance']
        logger.info(f"  OB imbalance  mean={o['mean']:.4f}  std={o['std']:.4f}")

    _log_all_metrics_summary(metrics, logger)
    
    logger.info("\n" + "=" * 70)
    logger.info("Evaluation complete")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
