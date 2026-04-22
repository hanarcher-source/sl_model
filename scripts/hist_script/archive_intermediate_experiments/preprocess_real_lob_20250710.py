import argparse
import json
import os
import sys
import time

import joblib

PROJECT_ROOT = "/finance_ML/zhanghaohan/stock_language_model"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utility.sim_helper_unified import process_lob_data_real_flow


PRICE_BIN_NUM = 26
QTY_BIN_NUM = 26
INTERVAL_BIN_NUM = 12
N_SIDE = 3


def main():
    parser = argparse.ArgumentParser(description="Preprocess real LOB flow for a single stock on 20250710.")
    parser.add_argument(
        "--stock",
        default="000617_XSHE",
        help="Stock code in *_XSHE format. Default: 000617_XSHE",
    )
    parser.add_argument(
        "--day",
        default="20250710",
        help="Trading day in YYYYMMDD format. Default: 20250710",
    )
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M")
    day = args.day
    stock = str(args.stock).strip()
    stock_tag = stock.replace("_", "")

    root_dir = "/finance_ML/zhanghaohan/LOB_data"
    liquidity_mask_dir = "/finance_ML/zhanghaohan/LOB_data/misc_data/AVG_AMT_3M_7_1D_8390e8742c5e.csv"

    lob_day_folder = os.path.join(root_dir, day)
    order_post_dir = os.path.join(lob_day_folder, "mdl_6_33_0.csv")
    lob_snap_dir = os.path.join(lob_day_folder, "mdl_6_28_0.csv")
    order_transac_dir = os.path.join(lob_day_folder, "mdl_6_36_0.csv")
    existing_bin_record_path = os.path.join(
        lob_day_folder,
        "bin_record_vocab24336_multidaypool(03_10)_samp_20260318_1743.json",
    )

    print(f"\\n================ {day} | {stock} ================")
    df_day, bin_record = process_lob_data_real_flow(
        order_post_dir=order_post_dir,
        lob_snap_dir=lob_snap_dir,
        order_transac_dir=order_transac_dir,
        liquidity_mask_dir=liquidity_mask_dir,
        selected_stocks=[stock],
        filter_bo=True,
        date_num_str=day,
        price_bin_num=PRICE_BIN_NUM,
        qty_bin_num=QTY_BIN_NUM,
        interval_bin_num=INTERVAL_BIN_NUM,
        n_side=N_SIDE,
        existing_bin_record_path=existing_bin_record_path if os.path.exists(existing_bin_record_path) else None,
        return_bin_record=True,
    )

    df_day = df_day.copy()
    df_day["TradeDate"] = day

    output_dir = os.path.join(PROJECT_ROOT, "saved_LOB_stream", "processed_real_flow")
    os.makedirs(output_dir, exist_ok=True)

    cache_path = os.path.join(
        output_dir,
        f"final_result_for_merge_realflow_{day}_{stock_tag}_{ts}.joblib",
    )
    joblib.dump(df_day, cache_path, compress=3)

    bin_record_path = os.path.join(
        output_dir,
        f"bin_record_realflow_{day}_{stock_tag}_{ts}.json",
    )
    with open(bin_record_path, "w", encoding="utf-8") as f:
        json.dump(bin_record, f, indent=2, ensure_ascii=False)

    summary = {
        "day": day,
        "stock": stock,
        "created_at": ts,
        "rows": int(len(df_day)),
        "stocks": int(df_day["SecurityID"].nunique()) if len(df_day) > 0 else 0,
        "columns": list(df_day.columns),
        "source": {
            "order_post": order_post_dir,
            "lob_snap": lob_snap_dir,
            "order_transac": order_transac_dir,
            "liquidity_mask": liquidity_mask_dir,
        },
        "output": {
            "joblib": cache_path,
            "bin_record": bin_record_path,
        },
        "binning": {
            "price_bin_num": PRICE_BIN_NUM,
            "qty_bin_num": QTY_BIN_NUM,
            "interval_bin_num": INTERVAL_BIN_NUM,
            "n_side": N_SIDE,
            "existing_bin_record_path": existing_bin_record_path if os.path.exists(existing_bin_record_path) else None,
            "tokenizable_rows": int(df_day["tokenizable_event"].fillna(False).astype(bool).sum()) if "tokenizable_event" in df_day.columns else 0,
        },
    }

    summary_path = os.path.join(
        output_dir,
        f"final_result_for_merge_realflow_{day}_{stock_tag}_{ts}.json",
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[{day}] stock: {stock}")
    print(f"[{day}] rows: {len(df_day)}")
    print(f"[{day}] stocks: {df_day['SecurityID'].nunique()}")
    print(f"[cache] df -> {cache_path}")
    print(f"[cache] bin_record -> {bin_record_path}")
    print(f"[cache] summary -> {summary_path}")


if __name__ == "__main__":
    main()
