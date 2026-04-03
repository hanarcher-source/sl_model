import json
import os
import sys
import time

import joblib

PROJECT_ROOT = "/finance_ML/zhanghaohan/stock_language_model"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utility.sim_helper_unified import process_lob_data_real_flow


def main():
    ts = time.strftime("%Y%m%d_%H%M")
    day = "20250710"

    root_dir = "/finance_ML/zhanghaohan/LOB_data"
    liquidity_mask_dir = "/finance_ML/zhanghaohan/LOB_data/misc_data/AVG_AMT_3M_7_1D_8390e8742c5e.csv"

    lob_day_folder = os.path.join(root_dir, day)
    order_post_dir = os.path.join(lob_day_folder, "mdl_6_33_0.csv")
    lob_snap_dir = os.path.join(lob_day_folder, "mdl_6_28_0.csv")
    order_transac_dir = os.path.join(lob_day_folder, "mdl_6_36_0.csv")

    print(f"\\n================ {day} ================")
    df_day = process_lob_data_real_flow(
        order_post_dir=order_post_dir,
        lob_snap_dir=lob_snap_dir,
        order_transac_dir=order_transac_dir,
        liquidity_mask_dir=liquidity_mask_dir,
        selected_stocks=["000617_XSHE"],
        filter_bo=True,
        date_num_str=day,
    )

    df_day = df_day.copy()
    df_day["TradeDate"] = day

    output_dir = os.path.join(PROJECT_ROOT, "saved_LOB_stream", "processed_real_flow")
    os.makedirs(output_dir, exist_ok=True)

    cache_path = os.path.join(output_dir, f"final_result_for_merge_realflow_{day}_{ts}.joblib")
    joblib.dump(df_day, cache_path, compress=3)

    summary = {
        "day": day,
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
        },
    }

    summary_path = os.path.join(output_dir, f"final_result_for_merge_realflow_{day}_{ts}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[{day}] rows: {len(df_day)}")
    print(f"[{day}] stocks: {df_day['SecurityID'].nunique()}")
    print(f"[cache] df -> {cache_path}")
    print(f"[cache] summary -> {summary_path}")


if __name__ == "__main__":
    main()
