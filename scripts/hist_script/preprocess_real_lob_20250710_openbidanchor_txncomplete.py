import argparse
import json
import os
import sys
import time

import joblib

PROJECT_ROOT = "/finance_ML/zhanghaohan/stock_language_model"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utility.sim_helper_unified import process_lob_data_real_flow_open_anchor_txn_complete


PRICE_BIN_NUM = 26
QTY_BIN_NUM = 26
INTERVAL_BIN_NUM = 12
ANCHOR_TIME = "09:31:00"


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess real LOB flow with a fixed 09:31 L1 bid anchor and explicit transaction-complete events."
    )
    parser.add_argument("--stock", default="000617_XSHE", help="Stock code in *_XSHE format.")
    parser.add_argument("--day", default="20250710", help="Trading day in YYYYMMDD format.")
    parser.add_argument(
        "--split-cancel-sides",
        action="store_true",
        help="Use 6-way side vocabulary: separate bid-cancel (97) and ask-cancel (98) instead of single cancel (99). Requires n_side=6.",
    )
    parser.add_argument(
        "--output-parent-subdir",
        default="",
        help=(
            "Optional folder name under saved_LOB_stream/processed_real_flow/ for joblib/json outputs "
            "(e.g. 20250709_openbidanchor_txncomplete). Empty = write directly under processed_real_flow/."
        ),
    )
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M")
    day = args.day
    stock = str(args.stock).strip()
    stock_tag = stock.replace("_", "")

    n_side = 6 if args.split_cancel_sides else 5
    side_to_bin = (
        {49: 0, 50: 1, 97: 2, 98: 3, 129: 4, 130: 5}
        if args.split_cancel_sides
        else {49: 0, 50: 1, 99: 2, 129: 3, 130: 4}
    )
    flow_tag = "openbidanchor_txncomplete_splitcancel" if args.split_cancel_sides else "openbidanchor_txncomplete"

    root_dir = "/finance_ML/zhanghaohan/LOB_data"
    liquidity_mask_dir = "/finance_ML/zhanghaohan/LOB_data/misc_data/AVG_AMT_3M_7_1D_8390e8742c5e.csv"

    lob_day_folder = os.path.join(root_dir, day)
    order_post_dir = os.path.join(lob_day_folder, "mdl_6_33_0.csv")
    lob_snap_dir = os.path.join(lob_day_folder, "mdl_6_28_0.csv")
    order_transac_dir = os.path.join(lob_day_folder, "mdl_6_36_0.csv")

    print(f"\n================ {day} | {stock} | {flow_tag} ================")
    df_day, bin_record = process_lob_data_real_flow_open_anchor_txn_complete(
        order_post_dir=order_post_dir,
        lob_snap_dir=lob_snap_dir,
        order_transac_dir=order_transac_dir,
        liquidity_mask_dir=liquidity_mask_dir,
        selected_stocks=[stock],
        filter_bo=True,
        date_num_str=day,
        anchor_time=ANCHOR_TIME,
        price_bin_num=PRICE_BIN_NUM,
        qty_bin_num=QTY_BIN_NUM,
        interval_bin_num=INTERVAL_BIN_NUM,
        n_side=n_side,
        side_to_bin=side_to_bin,
        return_bin_record=True,
        split_cancel_sides=bool(args.split_cancel_sides),
    )

    df_day = df_day.copy()
    df_day["TradeDate"] = day

    base_processed = os.path.join(PROJECT_ROOT, "saved_LOB_stream", "processed_real_flow")
    subdir = str(args.output_parent_subdir).strip().strip("/").replace("..", "")
    output_dir = os.path.join(base_processed, subdir) if subdir else base_processed
    os.makedirs(output_dir, exist_ok=True)

    cache_path = os.path.join(
        output_dir,
        f"final_result_for_merge_realflow_{flow_tag}_{day}_{stock_tag}_{ts}.joblib",
    )
    joblib.dump(df_day, cache_path, compress=3)

    bin_record_path = os.path.join(
        output_dir,
        f"bin_record_realflow_{flow_tag}_{day}_{stock_tag}_{ts}.json",
    )
    with open(bin_record_path, "w", encoding="utf-8") as f:
        json.dump(bin_record, f, indent=2, ensure_ascii=False)

    anchor_meta = (bin_record or {}).get("price_anchor_by_stock", {}).get(stock, {})
    summary = {
        "day": day,
        "stock": stock,
        "regime_label": f"{day}_{flow_tag}",
        "output_parent_subdir": subdir if subdir else None,
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
        "price_anchor": {
            "mode": (bin_record or {}).get("price_anchor_mode"),
            "time": ANCHOR_TIME,
            "stock_anchor": anchor_meta,
        },
        "event_schema": {
            "n_side": n_side,
            "side_to_bin": side_to_bin,
            "split_cancel_sides": bool(args.split_cancel_sides),
            "token_side_mapping": (bin_record or {}).get("token_side_mapping", {}),
            "transaction_complete_mode": (bin_record or {}).get("transaction_complete_mode"),
        },
        "binning": {
            "price_bin_num": PRICE_BIN_NUM,
            "qty_bin_num": QTY_BIN_NUM,
            "interval_bin_num": INTERVAL_BIN_NUM,
            "n_side": n_side,
            "tokenizable_rows": int(df_day["tokenizable_event"].fillna(False).astype(bool).sum()) if "tokenizable_event" in df_day.columns else 0,
            "event_semantic_counts": df_day["EventSemantic"].value_counts(dropna=False).to_dict() if "EventSemantic" in df_day.columns else {},
            "side_counts": {str(int(k)): int(v) for k, v in df_day["Side"].dropna().astype(int).value_counts().sort_index().items()} if "Side" in df_day.columns else {},
        },
    }

    summary_path = os.path.join(
        output_dir,
        f"final_result_for_merge_realflow_{flow_tag}_{day}_{stock_tag}_{ts}.json",
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[{day}] stock: {stock}")
    print(f"[{day}] rows: {len(df_day)}")
    print(f"[{day}] stocks: {df_day['SecurityID'].nunique()}")
    print(f"[anchor] {anchor_meta}")
    print(f"[schema] {(bin_record or {}).get('token_side_mapping', {})}")
    print(f"[cache] df -> {cache_path}")
    print(f"[cache] bin_record -> {bin_record_path}")
    print(f"[cache] summary -> {summary_path}")


if __name__ == "__main__":
    main()