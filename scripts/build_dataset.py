"""
AlphaHelix 训练数据集构建

对指定区间内的每个再平衡日，调用 screen.py 生成 Pass2 候选池特征，
并计算未来 H 日的个股收益与相对基准的超额收益，保存为模型训练数据集。

输出：memory/dataset/features_h{H}.parquet
"""
import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import get_trade_calendar, get_trade_date_after
from screen import screen
from evaluate import get_close_price
from feature_engineering import build_numeric_features

DATASET_DIR = Path("memory/dataset")


def get_rebalance_dates(start_date: str, end_date: str, delta_days: int) -> list:
    cal = get_trade_calendar("SSE", start_date, end_date)
    cal = cal[cal["is_open"].astype(int) == 1].copy()
    dates = sorted(cal["cal_date"].astype(str).tolist())
    if not dates:
        return []

    rebalance = []
    for i in range(0, len(dates), delta_days):
        t = dates[i]
        try:
            t_next = get_trade_date_after(t, days=delta_days)
        except Exception:
            break
        if t_next > end_date:
            break
        rebalance.append(t)
    return rebalance


def build_dataset(start_date: str, end_date: str, horizon: int,
                  strategy: str = "regime", delta_days: int = 5,
                  universe_size: int = 200, skip_st_check: bool = True):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATASET_DIR / f"features_h{horizon}.parquet"

    rebalance_dates = get_rebalance_dates(start_date, end_date, delta_days)
    print(f"[build_dataset] Building dataset for horizon={horizon}, {len(rebalance_dates)} dates")

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        processed = set(existing["date"].astype(str).unique())
        print(f"[build_dataset] Found existing dataset with {len(processed)} dates")
    else:
        existing = pd.DataFrame()
        processed = set()

    rows = []
    for idx, t in enumerate(rebalance_dates, 1):
        if t in processed:
            continue
        print(f"[build_dataset] [{idx}/{len(rebalance_dates)}] {t} ...", end=" ", flush=True)
        try:
            _, df_pass2 = screen(
                t, strategy,
                top_n=999,
                return_full=True,
                max_positions=999,
            )
            if df_pass2 is None or df_pass2.empty:
                print("empty")
                continue

            df_feat = build_numeric_features(df_pass2)
            feature_cols = [c for c in df_feat.columns if c in [
                "mom_5", "mom_20", "mom_60", "mom_120",
                "risk_adj_mom", "relative_strength",
                "ep", "bp", "sp", "dv_ratio",
                "roe", "revenue_growth", "profit_growth", "ocf_growth",
                "net_mf_ratio", "net_mf_divergence",
                "sector_momentum", "relative_to_sector", "sector_breadth",
                "forecast_type_score", "forecast_pchange_mid", "express_diluted_roe",
                "reversal_score", "amount_ratio_5d", "volatility_20", "liquidity", "total_mv",
                # Phase 2
                "margin_total_balance", "margin_financing_ratio",
                "northbound_net", "northbound_net_5d",
                "top_list_flag", "top_list_net_amount", "top_list_amount_rate", "top_list_turnover_rate", "top_list_pct_change",
                "days_to_disclosure", "days_since_disclosure",
                # Composite factors
                "defensive_quality", "smart_money_per_risk", "quality_growth", "value_quality",
                "earnings_surprise_momentum", "growth_consistency", "risk_adj_momentum_20",
            ]]

            t_exit = get_trade_date_after(t, days=horizon)
            bench_t = get_close_price("000300.SH", t)
            bench_exit = get_close_price("000300.SH", t_exit)
            bench_ret = bench_exit / bench_t - 1

            n_valid = 0
            for _, row in df_feat.iterrows():
                ts_code = row.get("ts_code")
                if not ts_code:
                    continue
                try:
                    p_t = get_close_price(ts_code, t)
                    p_exit = get_close_price(ts_code, t_exit)
                    stock_ret = p_exit / p_t - 1
                    excess_ret = stock_ret - bench_ret
                except Exception:
                    continue

                rec = {
                    "date": t,
                    "exit_date": t_exit,
                    "ts_code": ts_code,
                    "stock_return": stock_ret,
                    "benchmark_return": bench_ret,
                    "excess_return": excess_ret,
                    "industry": row.get("industry"),
                }
                for col in feature_cols:
                    rec[col] = row[col]
                rows.append(rec)
                n_valid += 1

            print(f"{n_valid} samples")
        except Exception as e:
            print(f"ERROR: {e}")

        # 每 10 期保存一次增量结果
        if idx % 10 == 0 and rows:
            df_new = pd.DataFrame(rows)
            if not existing.empty:
                df_save = pd.concat([existing, df_new], ignore_index=True)
            else:
                df_save = df_new
            df_save.to_parquet(output_path, index=False)
            print(f"[build_dataset] Saved {len(df_save)} rows to {output_path}")
            existing = df_save
            processed = set(existing["date"].astype(str).unique())
            rows = []

    if rows:
        df_new = pd.DataFrame(rows)
        if not existing.empty:
            df_save = pd.concat([existing, df_new], ignore_index=True)
        else:
            df_save = df_new
        df_save.to_parquet(output_path, index=False)
        print(f"[build_dataset] Saved {len(df_save)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--delta-days", type=int, default=5)
    parser.add_argument("--strategy", default="regime")
    parser.add_argument("--universe-size", type=int, default=200)
    parser.add_argument("--skip-st-check", action="store_true", default=True)
    args = parser.parse_args()

    if args.universe_size is not None:
        os.environ["AH_UNIVERSE_SAMPLE"] = str(args.universe_size)
    if args.skip_st_check:
        os.environ["AH_SKIP_ST_CHECK"] = "1"
    os.environ["AH_BACKTEST_MODE"] = "1"

    build_dataset(
        args.start, args.end, args.horizon,
        strategy=args.strategy,
        delta_days=args.delta_days,
        universe_size=args.universe_size,
        skip_st_check=args.skip_st_check,
    )


if __name__ == "__main__":
    main()
