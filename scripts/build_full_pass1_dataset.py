"""
全市场 PASS1 候选池数据集构建（不进入 PASS2）

目标：解决 GBDT 训练集只在 top-80 候选池里学习的问题。
方法：对每个再平衡日，执行 screen.py 的 PASS1（流动性、波动率过滤），
      保留所有幸存者（通过设置 AH_PASS1_TOP_K=9999），不再做耗时的 PASS2 财务/资金流抓取。
      用 PASS1 特征 + pass1_score 训练 GBDT，看是否能在全市场分布上泛化。

输出：memory/dataset/features_h{H}_pass1_full.parquet
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# 必须在 import screen 前设置，让 PASS1_TOP_K 足够大
os.environ["AH_PASS1_TOP_K"] = "9999"
os.environ["AH_UNIVERSE_SAMPLE"] = "9999"
os.environ["AH_SKIP_ST_CHECK"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import get_trade_calendar, get_trade_date_after
from screen import build_universe, pass1_screen, STRATEGIES, classify_regime, regime_to_strategy
from evaluate import get_close_price
from feature_engineering import build_numeric_features

DATASET_DIR = Path("memory/dataset")


def get_rebalance_dates(start_date: str, end_date: str, delta_days: int) -> list:
    cal = get_trade_calendar("SSE", start_date, end_date)
    cal = cal[cal["is_open"].astype(int) == 1].copy()
    dates = sorted(cal["cal_date"].astype(str).tolist())
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


def _strategy_config(date: str, strategy: str):
    actual = strategy
    if strategy == "regime":
        info = classify_regime(date)
        actual = regime_to_strategy(info["regime"], available=list(STRATEGIES.keys()))
    return dict(STRATEGIES[actual])


def build_pass1_full_dataset(start_date: str, end_date: str, horizon: int,
                             delta_days: int = 5, strategy: str = "regime",
                             output_path: str = None):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = DATASET_DIR / f"features_h{horizon}_pass1_full.parquet"
    else:
        output_path = Path(output_path)

    rebalance = get_rebalance_dates(start_date, end_date, delta_days)
    print(f"[build_pass1_full] {len(rebalance)} rebalance dates, strategy={strategy}")

    rows = []
    for idx, t in enumerate(rebalance, 1):
        print(f"[build_pass1_full] [{idx}/{len(rebalance)}] {t} ...", end=" ", flush=True)
        try:
            config = _strategy_config(t, strategy)
            df_universe = build_universe(t)
            if df_universe.empty:
                print("empty universe")
                continue

            df_pass1 = pass1_screen(df_universe, t, config["pass1"])
            if df_pass1 is None or df_pass1.empty:
                print("empty pass1")
                continue

            feature_cols = ["mom_5", "mom_20", "mom_60", "mom_120",
                            "risk_adj_mom", "relative_strength",
                            "ep", "bp", "sp", "dv_ratio",
                            "sector_momentum", "relative_to_sector", "sector_breadth",
                            "reversal_score", "amount_ratio_5d", "volatility_20", "liquidity", "total_mv",
                            "pass1_score"]
            # 兼容性：pass1_screen 输出列名是 avg_amount_20，重命名为 liquidity
            df_pass1 = df_pass1.rename(columns={"avg_amount_20": "liquidity"})
            # 计算估值指标
            for col in ["pe", "pb", "ps", "dv_ratio", "total_mv"]:
                if col in df_pass1.columns:
                    df_pass1[col] = pd.to_numeric(df_pass1[col], errors="coerce")
            df_pass1["ep"] = 1.0 / df_pass1["pe"]
            df_pass1["bp"] = 1.0 / df_pass1["pb"]
            df_pass1["sp"] = 1.0 / df_pass1["ps"]
            df_pass1 = build_numeric_features(df_pass1, feature_cols=feature_cols, neutralize=True, rank=True, winsorize=True)

            t_exit = get_trade_date_after(t, days=horizon)
            bench_t = get_close_price("000300.SH", t)
            bench_exit = get_close_price("000300.SH", t_exit)
            bench_ret = bench_exit / bench_t - 1

            n_valid = 0
            for _, row in df_pass1.iterrows():
                code = row["ts_code"]
                try:
                    p_t = get_close_price(code, t)
                    p_exit = get_close_price(code, t_exit)
                    if np.isnan(p_t) or np.isnan(p_exit):
                        continue
                except Exception:
                    continue
                stock_ret = p_exit / p_t - 1
                excess_ret = stock_ret - bench_ret
                rec = {
                    "date": t,
                    "exit_date": t_exit,
                    "ts_code": code,
                    "stock_return": stock_ret,
                    "benchmark_return": bench_ret,
                    "excess_return": excess_ret,
                    "industry": row.get("industry"),
                }
                for col in feature_cols:
                    rec[col] = row.get(col)
                rows.append(rec)
                n_valid += 1

            print(f"{n_valid} samples")
        except Exception as e:
            print(f"ERROR: {e}")

        if idx % 10 == 0 and rows:
            pd.DataFrame(rows).to_parquet(output_path, index=False)
            print(f"[build_pass1_full] Saved {len(rows)} rows to {output_path}")

    if rows:
        pd.DataFrame(rows).to_parquet(output_path, index=False)
        print(f"[build_pass1_full] Saved {len(rows)} rows to {output_path}")
    else:
        print("[build_pass1_full] No rows generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--delta-days", type=int, default=5)
    parser.add_argument("--strategy", default="regime")
    parser.add_argument("--output", default=None, help="输出文件路径")
    args = parser.parse_args()

    build_pass1_full_dataset(args.start, args.end, args.horizon,
                             delta_days=args.delta_days, strategy=args.strategy,
                             output_path=args.output)
