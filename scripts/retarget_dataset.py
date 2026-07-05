"""
对已有 dataset 重新计算目标收益（持有期 horizon）。

用途：复用 screen.py 生成的特征截面，只换标签长度，避免重复跑完整 build_dataset。
例如把 features_h10 转成 features_h20。
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import get_trade_date_after, tushare_call


def retarget_dataset(input_path: str, output_path: str, horizon: int):
    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")

    dates = sorted(df["date"].unique())
    # 预计算每个 T 对应的 T+H 日期
    exit_dates = {d: get_trade_date_after(d, days=horizon) for d in dates}

    # 预加载所有需要的收盘价：按日期加载截面，比逐行查询快
    all_dates = set(dates) | set(exit_dates.values())
    close_cache = {}
    ts_codes = set(df["ts_code"].unique())

    for d in sorted(all_dates):
        try:
            cs = tushare_call("daily", {"trade_date": d})
            if not cs.empty:
                cs["ts_code"] = cs["ts_code"].astype(str)
                cs["close"] = pd.to_numeric(cs["close"], errors="coerce")
                for _, row in cs.iterrows():
                    if row["ts_code"] in ts_codes:
                        close_cache[(row["ts_code"], d)] = float(row["close"])
        except Exception:
            pass
        try:
            idx = tushare_call("index_daily", {"ts_code": "000300.SH", "trade_date": d})
            if not idx.empty:
                close_cache[("000300.SH", d)] = float(pd.to_numeric(idx.iloc[0]["close"], errors="coerce"))
        except Exception:
            pass

    def compute_row(row):
        d = row["date"]
        d_exit = exit_dates.get(d)
        code = row["ts_code"]
        p0 = close_cache.get((code, d), np.nan)
        p1 = close_cache.get((code, d_exit), np.nan)
        b0 = close_cache.get(("000300.SH", d), np.nan)
        b1 = close_cache.get(("000300.SH", d_exit), np.nan)
        if np.isnan(p0) or np.isnan(p1) or p0 <= 0 or np.isnan(b0) or np.isnan(b1) or b0 <= 0:
            return pd.Series([np.nan, np.nan, np.nan])
        stock_ret = p1 / p0 - 1
        bench_ret = b1 / b0 - 1
        return pd.Series([stock_ret, bench_ret, stock_ret - bench_ret])

    df[["stock_return", "benchmark_return", "excess_return"]] = df.apply(compute_row, axis=1)

    before = len(df)
    df = df.dropna(subset=["stock_return", "benchmark_return", "excess_return"])
    after = len(df)
    if after < before:
        print(f"[retarget] Dropped {before - after} rows with missing prices")

    # 更新 exit_date
    df["exit_date"] = df["date"].map(exit_dates)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[retarget] Saved {after} rows to {output_path}")
    print(f"[retarget] New horizon={horizon}, avg excess={df['excess_return'].mean():.4f}, "
          f"std={df['excess_return'].std():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="memory/dataset/features_h10_composite.parquet")
    parser.add_argument("--output", default=None)
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()

    output = args.output or f"memory/dataset/features_h{args.horizon}_composite.parquet"
    retarget_dataset(args.input, output, args.horizon)


if __name__ == "__main__":
    main()
