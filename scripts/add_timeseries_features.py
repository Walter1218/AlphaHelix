"""
AlphaHelix 时序特征扩展

新增基于时间序列的特征：
1. 滚动收益：过去 N 天的收益
2. 滚动波动率：过去 N 天的波动率
3. 动量模式：收益趋势
4. 成交量模式：成交量变化

用法：
    python scripts/add_timeseries_features.py --input memory/dataset/features_h10_selected_v2.parquet --output memory/dataset/features_h10_timeseries.parquet
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def add_timeseries_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加时序特征"""
    df = df.copy()
    
    # 按股票和日期排序
    df = df.sort_values(["ts_code", "date"])
    
    # 1. 滚动收益（过去 N 天的收益）
    for period in [5, 10, 20]:
        # 这里用 mom_5 作为日收益的代理
        if "mom_5" in df.columns:
            df[f"rolling_return_{period}d"] = df.groupby("ts_code")["mom_5"].transform(
                lambda x: x.rolling(period, min_periods=1).mean()
            )
    
    # 2. 滚动波动率
    if "volatility_20" in df.columns:
        df["volatility_10d"] = df.groupby("ts_code")["volatility_20"].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )
        df["volatility_5d"] = df.groupby("ts_code")["volatility_20"].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
    
    # 3. 动量趋势（动量加速度）
    if "mom_20" in df.columns and "mom_60" in df.columns:
        df["momentum_accel"] = df["mom_20"] - df["mom_60"]
    
    if "mom_5" in df.columns and "mom_20" in df.columns:
        df["momentum_short_long"] = df["mom_5"] - df["mom_20"]
    
    # 4. 成交量模式
    if "amount_ratio_5d" in df.columns:
        df["volume_trend"] = df.groupby("ts_code")["amount_ratio_5d"].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )
    
    # 5. 风险调整收益趋势
    if "risk_adj_mom" in df.columns:
        df["risk_adj_mom_10d"] = df.groupby("ts_code")["risk_adj_mom"].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )
    
    # 6. 资金流趋势
    if "net_mf_ratio" in df.columns:
        df["net_mf_ratio_5d"] = df.groupby("ts_code")["net_mf_ratio"].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    args = parser.parse_args()
    
    df = pd.read_parquet(args.input)
    print(f"[timeseries] Input: {len(df)} rows, {len(df.columns)} cols")
    
    df = add_timeseries_features(df)
    
    # 列出新增特征
    original_cols = set(pd.read_parquet(args.input).columns)
    new_cols = [c for c in df.columns if c not in original_cols]
    print(f"[timeseries] Added {len(new_cols)} new features: {new_cols}")
    
    df.to_parquet(args.output, index=False)
    print(f"[timeseries] Output: {len(df)} rows, {len(df.columns)} cols")


if __name__ == "__main__":
    main()
