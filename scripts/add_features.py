"""
AlphaHelix 特征工程扩展

新增特征：
1. 技术指标：RSI, MACD
2. 特征交互：mom*vol, value*quality
3. 行业相对：行业内 rank, 行业广度
4. 滞后特征：过去收益, 过去 IC
5. Regime：市场宽度, 波动率状态

用法：
    python scripts/add_features.py --input memory/dataset/features_h10_composite.parquet --output memory/dataset/features_h10_enhanced.parquet
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def add_rsi(df, period=14):
    """RSI 相对强弱指标"""
    # 用 mom_5 作为价格变化的代理
    if "mom_5" in df.columns:
        delta = df["mom_5"]
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(period, min_periods=1).mean()
        avg_loss = loss.rolling(period, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    """MACD 指标"""
    if "mom_20" in df.columns:
        # 用 mom_20 作为价格的代理
        ema_fast = df["mom_20"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["mom_20"].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_atr(df, period=14):
    """ATR 平均真实波幅"""
    if "volatility_20" in df.columns:
        # 用波动率作为 ATR 的代理
        df["atr"] = df["volatility_20"].rolling(period, min_periods=1).mean()
    return df


def add_feature_interactions(df):
    """特征交互"""
    if "mom_20" in df.columns and "volatility_20" in df.columns:
        df["mom_x_vol"] = df["mom_20"] * df["volatility_20"]
    if "roe" in df.columns and "profit_growth" in df.columns:
        df["quality_x_growth"] = df["roe"] * df["profit_growth"]
    if "dv_ratio" in df.columns and "total_mv" in df.columns:
        df["value_x_size"] = df["dv_ratio"] * df["total_mv"]
    if "net_mf_ratio" in df.columns and "volatility_20" in df.columns:
        df["flow_x_vol"] = df["net_mf_ratio"] * df["volatility_20"]
    return df


def add_sector_rank(df):
    """行业内 rank"""
    if "date" in df.columns and "industry" in df.columns:
        for col in ["mom_20", "volatility_20", "total_mv", "roe"]:
            if col in df.columns:
                df[f"{col}_sector_rank"] = df.groupby(["date", "industry"])[col].rank(pct=True, na_option="keep")
    return df


def add_lagged_features(df):
    """滞后特征：过去收益"""
    if "date" in df.columns and "ts_code" in df.columns:
        df = df.sort_values(["ts_code", "date"])
        for col in ["mom_20", "volatility_20", "net_mf_ratio"]:
            if col in df.columns:
                df[f"{col}_lag1"] = df.groupby("ts_code")[col].shift(1)
                df[f"{col}_lag2"] = df.groupby("ts_code")[col].shift(2)
    return df


def add_regime_features(df):
    """市场 Regime 特征"""
    if "date" in df.columns:
        # 市场宽度：所有股票 mom_20 > 0 的比例
        market_breadth = df.groupby("date")["mom_20"].apply(lambda x: (x > 0).mean())
        df["market_breadth"] = df["date"].map(market_breadth)
        
        # 市场波动率：所有股票 volatility_20 的均值
        market_vol = df.groupby("date")["volatility_20"].mean()
        df["market_volatility"] = df["date"].map(market_vol)
        
        # 波动率状态
        df["vol_regime"] = (df["market_volatility"] > df["market_volatility"].median()).astype(int)
    return df


def add_all_features(df):
    """添加所有新特征"""
    df = df.copy()
    
    # 技术指标
    df = add_rsi(df)
    df = add_macd(df)
    df = add_atr(df)
    
    # 特征交互
    df = add_feature_interactions(df)
    
    # 行业相对
    df = add_sector_rank(df)
    
    # 滞后特征
    df = add_lagged_features(df)
    
    # Regime
    df = add_regime_features(df)
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入 parquet 文件")
    parser.add_argument("--output", required=True, help="输出 parquet 文件")
    args = parser.parse_args()
    
    df = pd.read_parquet(args.input)
    print(f"[add_features] Input: {len(df)} rows, {len(df.columns)} cols")
    
    df = add_all_features(df)
    
    new_cols = [c for c in df.columns if c not in pd.read_parquet(args.input).columns]
    print(f"[add_features] Added {len(new_cols)} new features: {new_cols}")
    
    df.to_parquet(args.output, index=False)
    print(f"[add_features] Output: {len(df)} rows, {len(df.columns)} cols")


if __name__ == "__main__":
    main()
