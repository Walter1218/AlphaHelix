"""
AlphaHelix 自适应特征

根据市场状态（regime）创建自适应特征：
1. Regime-conditional 特征：不同市场下同一特征有不同含义
2. 交互特征：regime × 关键因子
3. 相对特征：相对于市场基准的特征

用法：
    python scripts/add_adaptive_features.py --input memory/dataset/features_h10_selected_v2.parquet --output memory/dataset/features_h10_adaptive.parquet
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def detect_market_regime(df):
    """检测市场状态"""
    market_stats = df.groupby('date').agg({
        'excess_return': ['mean', 'std', lambda x: (x > 0).mean()],
        'volatility_20': 'mean',
    })
    market_stats.columns = ['avg_excess', 'vol_excess', 'win_rate', 'market_vol']
    
    # 滚动统计
    market_stats['avg_ma'] = market_stats['avg_excess'].rolling(20, min_periods=1).mean()
    market_stats['vol_ma'] = market_stats['vol_excess'].rolling(20, min_periods=1).mean()
    market_stats['avg_median'] = market_stats['avg_excess'].rolling(60, min_periods=1).median()
    market_stats['vol_median'] = market_stats['vol_excess'].rolling(60, min_periods=1).median()
    
    # Regime 判断
    market_stats['regime'] = 'sideways'
    market_stats.loc[(market_stats['avg_excess'] > market_stats['avg_median'] * 1.5) & 
                     (market_stats['vol_excess'] < market_stats['vol_median'] * 0.8), 'regime'] = 'bull'
    market_stats.loc[(market_stats['avg_excess'] < market_stats['avg_median'] * 0.5) & 
                     (market_stats['vol_excess'] > market_stats['vol_median'] * 1.2), 'regime'] = 'bear'
    
    return market_stats


def add_adaptive_features(df):
    """添加自适应特征"""
    df = df.copy()
    
    # 检测市场状态
    market_stats = detect_market_regime(df)
    df['regime'] = df['date'].map(market_stats['regime']).fillna('sideways')
    
    # 1. Regime-conditional 特征
    # 在不同 regime 下，同一特征有不同含义
    for col in ['mom_20', 'volatility_20', 'total_mv']:
        if col in df.columns:
            # 创建 regime-specific 特征
            df[f'{col}_bull'] = df[col] * (df['regime'] == 'bull').astype(float)
            df[f'{col}_bear'] = df[col] * (df['regime'] == 'bear').astype(float)
            df[f'{col}_sideways'] = df[col] * (df['regime'] == 'sideways').astype(float)
    
    # 2. 交互特征：regime × 关键因子
    if 'mom_20' in df.columns and 'volatility_20' in df.columns:
        df['mom_vol_bull'] = df['mom_20'] * df['volatility_20'] * (df['regime'] == 'bull').astype(float)
        df['mom_vol_bear'] = df['mom_20'] * df['volatility_20'] * (df['regime'] == 'bear').astype(float)
    
    # 3. 相对特征：相对于市场基准
    if 'mom_20' in df.columns:
        market_mom = df.groupby('date')['mom_20'].transform('mean')
        df['mom_20_vs_market'] = df['mom_20'] - market_mom
    
    if 'volatility_20' in df.columns:
        market_vol = df.groupby('date')['volatility_20'].transform('mean')
        df['vol_20_vs_market'] = df['volatility_20'] - market_vol
    
    # 4. Regime 编码（数值）
    regime_map = {'bull': 1, 'sideways': 0, 'bear': -1}
    df['regime_numeric'] = df['regime'].map(regime_map).fillna(0)
    
    # 5. 市场状态强度
    market_stats['regime_strength'] = 0.0
    market_stats.loc[market_stats['regime'] == 'bull', 'regime_strength'] = (
        market_stats.loc[market_stats['regime'] == 'bull', 'avg_excess'] / 
        market_stats.loc[market_stats['regime'] == 'bull', 'avg_median']
    )
    market_stats.loc[market_stats['regime'] == 'bear', 'regime_strength'] = (
        market_stats.loc[market_stats['regime'] == 'bear', 'avg_median'] / 
        market_stats.loc[market_stats['regime'] == 'bear', 'avg_excess']
    )
    df['regime_strength'] = df['date'].map(market_stats['regime_strength']).fillna(0)
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    args = parser.parse_args()
    
    df = pd.read_parquet(args.input)
    print(f"[adaptive] Input: {len(df)} rows, {len(df.columns)} cols")
    
    df = add_adaptive_features(df)
    
    # 列出新增特征
    original_cols = set(pd.read_parquet(args.input).columns)
    new_cols = [c for c in df.columns if c not in original_cols]
    print(f"[adaptive] Added {len(new_cols)} new features: {new_cols}")
    
    df.to_parquet(args.output, index=False)
    print(f"[adaptive] Output: {len(df)} rows, {len(df.columns)} cols")


if __name__ == "__main__":
    main()
