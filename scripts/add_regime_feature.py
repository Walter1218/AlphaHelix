"""
AlphaHelix Regime 特征

将市场 regime 分类结果作为特征输入模型：
- regime_bull: 是否牛市
- regime_bear: 是否熊市
- regime_sideways: 是否震荡
- regime_strength: regime 强度
- regime_confidence: regime 置信度

用法：
    python scripts/add_regime_feature.py --input memory/dataset/features_h10_selected_v2.parquet --output memory/dataset/features_h10_regime.parquet
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def detect_regime(df):
    """检测市场 regime"""
    # 计算市场指标
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
    
    # Regime 强度
    market_stats['regime_strength'] = 0.0
    market_stats.loc[market_stats['regime'] == 'bull', 'regime_strength'] = (
        market_stats.loc[market_stats['regime'] == 'bull', 'avg_excess'] / 
        market_stats.loc[market_stats['regime'] == 'bull', 'avg_median'].abs()
    )
    market_stats.loc[market_stats['regime'] == 'bear', 'regime_strength'] = (
        market_stats.loc[market_stats['regime'] == 'bear', 'avg_median'].abs() / 
        market_stats.loc[market_stats['regime'] == 'bear', 'avg_excess'].abs()
    )
    
    # Regime 置信度（基于 market_breadth）
    market_stats['regime_confidence'] = market_stats['win_rate'].abs()
    
    return market_stats


def add_regime_feature(df):
    """添加 regime 特征"""
    df = df.copy()
    
    # 检测 regime
    market_stats = detect_regime(df)
    
    # 添加 regime 特征
    df['regime'] = df['date'].map(market_stats['regime']).fillna('sideways')
    df['regime_strength'] = df['date'].map(market_stats['regime_strength']).fillna(0)
    df['regime_confidence'] = df['date'].map(market_stats['regime_confidence']).fillna(0.5)
    
    # One-hot 编码
    df['regime_bull'] = (df['regime'] == 'bull').astype(int)
    df['regime_bear'] = (df['regime'] == 'bear').astype(int)
    df['regime_sideways'] = (df['regime'] == 'sideways').astype(int)
    
    # Regime 数值编码
    regime_map = {'bull': 1, 'sideways': 0, 'bear': -1}
    df['regime_numeric'] = df['regime'].map(regime_map).fillna(0)
    
    # 市场状态指标
    df['market_avg_excess'] = df['date'].map(market_stats['avg_excess']).fillna(0)
    df['market_vol_excess'] = df['date'].map(market_stats['vol_excess']).fillna(0)
    df['market_win_rate'] = df['date'].map(market_stats['win_rate']).fillna(0.5)
    
    # 交互特征
    df['regime_x_mom'] = df['regime_numeric'] * df['mom_20']
    df['regime_x_vol'] = df['regime_numeric'] * df['volatility_20']
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    args = parser.parse_args()
    
    df = pd.read_parquet(args.input)
    print(f"[regime] Input: {len(df)} rows, {len(df.columns)} cols")
    
    df = add_regime_feature(df)
    
    # 列出新增特征
    original_cols = set(pd.read_parquet(args.input).columns)
    new_cols = [c for c in df.columns if c not in original_cols]
    print(f"[regime] Added {len(new_cols)} new features: {new_cols}")
    
    df.to_parquet(args.output, index=False)
    print(f"[regime] Output: {len(df)} rows, {len(df.columns)} cols")


if __name__ == "__main__":
    main()
