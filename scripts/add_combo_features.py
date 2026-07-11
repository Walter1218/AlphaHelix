"""
AlphaHelix 组合特征与查找特征

1. Combo Features：因子交互组合
2. Lookup Features：预计算的查找表特征
3. Regularization：正则化手段

用法：
    python scripts/add_combo_features.py --input memory/dataset/features_h10_selected_v2.parquet --output memory/dataset/features_h10_combo.parquet
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def add_combo_features(df):
    """添加组合特征（因子交互）"""
    df = df.copy()
    
    # 1. 动量 × 波动率
    if 'mom_20' in df.columns and 'volatility_20' in df.columns:
        df['combo_mom_vol'] = df['mom_20'] * df['volatility_20']
    
    # 2. 价值 × 质量
    if 'dv_ratio' in df.columns and 'roe' in df.columns:
        df['combo_value_quality'] = df['dv_ratio'] * df['roe']
    
    # 3. 动量 × 资金流
    if 'mom_20' in df.columns and 'net_mf_ratio' in df.columns:
        df['combo_mom_flow'] = df['mom_20'] * df['net_mf_ratio']
    
    # 4. 波动率 × 市值
    if 'volatility_20' in df.columns and 'total_mv' in df.columns:
        df['combo_vol_size'] = df['volatility_20'] * df['total_mv']
    
    # 5. 成长 × 价值
    if 'profit_growth' in df.columns and 'dv_ratio' in df.columns:
        df['combo_growth_value'] = df['profit_growth'] * df['dv_ratio']
    
    # 6. 资金流 × 波动率
    if 'net_mf_ratio' in df.columns and 'volatility_20' in df.columns:
        df['combo_flow_vol'] = df['net_mf_ratio'] * df['volatility_20']
    
    # 7. 动量反转组合
    if 'mom_20' in df.columns and 'reversal_score' in df.columns:
        df['combo_mom_reversal'] = df['mom_20'] * df['reversal_score']
    
    # 8. 估值动量组合
    if 'dv_ratio' in df.columns and 'mom_60' in df.columns:
        df['combo_value_mom'] = df['dv_ratio'] * df['mom_60']
    
    # 9. 质量波动率组合
    if 'roe' in df.columns and 'volatility_20' in df.columns:
        df['combo_quality_vol'] = df['roe'] * df['volatility_20']
    
    # 10. 资金流估值组合
    if 'net_mf_ratio' in df.columns and 'total_mv' in df.columns:
        df['combo_flow_size'] = df['net_mf_ratio'] * df['total_mv']
    
    return df


def add_lookup_features(df):
    """添加查找特征（预计算的截面统计）"""
    df = df.copy()
    
    # 1. 市场宽度：上涨股票比例
    if 'mom_20' in df.columns:
        market_breadth = df.groupby('date')['mom_20'].apply(lambda x: (x > 0.5).mean())
        df['lookup_market_breadth'] = df['date'].map(market_breadth)
    
    # 2. 市场波动率：平均波动率
    if 'volatility_20' in df.columns:
        market_vol = df.groupby('date')['volatility_20'].mean()
        df['lookup_market_vol'] = df['date'].map(market_vol)
    
    # 3. 市场动量：平均动量
    if 'mom_20' in df.columns:
        market_mom = df.groupby('date')['mom_20'].mean()
        df['lookup_market_mom'] = df['date'].map(market_mom)
    
    # 4. 市场估值：平均估值
    if 'dv_ratio' in df.columns:
        market_dv = df.groupby('date')['dv_ratio'].mean()
        df['lookup_market_dv'] = df['date'].map(market_dv)
    
    # 5. 市场资金流：平均资金流
    if 'net_mf_ratio' in df.columns:
        market_mf = df.groupby('date')['net_mf_ratio'].mean()
        df['lookup_market_mf'] = df['date'].map(market_mf)
    
    # 6. 个股相对市场
    if 'mom_20' in df.columns:
        df['lookup_mom_vs_market'] = df['mom_20'] - df['lookup_market_mom']
    
    if 'volatility_20' in df.columns:
        df['lookup_vol_vs_market'] = df['volatility_20'] - df['lookup_market_vol']
    
    return df


def add_rolling_features(df, periods=[5, 10, 20]):
    """添加滚动特征"""
    df = df.copy()
    
    # 按股票和日期排序
    df = df.sort_values(['ts_code', 'date'])
    
    for period in periods:
        # 滚动收益
        if 'mom_5' in df.columns:
            df[f'rolling_mom_{period}d'] = df.groupby('ts_code')['mom_5'].transform(
                lambda x: x.rolling(period, min_periods=1).mean()
            )
        
        # 滚动波动率
        if 'volatility_20' in df.columns:
            df[f'rolling_vol_{period}d'] = df.groupby('ts_code')['volatility_20'].transform(
                lambda x: x.rolling(period, min_periods=1).mean()
            )
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    args = parser.parse_args()
    
    df = pd.read_parquet(args.input)
    print(f"[combo] Input: {len(df)} rows, {len(df.columns)} cols")
    
    df = add_combo_features(df)
    df = add_lookup_features(df)
    df = add_rolling_features(df)
    
    # 列出新增特征
    original_cols = set(pd.read_parquet(args.input).columns)
    new_cols = [c for c in df.columns if c not in original_cols]
    print(f"[combo] Added {len(new_cols)} new features: {new_cols}")
    
    df.to_parquet(args.output, index=False)
    print(f"[combo] Output: {len(df)} rows, {len(df.columns)} cols")


if __name__ == "__main__":
    main()
