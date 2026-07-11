"""
AlphaHelix 独立特征

创建更独立的特征，避免高度相关的线性组合：
1. 比率特征：risk-adjusted momentum
2. 差异特征：momentum acceleration
3. 非线性特征：log/sqrt 变换
4. 截面特征：sector rank

用法：
    python scripts/add_independent_features.py --input memory/dataset/features_h10_selected_v2.parquet --output memory/dataset/features_h10_independent.parquet
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def add_ratio_features(df):
    """添加比率特征（非线性组合）"""
    df = df.copy()
    
    # 1. 风险调整动量
    if 'mom_20' in df.columns and 'volatility_20' in df.columns:
        df['ratio_mom_vol'] = df['mom_20'] / (df['volatility_20'] + 1e-9)
    
    # 2. 价值风险比
    if 'dv_ratio' in df.columns and 'volatility_20' in df.columns:
        df['ratio_value_vol'] = df['dv_ratio'] / (df['volatility_20'] + 1e-9)
    
    # 3. 资金流动量比
    if 'net_mf_ratio' in df.columns and 'mom_20' in df.columns:
        df['ratio_flow_mom'] = df['net_mf_ratio'] / (df['mom_20'].abs() + 1e-9)
    
    # 4. 质量波动率比
    if 'roe' in df.columns and 'volatility_20' in df.columns:
        df['ratio_quality_vol'] = df['roe'] / (df['volatility_20'] + 1e-9)
    
    # 5. 增长估值比
    if 'profit_growth' in df.columns and 'dv_ratio' in df.columns:
        df['ratio_growth_value'] = df['profit_growth'] / (df['dv_ratio'] + 1e-9)
    
    return df


def add_difference_features(df):
    """添加差异特征（趋势变化）"""
    df = df.copy()
    
    # 1. 短期 vs 长期动量
    if 'mom_5' in df.columns and 'mom_20' in df.columns:
        df['diff_mom_short_long'] = df['mom_5'] - df['mom_20']
    
    if 'mom_20' in df.columns and 'mom_60' in df.columns:
        df['diff_mom_med_long'] = df['mom_20'] - df['mom_60']
    
    # 2. 波动率变化
    if 'volatility_20' in df.columns and 'volatility_20' in df.columns:
        df['diff_vol_trend'] = df['volatility_20'] - df['volatility_20'].rolling(10, min_periods=1).mean()
    
    # 3. 资金流变化
    if 'net_mf_ratio' in df.columns:
        df['diff_flow_trend'] = df['net_mf_ratio'] - df.groupby('date')['net_mf_ratio'].transform('mean')
    
    # 4. 相对强度变化
    if 'relative_strength' in df.columns:
        df['diff_rel_strength'] = df['relative_strength'] - df.groupby('date')['relative_strength'].transform('mean')
    
    return df


def add_nonlinear_features(df):
    """添加非线性变换特征"""
    df = df.copy()
    
    # 1. 对数市值
    if 'total_mv' in df.columns:
        df['log_total_mv'] = np.log1p(df['total_mv'])
    
    # 2. 平方根波动率
    if 'volatility_20' in df.columns:
        df['sqrt_volatility'] = np.sqrt(df['volatility_20'])
    
    # 3. 对数动量
    if 'mom_20' in df.columns:
        df['log_mom_20'] = np.log1p(df['mom_20'].clip(lower=0))
    
    # 4. 交互对数
    if 'roe' in df.columns and 'dv_ratio' in df.columns:
        df['log_quality_value'] = np.log1p(df['roe'] * df['dv_ratio'])
    
    return df


def add_cross_sectional_features(df):
    """添加截面特征（行业 rank）"""
    df = df.copy()
    
    # 1. 行业内动量 rank
    if 'mom_20' in df.columns and 'industry' in df.columns:
        df['cs_mom_rank'] = df.groupby(['date', 'industry'])['mom_20'].rank(pct=True)
    
    # 2. 行业内波动率 rank
    if 'volatility_20' in df.columns and 'industry' in df.columns:
        df['cs_vol_rank'] = df.groupby(['date', 'industry'])['volatility_20'].rank(pct=True)
    
    # 3. 行业内估值 rank
    if 'dv_ratio' in df.columns and 'industry' in df.columns:
        df['cs_value_rank'] = df.groupby(['date', 'industry'])['dv_ratio'].rank(pct=True)
    
    # 4. 行业内资金流 rank
    if 'net_mf_ratio' in df.columns and 'industry' in df.columns:
        df['cs_flow_rank'] = df.groupby(['date', 'industry'])['net_mf_ratio'].rank(pct=True)
    
    return df


def add_all_independent_features(df):
    """添加所有独立特征"""
    df = df.copy()
    df = add_ratio_features(df)
    df = add_difference_features(df)
    df = add_nonlinear_features(df)
    df = add_cross_sectional_features(df)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    args = parser.parse_args()
    
    df = pd.read_parquet(args.input)
    print(f"[independent] Input: {len(df)} rows, {len(df.columns)} cols")
    
    df = add_all_independent_features(df)
    
    # 列出新增特征
    original_cols = set(pd.read_parquet(args.input).columns)
    new_cols = [c for c in df.columns if c not in original_cols]
    print(f"[independent] Added {len(new_cols)} new features: {new_cols}")
    
    df.to_parquet(args.output, index=False)
    print(f"[independent] Output: {len(df)} rows, {len(df.columns)} cols")


if __name__ == "__main__":
    main()
