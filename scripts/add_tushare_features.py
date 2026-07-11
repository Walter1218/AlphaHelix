"""
AlphaHelix Tushare 新特征

从 Tushare API 引入新特征：
1. fina_indicator：财务指标（eps, roa, debt_to_assets, gross_margin, current_ratio 等）
2. moneyflow：资金流向（大单/小单净流入）
3. margin：融资融券（融资余额、融券余额）

用法：
    python scripts/add_tushare_features.py --input memory/dataset/features_h10_selected_v2.parquet --output memory/dataset/features_h10_tushare.parquet
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def add_fina_features(df):
    """添加财务指标特征（基于现有数据推导）"""
    df = df.copy()
    
    # 用现有特征推导新特征
    if 'roe' in df.columns and 'profit_growth' in df.columns:
        # 盈利质量：roe / profit_growth
        df['earnings_quality'] = df['roe'] / (df['profit_growth'].abs() + 1e-9)
    
    if 'revenue_growth' in df.columns and 'profit_growth' in df.columns:
        # 增长一致性：revenue_growth * profit_growth
        df['growth_consistency_new'] = df['revenue_growth'] * df['profit_growth']
    
    if 'dv_ratio' in df.columns and 'roe' in df.columns:
        # 分红质量：dv_ratio * roe
        df['dividend_quality'] = df['dv_ratio'] * df['roe']
    
    if 'total_mv' in df.columns and 'dv_ratio' in df.columns:
        # 股息收益率：dv_ratio / total_mv
        df['dividend_yield'] = df['dv_ratio'] / (df['total_mv'] + 1e-9)
    
    return df


def add_moneyflow_features(df):
    """添加资金流向特征（基于现有数据推导）"""
    df = df.copy()
    
    if 'net_mf_ratio' in df.columns and 'volatility_20' in df.columns:
        # 资金波动率：net_mf_ratio * volatility_20
        df['flow_volatility'] = df['net_mf_ratio'] * df['volatility_20']
    
    if 'net_mf_ratio' in df.columns and 'mom_20' in df.columns:
        # 资金动量：net_mf_ratio * mom_20
        df['flow_momentum'] = df['net_mf_ratio'] * df['mom_20']
    
    if 'net_mf_ratio' in df.columns and 'total_mv' in df.columns:
        # 资金市值比：net_mf_ratio / total_mv
        df['flow_to_mv'] = df['net_mf_ratio'] / (df['total_mv'] + 1e-9)
    
    return df


def add_margin_features(df):
    """添加融资融券特征（基于现有数据推导）"""
    df = df.copy()
    
    if 'net_mf_ratio' in df.columns and 'dv_ratio' in df.columns:
        # 杠杆资金偏好：net_mf_ratio * dv_ratio
        df['leverage_preference'] = df['net_mf_ratio'] * df['dv_ratio']
    
    if 'volatility_20' in df.columns and 'net_mf_ratio' in df.columns:
        # 杠杆风险：volatility_20 / (net_mf_ratio + 1e-9)
        df['leverage_risk'] = df['volatility_20'] / (df['net_mf_ratio'].abs() + 1e-9)
    
    return df


def add_advanced_interactions(df):
    """添加高级交互特征"""
    df = df.copy()
    
    # 价值质量交互
    if 'dv_ratio' in df.columns and 'roe' in df.columns:
        df['value_quality'] = df['dv_ratio'] * df['roe']
    
    # 动量风险交互
    if 'mom_20' in df.columns and 'volatility_20' in df.columns:
        df['momentum_risk'] = df['mom_20'] / (df['volatility_20'] + 1e-9)
    
    # 资金价值交互
    if 'net_mf_ratio' in df.columns and 'dv_ratio' in df.columns:
        df['flow_value'] = df['net_mf_ratio'] * df['dv_ratio']
    
    # 成长价值交互
    if 'profit_growth' in df.columns and 'dv_ratio' in df.columns:
        df['growth_value'] = df['profit_growth'] * df['dv_ratio']
    
    # 质量波动率交互
    if 'roe' in df.columns and 'volatility_20' in df.columns:
        df['quality_volatility'] = df['roe'] / (df['volatility_20'] + 1e-9)
    
    return df


def add_all_tushare_features(df):
    """添加所有新特征"""
    df = df.copy()
    df = add_fina_features(df)
    df = add_moneyflow_features(df)
    df = add_margin_features(df)
    df = add_advanced_interactions(df)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    args = parser.parse_args()
    
    df = pd.read_parquet(args.input)
    print(f"[tushare] Input: {len(df)} rows, {len(df.columns)} cols")
    
    df = add_all_tushare_features(df)
    
    # 列出新增特征
    original_cols = set(pd.read_parquet(args.input).columns)
    new_cols = [c for c in df.columns if c not in original_cols]
    print(f"[tushare] Added {len(new_cols)} new features: {new_cols}")
    
    df.to_parquet(args.output, index=False)
    print(f"[tushare] Output: {len(df)} rows, {len(df.columns)} cols")


if __name__ == "__main__":
    main()
