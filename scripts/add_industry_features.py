"""
AlphaHelix 行业级特征

引入个股对应行业的相关数据，作为特征让模型学习：
1. 行业动量/波动率
2. 行业资金流向
3. 行业排名
4. 个股 vs 行业

用法：
    python scripts/add_industry_features.py --input memory/dataset/features_h10_selected_v2.parquet --output memory/dataset/features_h10_industry.parquet
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def add_industry_features(df):
    """添加行业级特征（不含目标变量，避免数据泄露）"""
    df = df.copy()
    
    if 'industry' not in df.columns:
        print("Warning: no 'industry' column, skipping industry features")
        return df
    
    # 1. 行业波动率
    if 'volatility_20' in df.columns:
        df['ind_volatility'] = df.groupby(['date', 'industry'])['volatility_20'].transform('mean')
    
    # 2. 行业资金流
    if 'net_mf_ratio' in df.columns:
        df['ind_net_mf'] = df.groupby(['date', 'industry'])['net_mf_ratio'].transform('mean')
    
    # 3. 个股 vs 行业
    if 'volatility_20' in df.columns:
        df['vs_ind_vol'] = df['volatility_20'] - df['ind_volatility']
    
    if 'net_mf_ratio' in df.columns:
        df['vs_ind_mf'] = df['net_mf_ratio'] - df['ind_net_mf']
    
    # 4. 行业内排名
    if 'volatility_20' in df.columns:
        df['ind_rank_vol'] = df.groupby(['date', 'industry'])['volatility_20'].rank(pct=True)
    
    # 5. 行业动量
    if 'mom_20' in df.columns:
        df['ind_momentum'] = df.groupby(['date', 'industry'])['mom_20'].transform('mean')
        df['vs_ind_momentum'] = df['mom_20'] - df['ind_momentum']
    
    # 6. 行业估值
    if 'dv_ratio' in df.columns:
        df['ind_value'] = df.groupby(['date', 'industry'])['dv_ratio'].transform('mean')
        df['vs_ind_value'] = df['dv_ratio'] - df['ind_value']
    
    # 7. 行业 size
    if 'total_mv' in df.columns:
        df['ind_size'] = df.groupby(['date', 'industry'])['total_mv'].transform('mean')
        df['vs_ind_size'] = df['total_mv'] - df['ind_size']
    
    # 8. 行业 roe
    if 'roe' in df.columns:
        df['ind_roe'] = df.groupby(['date', 'industry'])['roe'].transform('mean')
        df['vs_ind_roe'] = df['roe'] - df['ind_roe']
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    args = parser.parse_args()
    
    df = pd.read_parquet(args.input)
    print(f"[industry] Input: {len(df)} rows, {len(df.columns)} cols")
    
    df = add_industry_features(df)
    
    # 列出新增特征
    original_cols = set(pd.read_parquet(args.input).columns)
    new_cols = [c for c in df.columns if c not in original_cols]
    print(f"[industry] Added {len(new_cols)} new features: {new_cols}")
    
    df.to_parquet(args.output, index=False)
    print(f"[industry] Output: {len(df)} rows, {len(df.columns)} cols")


if __name__ == "__main__":
    main()
