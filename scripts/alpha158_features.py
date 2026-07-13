"""
Alpha158 风格特征集

参考 Microsoft Qlib 的 Alpha158 特征集，适配我们的数据格式。

用法：
    python alpha158_features.py --input memory/dataset/features_h10_full.parquet --output memory/dataset/features_alpha158.parquet
"""
import sys
import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")


class Alpha158Features:
    """
    Alpha158 风格特征集
    
    参考 Qlib 的 Alpha158，适配我们的数据格式。
    我们没有原始 OHLCV，但有动量、波动率等预计算特征。
    """
    
    # 关键特征（类似 Qlib 的价格和成交量）
    PRICE_FEATURES = ['mom_5', 'mom_20', 'mom_60', 'mom_120']
    VOLATILITY_FEATURES = ['volatility_20']
    VOLUME_FEATURES = ['amount_ratio_5d', 'liquidity']
    FUNDAMENTAL_FEATURES = ['roe', 'dv_ratio', 'ep', 'bp', 'sp', 'total_mv']
    
    @staticmethod
    def get_rolling_features(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
        """
        计算滚动特征（类似 Qlib 的 ROC/MA/STD）
        
        Args:
            df: 数据框
            windows: 滚动窗口列表
        
        Returns:
            包含滚动特征的数据框
        """
        if windows is None:
            windows = [5, 10, 20, 30, 60]
        
        result = df.copy()
        
        # 对每个关键特征计算滚动统计
        key_features = (
            Alpha158Features.PRICE_FEATURES + 
            Alpha158Features.VOLATILITY_FEATURES +
            Alpha158Features.VOLUME_FEATURES
        )
        
        for feature in key_features:
            if feature not in df.columns:
                continue
            
            for d in windows:
                # ROC (Rate of Change) - 类似 Qlib
                # 我们用特征值的变化率
                result[f'{feature}_ROC{d}'] = (
                    df[feature].shift(d) / (df[feature].abs() + 1e-6)
                )
                
                # MA (Moving Average)
                result[f'{feature}_MA{d}'] = (
                    df[feature].rolling(d, min_periods=1).mean()
                )
                
                # STD (Standard Deviation)
                result[f'{feature}_STD{d}'] = (
                    df[feature].rolling(d, min_periods=1).std()
                )
                
                # MAX (Maximum)
                result[f'{feature}_MAX{d}'] = (
                    df[feature].rolling(d, min_periods=1).max()
                )
                
                # MIN (Minimum)
                result[f'{feature}_MIN{d}'] = (
                    df[feature].rolling(d, min_periods=1).min()
                )
                
                # RSV (Relative Strength Value)
                rolling_min = df[feature].rolling(d, min_periods=1).min()
                rolling_max = df[feature].rolling(d, min_periods=1).max()
                result[f'{feature}_RSV{d}'] = (
                    (df[feature] - rolling_min) / (rolling_max - rolling_min + 1e-6)
                )
        
        return result
    
    @staticmethod
    def get_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算截面特征（类似 Qlib 的 RANK）
        
        Args:
            df: 数据框
        
        Returns:
            包含截面特征的数据框
        """
        result = df.copy()
        
        key_features = (
            Alpha158Features.PRICE_FEATURES + 
            Alpha158Features.VOLATILITY_FEATURES +
            Alpha158Features.FUNDAMENTAL_FEATURES
        )
        
        for feature in key_features:
            if feature not in df.columns:
                continue
            
            # 截面排名（每日排名百分位）
            result[f'{feature}_RANK'] = df.groupby('date')[feature].rank(pct=True)
        
        return result
    
    @staticmethod
    def get_industry_neutral_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算行业中性特征
        
        Args:
            df: 数据框
        
        Returns:
            包含行业中性特征的数据框
        """
        result = df.copy()
        
        key_features = ['mom_20', 'roe', 'volatility_20', 'dv_ratio']
        
        for feature in key_features:
            if feature not in df.columns:
                continue
            
            # 行业内排名
            result[f'{feature}_IND_RANK'] = df.groupby(['date', 'industry'])[feature].rank(pct=True)
            
            # 行业内 z-score
            def industry_zscore(group):
                if len(group) < 3:
                    return pd.Series(0, index=group.index)
                mean = group[feature].mean()
                std = group[feature].std()
                return (group[feature] - mean) / (std + 1e-6)
            
            result[f'{feature}_IND_Z'] = df.groupby(['date', 'industry'], group_keys=False).apply(
                industry_zscore
            )
        
        return result
    
    @staticmethod
    def get_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算交互特征
        
        Args:
            df: 数据框
        
        Returns:
            包含交互特征的数据框
        """
        result = df.copy()
        
        # 价格-波动率交互
        if 'mom_20' in df.columns and 'volatility_20' in df.columns:
            result['MOM_VOL_RATIO'] = df['mom_20'] / (df['volatility_20'] + 1e-6)
            result['MOM_VOL_PRODUCT'] = df['mom_20'] * df['volatility_20']
        
        # 短期-长期动量交互
        if 'mom_5' in df.columns and 'mom_20' in df.columns:
            result['MOM_5_20_RATIO'] = df['mom_5'] / (df['mom_20'].abs() + 1e-6)
            result['MOM_ACCEL'] = df['mom_5'] - df['mom_20']
        
        # 基本面-动量交互
        for feature in ['roe', 'dv_ratio', 'ep', 'bp']:
            if feature in df.columns and 'mom_20' in df.columns:
                result[f'{feature}_MOM20'] = df[feature] * df['mom_20']
        
        # 资金流动量交互
        if 'net_mf_ratio' in df.columns and 'mom_20' in df.columns:
            result['MF_MOM20'] = df['net_mf_ratio'] * df['mom_20']
        
        # 质量因子
        if 'roe' in df.columns and 'net_mf_ratio' in df.columns:
            result['QUALITY_FLOW'] = df['roe'] * df['net_mf_ratio']
        
        return result
    
    @staticmethod
    def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        构建所有 Alpha158 风格特征
        
        Args:
            df: 原始数据框
        
        Returns:
            包含所有特征的数据框
        """
        print('构建滚动特征...')
        result = Alpha158Features.get_rolling_features(df)
        
        print('构建截面特征...')
        result = Alpha158Features.get_cross_sectional_features(result)
        
        print('构建行业中性特征...')
        result = Alpha158Features.get_industry_neutral_features(result)
        
        print('构建交互特征...')
        result = Alpha158Features.get_interaction_features(result)
        
        return result


def main():
    parser = argparse.ArgumentParser(description="Alpha158 特征集构建")
    parser.add_argument("--input", type=str, default="memory/dataset/features_h10_full.parquet",
                        help="输入文件路径")
    parser.add_argument("--output", type=str, default="memory/dataset/features_alpha158.parquet",
                        help="输出文件路径")
    args = parser.parse_args()
    
    # 加载数据
    print(f'加载数据: {args.input}')
    df = pd.read_parquet(args.input)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f'原始特征数: {len(df.columns)}')
    print(f'数据行数: {len(df)}')
    
    # 构建特征
    result = Alpha158Features.build_all_features(df)
    
    print(f'新特征数: {len(result.columns)}')
    
    # 保存
    result.to_parquet(args.output, index=False)
    print(f'保存到: {args.output}')


if __name__ == "__main__":
    main()
