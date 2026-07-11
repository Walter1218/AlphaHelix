"""
AlphaHelix Regime 风险管理

用 Regime 检测做仓位管理，而非选股：
- Bull: 满仓
- Sideways: 正常仓位
- Bear: 半仓或空仓

用法：
    python scripts/regime_risk_management.py --pred-path memory/predictions/predictions_h10_stacking_5model.parquet
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_trainer import load_dataset

PRED_DIR = Path("memory/predictions")


def detect_regime(df):
    """检测市场 regime"""
    market_stats = df.groupby('date').agg({
        'excess_return': ['mean', 'std'],
    })
    market_stats.columns = ['avg_excess', 'vol_excess']
    market_stats['avg_median'] = market_stats['avg_excess'].rolling(60, min_periods=1).median()
    market_stats['vol_median'] = market_stats['vol_excess'].rolling(60, min_periods=1).median()
    market_stats['regime'] = 'sideways'
    market_stats.loc[(market_stats['avg_excess'] > market_stats['avg_median'] * 1.5) & 
                     (market_stats['vol_excess'] < market_stats['vol_median'] * 0.8), 'regime'] = 'bull'
    market_stats.loc[(market_stats['avg_excess'] < market_stats['avg_median'] * 0.5) & 
                     (market_stats['vol_excess'] > market_stats['vol_median'] * 1.2), 'regime'] = 'bear'
    return market_stats


def regime_position_scale(regime):
    """根据 regime 决定仓位比例"""
    if regime == 'bull':
        return 1.0  # 满仓
    elif regime == 'bear':
        return 0.5  # 半仓
    else:
        return 0.8  # 正常仓位


def walk_forward_regime_risk(predictions, feature_df, horizon=10):
    """Walk-forward regime 风险管理"""
    # 检测 regime
    market_stats = detect_regime(feature_df)
    
    predictions['date'] = pd.to_datetime(predictions['date'])
    predictions['year_month'] = predictions['date'].dt.to_period('M')
    months = sorted(predictions['year_month'].unique())
    
    all_preds = []
    
    for i, test_month in enumerate(months):
        test_data = predictions[predictions['year_month'] == test_month].copy()
        if test_data.empty:
            continue
        
        # 获取当前 regime
        current_date = test_data['date'].iloc[0]
        if current_date in market_stats.index:
            current_regime = market_stats.loc[current_date, 'regime']
        else:
            current_regime = 'sideways'
        
        # 根据 regime 决定仓位
        scale = regime_position_scale(current_regime)
        
        # 调整预测分数
        test_data['predicted'] = test_data['predicted'] * scale
        test_data['regime'] = current_regime
        test_data['position_scale'] = scale
        
        all_preds.append(test_data)
    
    if not all_preds:
        return pd.DataFrame()
    
    return pd.concat(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    
    print("[regime_risk] Loading predictions...")
    predictions = pd.read_parquet(args.pred_path)
    print(f"  Loaded {len(predictions)} rows, {predictions['date'].nunique()} dates")
    
    print("[regime_risk] Loading feature data...")
    feature_df = load_dataset(args.horizon, "memory/dataset/features_h10_composite_fixed.parquet")
    
    print("[regime_risk] Running regime risk management...")
    result = walk_forward_regime_risk(predictions, feature_df, args.horizon)
    
    if result.empty:
        print("[regime_risk] No predictions generated")
        return
    
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"predictions_h{args.horizon}_regime_risk.parquet"
    output_path = PRED_DIR / output_name
    result.to_parquet(output_path, index=False)
    print(f"\n[regime_risk] Saved {len(result)} predictions to {output_path}")
    
    # 统计
    print("\n=== Regime Distribution ===")
    print(result['regime'].value_counts())
    print("\n=== Position Scale Distribution ===")
    print(result['position_scale'].value_counts().sort_index())


if __name__ == "__main__":
    main()
