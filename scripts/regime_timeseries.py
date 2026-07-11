"""
AlphaHelix 时序特征 Regime 检测

用时序特征检测市场状态，然后为每个状态选择最佳模型。

用法：
    python scripts/regime_timeseries.py --horizon 10
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_trainer import load_dataset, get_feature_cols

PRED_DIR = Path("memory/predictions")


def detect_regime_from_timeseries(df):
    """用时序特征检测市场状态"""
    if 'date' not in df.columns:
        return df
    
    # 计算市场级别的时序指标
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


def select_model_by_regime(regime, regime_performance):
    """根据 regime 选择模型"""
    if regime in regime_performance:
        return regime_performance[regime]
    return 'full_46_equal'


def walk_forward_regime_timeseries(predictions, feature_df, horizon=10):
    """用时序特征做 regime 检测的 walk-forward"""
    market_stats = detect_regime_from_timeseries(feature_df)
    
    # 合并所有预测
    unified = None
    for model_name, pred_df in predictions.items():
        pred_df = pred_df.copy()
        pred_df['model_name'] = model_name
        if unified is None:
            unified = pred_df
        else:
            unified = pd.concat([unified, pred_df], ignore_index=True)
    
    if unified is None:
        return pd.DataFrame()
    
    unified['date'] = pd.to_datetime(unified['date'])
    unified['year_month'] = unified['date'].dt.to_period('M')
    months = sorted(unified['year_month'].unique())
    
    # 预计算每个 regime 的最佳模型
    regime_performance = {}
    for regime in ['bull', 'bear', 'sideways']:
        best_model = None
        best_ic = -999
        
        for model_name in predictions.keys():
            ics = []
            for month in months[-12:]:  # 用最近12个月
                month_data = unified[(unified['year_month'] == month) & 
                                     (unified['model_name'] == model_name)]
                month_date = month_data['date'].iloc[0] if not month_data.empty else None
                
                if month_date and month_date in market_stats.index:
                    month_regime = market_stats.loc[month_date, 'regime']
                    if month_regime == regime and len(month_data) > 10:
                        ic = month_data['predicted'].corr(month_data['excess_return'], method='spearman')
                        if not np.isnan(ic):
                            ics.append(ic)
            
            if ics:
                avg_ic = np.mean(ics)
                if avg_ic > best_ic:
                    best_ic = avg_ic
                    best_model = model_name
        
        if best_model:
            regime_performance[regime] = best_model
            print(f"  {regime}: best model = {best_model} (IC = {best_ic:.4f})")
    
    # 逐月预测
    all_preds = []
    for i, test_month in enumerate(months):
        test_data = unified[unified['year_month'] == test_month]
        if test_data.empty:
            continue
        
        current_date = test_data['date'].iloc[0]
        if current_date in market_stats.index:
            current_regime = market_stats.loc[current_date, 'regime']
        else:
            current_regime = 'sideways'
        
        best_model = select_model_by_regime(current_regime, regime_performance)
        model_pred = test_data[test_data['model_name'] == best_model]
        
        if not model_pred.empty:
            pred = model_pred[['date', 'ts_code', 'stock_return', 'benchmark_return', 
                               'excess_return', 'industry', 'predicted']].copy()
            pred['regime'] = current_regime
            pred['model_name'] = best_model
            all_preds.append(pred)
    
    if not all_preds:
        return pd.DataFrame()
    
    return pd.concat(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    
    print("[regime_timeseries] Loading predictions...")
    predictions = {}
    configs = {
        'full_46_equal': f'predictions_h{args.horizon}_full_18m.parquet',
        'pruned_36_equal': f'predictions_h{args.horizon}_pruned_18m.parquet',
        'original_30_equal': f'predictions_h{args.horizon}_original_30_fixed.parquet',
    }
    
    for name, filename in configs.items():
        path = PRED_DIR / filename
        if path.exists():
            df = pd.read_parquet(path)
            df['date'] = pd.to_datetime(df['date'])
            predictions[name] = df
            print(f"  Loaded {name}: {len(df)} rows")
    
    if not predictions:
        print("[regime_timeseries] No predictions loaded")
        return
    
    print("\n[regime_timeseries] Loading feature data...")
    feature_df = load_dataset(args.horizon, "memory/dataset/features_h10_composite_fixed.parquet")
    
    print("\n[regime_timeseries] Running regime-adaptive walk-forward...")
    result = walk_forward_regime_timeseries(predictions, feature_df, args.horizon)
    
    if result.empty:
        print("[regime_timeseries] No predictions generated")
        return
    
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"predictions_h{args.horizon}_regime_timeseries.parquet"
    output_path = PRED_DIR / output_name
    result.to_parquet(output_path, index=False)
    print(f"\n[regime_timeseries] Saved {len(result)} predictions to {output_path}")
    
    print("\n=== Regime Statistics ===")
    print(result['regime'].value_counts())
    print("\n=== Model Usage ===")
    print(result['model_name'].value_counts())


if __name__ == "__main__":
    main()
