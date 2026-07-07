"""
AlphaRegime 自适应模型 V2

使用所有模型配置，根据市场 regime 动态选择最佳模型。

模型池：
- Full 46 特征 (equal/risk_parity)
- Pruned 36 特征 (equal/risk_parity)
- Original 30 特征 (equal/risk_parity)

Regime 检测：
- 市场宽度：mom_20 > 0 的股票比例
- 市场波动率：volatility_20 的均值
- 市场动量：mom_20 的均值

用法：
    python scripts/regime_adaptive_v2.py --horizon 10
"""
import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_trainer import load_dataset, get_feature_cols

PRED_DIR = Path("memory/predictions")


def load_all_predictions(horizon=10):
    """加载所有模型的预测结果"""
    predictions = {}
    
    # 定义所有模型配置
    configs = {
        'full_46_equal': f'predictions_h{horizon}_full_18m.parquet',
        'full_46_riskparity': f'predictions_h{horizon}_full_18m.parquet',  # 需要重新生成
        'pruned_36_equal': f'predictions_h{horizon}_pruned_18m.parquet',
        'pruned_36_riskparity': f'predictions_h{horizon}_pruned_18m.parquet',  # 需要重新生成
        'original_30_equal': f'predictions_h{horizon}_stacking_6m.parquet',
        'original_30_riskparity': f'predictions_h{horizon}_stacking_6m.parquet',  # 需要重新生成
    }
    
    for name, filename in configs.items():
        path = PRED_DIR / filename
        if path.exists():
            try:
                df = pd.read_parquet(path)
                df['date'] = pd.to_datetime(df['date'])
                predictions[name] = df
                print(f"  Loaded {name}: {len(df)} rows")
            except Exception as e:
                print(f"  Failed to load {name}: {e}")
    
    return predictions


def detect_regime(df):
    """检测市场 regime（使用原始收益率）"""
    if 'date' not in df.columns:
        return df
    
    # 计算市场指标（用原始收益率）
    market_stats = df.groupby('date').agg({
        'excess_return': ['mean', 'std', lambda x: (x > 0).mean()],
    })
    market_stats.columns = ['avg_excess', 'vol_excess', 'win_rate']
    
    # 计算滚动统计
    market_stats['avg_ma'] = market_stats['avg_excess'].rolling(20, min_periods=1).mean()
    market_stats['vol_ma'] = market_stats['vol_excess'].rolling(20, min_periods=1).mean()
    market_stats['avg_median'] = market_stats['avg_excess'].rolling(60, min_periods=1).median()
    market_stats['vol_median'] = market_stats['vol_excess'].rolling(60, min_periods=1).median()
    
    # 识别 regime
    market_stats['regime'] = 'sideways'
    market_stats.loc[(market_stats['avg_excess'] > market_stats['avg_median'] * 1.5) & 
                     (market_stats['vol_excess'] < market_stats['vol_median'] * 0.8), 'regime'] = 'bull'
    market_stats.loc[(market_stats['avg_excess'] < market_stats['avg_median'] * 0.5) & 
                     (market_stats['vol_excess'] > market_stats['vol_median'] * 1.2), 'regime'] = 'bear'
    
    return market_stats


def select_best_model(regime, regime_performance):
    """根据 regime 选择最佳模型"""
    if regime in regime_performance:
        return regime_performance[regime]
    return 'full_46_equal'  # 默认


def precompute_regime_performance(predictions, market_stats):
    """预计算每个 regime 的最佳模型（用 IC 作为指标）"""
    regime_performance = {}
    
    for regime in ['bull', 'bear', 'sideways']:
        best_model = None
        best_ic = -999
        
        for model_name, pred_df in predictions.items():
            pred_df = pred_df.copy()
            pred_df['date'] = pd.to_datetime(pred_df['date'])
            pred_df['year_month'] = pred_df['date'].dt.to_period('M')
            
            # 获取该模型在该 regime 下的 IC
            model_regime_ics = []
            for month in pred_df['year_month'].unique():
                month_data = pred_df[pred_df['year_month'] == month]
                month_date = month_data['date'].iloc[0]
                
                if month_date in market_stats.index:
                    month_regime = market_stats.loc[month_date, 'regime']
                    if month_regime == regime and len(month_data) > 10:
                        # 计算 rank IC
                        ic = month_data['predicted'].corr(month_data['excess_return'], method='spearman')
                        if not np.isnan(ic):
                            model_regime_ics.append(ic)
            
            if model_regime_ics:
                avg_ic = np.mean(model_regime_ics)
                if avg_ic > best_ic:
                    best_ic = avg_ic
                    best_model = model_name
        
        if best_model:
            regime_performance[regime] = best_model
            print(f"  {regime}: best model = {best_model} (IC = {best_ic:.4f})")
    
    return regime_performance


def walk_forward_regime_adaptive_v2(predictions, feature_df, horizon=10):
    """Regime 自适应 walk-forward V2"""
    # 获取市场 regime
    market_stats = detect_regime(feature_df)
    
    # 合并所有预测
    all_models = list(predictions.keys())
    print(f"\n模型池: {all_models}")
    
    # 预计算每个 regime 的最佳模型
    print("\n预计算 regime 性能...")
    regime_performance = precompute_regime_performance(predictions, market_stats)
    
    # 构建统一的预测 DataFrame
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
    
    # 逐月做 regime 自适应选择
    all_preds = []
    
    for i, test_month in enumerate(months):
        test_data = unified[unified['year_month'] == test_month]
        if test_data.empty:
            continue
        
        # 获取当前 regime
        current_date = test_data['date'].iloc[0]
        if current_date in market_stats.index:
            current_regime = market_stats.loc[current_date, 'regime']
        else:
            current_regime = 'sideways'
        
        # 选择最佳模型
        best_model = select_best_model(current_regime, regime_performance)
        
        # 获取该模型的预测
        model_pred = test_data[test_data['model_name'] == best_model]
        if model_pred.empty:
            # 如果该模型没有预测，用第一个可用的
            model_pred = test_data.groupby('model_name').first().reset_index()
            if not model_pred.empty:
                best_model = model_pred['model_name'].iloc[0]
                model_pred = test_data[test_data['model_name'] == best_model]
        
        if not model_pred.empty:
            # 添加到结果
            pred = model_pred[['date', 'ts_code', 'stock_return', 'benchmark_return', 'excess_return', 'industry', 'predicted', 'model_name']].copy()
            pred['regime'] = current_regime
            all_preds.append(pred)
            print(f"  {test_month}: regime={current_regime}, model={best_model}")
    
    if not all_preds:
        return pd.DataFrame()
    
    return pd.concat(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    
    print("[regime_adaptive_v2] Loading predictions...")
    predictions = load_all_predictions(args.horizon)
    
    if not predictions:
        print("[regime_adaptive_v2] No predictions loaded")
        return
    
    # 加载原始特征数据用于 regime 检测（非 rank 标准化）
    print("\n[regime_adaptive_v2] Loading original feature data for regime detection...")
    feature_df = load_dataset(args.horizon, "memory/dataset/features_h10_composite.parquet")
    
    print("\n[regime_adaptive_v2] Running regime-adaptive walk-forward...")
    result = walk_forward_regime_adaptive_v2(predictions, feature_df, args.horizon)
    
    if result.empty:
        print("[regime_adaptive_v2] No predictions generated")
        return
    
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"predictions_h{args.horizon}_regime_adaptive_v2.parquet"
    output_path = PRED_DIR / output_name
    result.to_parquet(output_path, index=False)
    print(f"\n[regime_adaptive_v2] Saved {len(result)} predictions to {output_path}")
    
    # 输出 regime 统计
    print("\n=== Regime Statistics ===")
    print(result['regime'].value_counts())
    print("\n=== Model Usage ===")
    print(result['model_name'].value_counts())


if __name__ == "__main__":
    main()
