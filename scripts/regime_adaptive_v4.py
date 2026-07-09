"""
AlphaRegime 自适应模型 V4

Regime-specific ensemble weights：
- Bull: 偏重树模型（捕捉趋势）
- Bear: 偏重线性模型（更稳健）
- Sideways: 均衡权重

用法：
    python scripts/regime_adaptive_v4.py --horizon 10
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


def get_regime_weights(regime, model_names):
    """根据 regime 返回模型权重"""
    # 默认权重
    weights = {name: 1.0 / len(model_names) for name in model_names}
    
    if regime == 'bull':
        # Bull: 偏重树模型（捕捉趋势）
        for name in model_names:
            if 'lightgbm' in name or 'xgboost' in name:
                weights[name] = 0.3
            elif 'catboost' in name:
                weights[name] = 0.2
            elif 'ridge' in name:
                weights[name] = 0.1
            elif 'mlp' in name:
                weights[name] = 0.1
            else:
                weights[name] = 0.1
    elif regime == 'bear':
        # Bear: 偏重线性模型（更稳健）
        for name in model_names:
            if 'ridge' in name:
                weights[name] = 0.3
            elif 'catboost' in name:
                weights[name] = 0.2
            elif 'lightgbm' in name or 'xgboost' in name:
                weights[name] = 0.15
            elif 'mlp' in name:
                weights[name] = 0.1
            else:
                weights[name] = 0.1
    else:
        # Sideways: 均衡权重
        pass
    
    # 归一化
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def walk_forward_regime_v4(predictions, feature_df, horizon=10):
    """Walk-forward Regime V4"""
    market_stats = detect_regime(feature_df)
    
    unified = None
    for model_name, pred_df in predictions.items():
        pred_df = pred_df.copy()
        pred_df['model_name'] = model_name
        if unified is None:
            unified = pred_df
        else:
            unified = pd.concat([unified, pred_df], ignore_index=True)
    
    unified['date'] = pd.to_datetime(unified['date'])
    unified['year_month'] = unified['date'].dt.to_period('M')
    months = sorted(unified['year_month'].unique())
    
    model_names = list(predictions.keys())
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
        
        # 获取 regime-specific 权重
        weights = get_regime_weights(current_regime, model_names)
        
        # 加权平均
        model_preds = {}
        for model_name in model_names:
            model_data = test_data[test_data['model_name'] == model_name]
            if not model_data.empty:
                model_preds[model_name] = model_data['predicted'].values
        
        if model_preds:
            # 用 numpy 计算加权平均
            weight_arr = np.array([weights.get(name, 0) for name in model_preds.keys()])
            pred_arr = np.column_stack(list(model_preds.values()))
            weighted_pred = pred_arr @ weight_arr
            
            # 使用第一个模型的框架
            first_model = list(model_preds.keys())[0]
            first_data = test_data[test_data['model_name'] == first_model]
            pred = first_data[['date', 'ts_code', 'stock_return', 'benchmark_return', 
                                'excess_return', 'industry']].copy()
            pred['predicted'] = weighted_pred
            pred['regime'] = current_regime
            all_preds.append(pred)
    
    if not all_preds:
        return pd.DataFrame()
    
    return pd.concat(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    
    print("[regime_v4] Loading predictions...")
    predictions = {}
    configs = {
        'full_46_equal': 'predictions_h10_stacking_5model.parquet',
        'pruned_36_equal': 'predictions_h10_pruned_18m.parquet',
    }
    
    for name, filename in configs.items():
        path = PRED_DIR / filename
        if path.exists():
            df = pd.read_parquet(path)
            df['date'] = pd.to_datetime(df['date'])
            predictions[name] = df
            print(f"  Loaded {name}: {len(df)} rows")
    
    if not predictions:
        print("[regime_v4] No predictions loaded")
        return
    
    print("\n[regime_v4] Loading feature data...")
    feature_df = load_dataset(args.horizon, "memory/dataset/features_h10_composite_fixed.parquet")
    
    print("\n[regime_v4] Running regime-adaptive walk-forward...")
    result = walk_forward_regime_v4(predictions, feature_df, args.horizon)
    
    if result.empty:
        print("[regime_v4] No predictions generated")
        return
    
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"predictions_h{args.horizon}_regime_v4.parquet"
    output_path = PRED_DIR / output_name
    result.to_parquet(output_path, index=False)
    print(f"\n[regime_v4] Saved {len(result)} predictions to {output_path}")
    
    print("\n=== Regime Statistics ===")
    print(result['regime'].value_counts())


if __name__ == "__main__":
    main()
