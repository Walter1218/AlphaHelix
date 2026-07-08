"""
AlphaRegime 自适应模型 V3

改进点：
1. 更好的 regime 检测（多指标）
2. 用累计超额代替 IC 选择模型
3. 动态加权代替硬选择

用法：
    python scripts/regime_adaptive_v3.py --horizon 10
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


def load_all_predictions(horizon=10):
    """加载所有模型的预测结果"""
    predictions = {}
    
    configs = {
        'full_46_equal': f'predictions_h{horizon}_full_18m.parquet',
        'pruned_36_equal': f'predictions_h{horizon}_pruned_18m.parquet',
        'original_30_equal': f'predictions_h{horizon}_original_30_fixed.parquet',
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


def detect_regime_v3(df):
    """改进的 regime 检测（多指标）"""
    if 'date' not in df.columns:
        return df
    
    # 计算多个市场指标
    market_stats = df.groupby('date').agg({
        'excess_return': ['mean', 'std', lambda x: (x > 0).mean()],
        'volatility_20': 'mean',
    })
    market_stats.columns = ['avg_excess', 'vol_excess', 'win_rate', 'market_vol']
    
    # 计算滚动统计
    market_stats['avg_ma'] = market_stats['avg_excess'].rolling(20, min_periods=1).mean()
    market_stats['vol_ma'] = market_stats['vol_excess'].rolling(20, min_periods=1).mean()
    market_stats['win_ma'] = market_stats['win_rate'].rolling(20, min_periods=1).mean()
    market_stats['avg_median'] = market_stats['avg_excess'].rolling(60, min_periods=1).median()
    market_stats['vol_median'] = market_stats['vol_excess'].rolling(60, min_periods=1).median()
    market_stats['win_median'] = market_stats['win_rate'].rolling(60, min_periods=1).median()
    
    # 多指标综合 regime 判断
    market_stats['regime_score'] = 0.0
    
    # 指标1: 平均超额收益
    market_stats.loc[market_stats['avg_excess'] > market_stats['avg_median'] * 1.2, 'regime_score'] += 1
    market_stats.loc[market_stats['avg_excess'] < market_stats['avg_median'] * 0.8, 'regime_score'] -= 1
    
    # 指标2: 胜率
    market_stats.loc[market_stats['win_rate'] > market_stats['win_median'] * 1.05, 'regime_score'] += 0.5
    market_stats.loc[market_stats['win_rate'] < market_stats['win_median'] * 0.95, 'regime_score'] -= 0.5
    
    # 指标3: 波动率（反向）
    market_stats.loc[market_stats['vol_excess'] < market_stats['vol_median'] * 0.9, 'regime_score'] += 0.5
    market_stats.loc[market_stats['vol_excess'] > market_stats['vol_median'] * 1.1, 'regime_score'] -= 0.5
    
    # 综合 regime
    market_stats['regime'] = 'sideways'
    market_stats.loc[market_stats['regime_score'] > 0.5, 'regime'] = 'bull'
    market_stats.loc[market_stats['regime_score'] < -0.5, 'regime'] = 'bear'
    
    return market_stats


def select_best_model_walkforward_v3(regime, test_month, predictions, market_stats, 
                                      lookback_months=12, metric='cumulative_excess'):
    """Walk-forward 选择最佳模型（用累计超额或 IC）"""
    best_model = None
    best_score = -999
    
    for model_name, pred_df in predictions.items():
        pred_df = pred_df.copy()
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        pred_df['year_month'] = pred_df['date'].dt.to_period('M')
        
        # 只用 test_month 之前的数据
        past_data = pred_df[pred_df['year_month'] < test_month].tail(lookback_months * 30)
        
        # 计算该模型在该 regime 下的历史表现
        model_regime_scores = []
        for month in past_data['year_month'].unique():
            month_data = past_data[past_data['year_month'] == month]
            month_date = month_data['date'].iloc[0]
            
            if month_date in market_stats.index:
                month_regime = market_stats.loc[month_date, 'regime']
                if month_regime == regime and len(month_data) > 10:
                    if metric == 'cumulative_excess':
                        score = month_data['excess_return'].mean()
                    else:  # IC
                        score = month_data['predicted'].corr(month_data['excess_return'], method='spearman')
                        if np.isnan(score):
                            continue
                    model_regime_scores.append(score)
        
        if model_regime_scores:
            avg_score = np.mean(model_regime_scores)
            if avg_score > best_score:
                best_score = avg_score
                best_model = model_name
    
    if best_model:
        return best_model, best_score
    return 'full_46_equal', 0.0


def dynamic_weight_models(predictions, test_month, market_stats, lookback_months=12):
    """动态加权：根据历史表现加权所有模型"""
    model_weights = {}
    model_preds = {}
    
    for model_name, pred_df in predictions.items():
        pred_df = pred_df.copy()
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        pred_df['year_month'] = pred_df['date'].dt.to_period('M')
        
        # 只用 test_month 之前的数据
        past_data = pred_df[pred_df['year_month'] < test_month].tail(lookback_months * 30)
        
        # 计算历史表现
        scores = []
        for month in past_data['year_month'].unique():
            month_data = past_data[past_data['year_month'] == month]
            if len(month_data) > 10:
                score = month_data['excess_return'].mean()
                scores.append(score)
        
        if scores:
            # 用 softmax 将分数转为权重
            model_weights[model_name] = np.exp(np.mean(scores) * 100)  # 温度参数 100
        else:
            model_weights[model_name] = 1.0
    
    # 归一化权重
    total = sum(model_weights.values())
    for name in model_weights:
        model_weights[name] /= total
    
    return model_weights


def walk_forward_regime_adaptive_v3(predictions, feature_df, horizon=10, 
                                    metric='cumulative_excess', use_dynamic_weight=False):
    """Regime 自适应 walk-forward V3"""
    market_stats = detect_regime_v3(feature_df)
    
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
        
        if use_dynamic_weight:
            # 动态加权
            model_weights = dynamic_weight_models(predictions, test_month, market_stats)
            weighted_preds = None
            for model_name, weight in model_weights.items():
                model_pred = test_data[test_data['model_name'] == model_name]
                if not model_pred.empty:
                    if weighted_preds is None:
                        weighted_preds = model_pred.copy()
                        weighted_preds['predicted'] = model_pred['predicted'] * weight
                    else:
                        weighted_preds['predicted'] += model_pred['predicted'] * weight
            
            if weighted_preds is not None and not weighted_preds.empty:
                pred = weighted_preds[['date', 'ts_code', 'stock_return', 'benchmark_return', 
                                        'excess_return', 'industry', 'predicted']].copy()
                pred['regime'] = current_regime
                pred['model_name'] = 'dynamic_weighted'
                all_preds.append(pred)
        else:
            # 选择最佳模型
            best_model, score = select_best_model_walkforward_v3(
                current_regime, test_month, predictions, market_stats, metric=metric
            )
            
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
    parser.add_argument("--metric", choices=["cumulative_excess", "ic"], default="cumulative_excess")
    parser.add_argument("--use-dynamic-weight", action="store_true")
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    
    print("[regime_adaptive_v3] Loading predictions...")
    predictions = load_all_predictions(args.horizon)
    
    if not predictions:
        print("[regime_adaptive_v3] No predictions loaded")
        return
    
    print("\n[regime_adaptive_v3] Loading feature data for regime detection...")
    feature_df = load_dataset(args.horizon, "memory/dataset/features_h10_composite_fixed.parquet")
    
    print("\n[regime_adaptive_v3] Running regime-adaptive walk-forward...")
    result = walk_forward_regime_adaptive_v3(
        predictions, feature_df, args.horizon,
        metric=args.metric,
        use_dynamic_weight=args.use_dynamic_weight,
    )
    
    if result.empty:
        print("[regime_adaptive_v3] No predictions generated")
        return
    
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"predictions_h{args.horizon}_regime_adaptive_v3.parquet"
    output_path = PRED_DIR / output_name
    result.to_parquet(output_path, index=False)
    print(f"\n[regime_adaptive_v3] Saved {len(result)} predictions to {output_path}")
    
    print("\n=== Regime Statistics ===")
    print(result['regime'].value_counts())
    print("\n=== Model Usage ===")
    print(result['model_name'].value_counts())


if __name__ == "__main__":
    main()
