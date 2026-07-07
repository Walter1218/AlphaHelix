"""
AlphaRegime 自适应模型

根据市场状态动态选择最佳模型配置：
1. Regime 检测：用市场指标识别当前状态
2. Regime 特定模型：为每个状态训练专属模型
3. 动态切换：根据当前状态选择最佳模型

用法：
    python scripts/regime_adaptive.py --dataset memory/dataset/features_h10_enhanced_fixed_v2.parquet --horizon 10
"""
import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_trainer import load_dataset, get_feature_cols, train_gbdt

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def detect_regime(df):
    """
    检测市场状态：
    - bull: 市场宽度 > 0.5 且波动率 < 中位数
    - bear: 市场宽度 < 0.5 且波动率 > 中位数
    - sideways: 其他情况
    """
    if "date" not in df.columns:
        return df
    
    # 计算市场指标
    market_stats = df.groupby("date").agg({
        "mom_20": lambda x: (x > 0).mean(),  # 市场宽度
        "volatility_20": "mean",  # 市场波动率
    }).rename(columns={"mom_20": "market_breadth", "volatility_20": "market_vol"})
    
    # 计算滚动中位数
    market_stats["breadth_median"] = market_stats["market_breadth"].rolling(60, min_periods=1).median()
    market_stats["vol_median"] = market_stats["market_vol"].rolling(60, min_periods=1).median()
    
    # 识别 regime
    market_stats["regime"] = "sideways"
    market_stats.loc[(market_stats["market_breadth"] > market_stats["breadth_median"]) & 
                     (market_stats["market_vol"] < market_stats["vol_median"]), "regime"] = "bull"
    market_stats.loc[(market_stats["market_breadth"] < market_stats["breadth_median"]) & 
                     (market_stats["market_vol"] > market_stats["vol_median"]), "regime"] = "bear"
    
    # 合并回原数据
    df = df.copy()
    df["regime"] = df["date"].map(market_stats["regime"])
    df["regime"] = df["regime"].fillna("sideways")
    
    return df, market_stats


def train_regime_models(df, feature_cols, train_window_months=18):
    """为每个 regime 训练专属模型"""
    df = df.sort_values("date").copy()
    df["year_month"] = df["date"].dt.to_period("M")
    months = sorted(df["year_month"].unique())
    
    # 按 regime 分组训练
    regime_models = {}
    
    for regime in ["bull", "bear", "sideways"]:
        regime_df = df[df["regime"] == regime]
        if len(regime_df) < 100:  # 样本太少不训练
            continue
        
        # 用最近的数据训练
        recent_months = sorted(regime_df["year_month"].unique())[-train_window_months:]
        train_data = regime_df[regime_df["year_month"].isin(recent_months)]
        
        if len(train_data) < 50:
            continue
        
        X = train_data[feature_cols].fillna(0).values
        y = train_data["excess_return"].values
        
        # 训练模型
        try:
            model = train_gbdt(X, y, X[:100], y[:100], feature_cols, 
                             model_type="lightgbm", target="excess_return")
            regime_models[regime] = model
            print(f"  {regime}: trained on {len(train_data)} samples")
        except Exception as e:
            print(f"  {regime}: failed - {e}")
    
    return regime_models


def walk_forward_regime_adaptive(df, feature_cols, train_window_months=18):
    """Regime 自适应 walk-forward"""
    df, market_stats = detect_regime(df)
    df = df.sort_values("date").copy()
    df["year_month"] = df["date"].dt.to_period("M")
    months = sorted(df["year_month"].unique())
    
    all_preds = []
    
    for i, test_month in enumerate(months):
        train_months = months[max(0, i - train_window_months):i]
        if len(train_months) < 3:
            continue
        
        test_df = df[df["year_month"] == test_month]
        if test_df.empty:
            continue
        
        # 获取当前 regime
        current_regime = test_df["regime"].mode()[0] if not test_df["regime"].mode().empty else "sideways"
        
        # 用当前 regime 的数据训练
        train_data = df[df["year_month"].isin(train_months) & (df["regime"] == current_regime)]
        if len(train_data) < 50:
            # 如果当前 regime 数据太少，用全部数据
            train_data = df[df["year_month"].isin(train_months)]
        
        if train_data.empty:
            continue
        
        X_tr = train_data[feature_cols].fillna(0).values
        y_tr = train_data["excess_return"].values
        X_test = test_df[feature_cols].fillna(0).values
        
        try:
            model = train_gbdt(X_tr, y_tr, X_tr[:100], y_tr[:100], feature_cols,
                             model_type="lightgbm", target="excess_return")
            preds = model.predict(X_test)
            
            pred = test_df[["date", "ts_code", "stock_return", "benchmark_return", "excess_return", "industry"]].copy()
            pred["predicted"] = preds
            pred["regime"] = current_regime
            all_preds.append(pred)
            print(f"  {test_month}: regime={current_regime}, {len(test_df)} stocks")
        except Exception as e:
            print(f"  {test_month}: failed - {e}")
    
    if not all_preds:
        return pd.DataFrame()
    
    return pd.concat(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--train-window-months", type=int, default=18)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    
    df = load_dataset(args.horizon, args.dataset)
    feature_cols = get_feature_cols(df)
    print(f"[regime_adaptive] Dataset: {len(df)} rows, features: {len(feature_cols)}")
    
    result = walk_forward_regime_adaptive(
        df, feature_cols,
        train_window_months=args.train_window_months,
    )
    
    if result.empty:
        print("[regime_adaptive] No predictions generated")
        return
    
    PRED_DIR = Path("memory/predictions")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"predictions_h{args.horizon}_regime_adaptive.parquet"
    output_path = PRED_DIR / output_name
    result.to_parquet(output_path, index=False)
    print(f"[regime_adaptive] Saved {len(result)} predictions to {output_path}")
    
    # 输出 regime 统计
    print("\n=== Regime Statistics ===")
    print(result["regime"].value_counts())


if __name__ == "__main__":
    main()
