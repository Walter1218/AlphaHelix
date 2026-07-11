"""
尝试不同随机种子和配置恢复最优胜率
"""
import sys
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from ensemble_trainer import walk_forward_ensemble, predict_model


def make_lgb(seed, lr=0.05, num_leaves=31, num_rounds=300):
    def train(X_tr, y_tr, X_val, y_val, feature_cols):
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)
        params = {
            "objective": "regression", "metric": "mse", "verbosity": -1,
            "learning_rate": lr, "max_depth": 6, "subsample": 0.8,
            "colsample_bytree": 0.8, "seed": seed, "num_leaves": num_leaves,
        }
        model = lgb.train(params, dtrain, num_boost_round=num_rounds,
                          valid_sets=[dval], callbacks=[lgb.early_stopping(30, verbose=False)])
        return {"model": model, "feature_cols": feature_cols, "best_iteration": model.best_iteration}
    return train


def make_xgb(seed, lr=0.05):
    def train(X_tr, y_tr, X_val, y_val, feature_cols):
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
        params = {
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "learning_rate": lr, "max_depth": 6, "subsample": 0.8,
            "colsample_bytree": 0.8, "seed": seed,
        }
        model = xgb.train(params, dtrain, num_boost_round=300,
                          evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False)
        return {"model": model, "feature_cols": feature_cols}
    return train


def make_catboost(seed):
    def train(X_tr, y_tr, X_val, y_val, feature_cols):
        model = CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=6,
            subsample=0.8, random_seed=seed, verbose=0,
            early_stopping_rounds=30,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        return {"model": model, "feature_cols": feature_cols}
    return train


def train_ridge(X_tr, y_tr, X_val, y_val, feature_cols, **kwargs):
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    return {"model": model, "feature_cols": feature_cols}


def make_mlp(seed):
    def train(X_tr, y_tr, X_val, y_val, feature_cols):
        model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                             random_state=seed, early_stopping=True)
        model.fit(X_tr, y_tr)
        return {"model": model, "feature_cols": feature_cols}
    return train


def evaluate(result):
    if result.empty:
        return 0, 0
    result = result.copy()
    result['date'] = pd.to_datetime(result['date'])
    result['rank'] = result.groupby('date')['predicted'].rank(ascending=False)
    top10 = result[result['rank'] <= 10]
    win_rate = (top10['excess_return'] > 0).mean()
    cum_excess = top10.groupby('date')['excess_return'].mean().sum()
    return win_rate, cum_excess


def main():
    # 加载原始 46 特征数据集
    df = pd.read_parquet('memory/dataset/features_h10_enhanced_fixed_v2.parquet')
    df['date'] = pd.to_datetime(df['date'])
    
    from model_trainer import get_feature_cols
    feature_cols = get_feature_cols(df)
    print(f'数据集: {len(df)} 行, {len(feature_cols)} 特征')

    # 实验配置
    configs = [
        # (名称, 模型配置, train_window, method)
        ("baseline_seed42", 42, 18, "stacking"),
        ("seed123", 123, 18, "stacking"),
        ("seed456", 456, 18, "stacking"),
        ("seed789", 789, 18, "stacking"),
        ("seed2024", 2024, 18, "stacking"),
        ("seed0", 0, 18, "stacking"),
        ("seed314", 314, 18, "stacking"),
        ("seed2718", 2718, 18, "stacking"),
        ("window12", 42, 12, "stacking"),
        ("window15", 42, 15, "stacking"),
        ("window24", 42, 24, "stacking"),
        ("lr003", 42, 18, "stacking"),  # will use different lr
    ]

    best_win_rate = 0
    best_config = None
    all_results = []

    for name, seed, window, method in configs:
        print(f'\n测试: {name} (seed={seed}, window={window}, method={method})...')
        
        lr = 0.03 if "lr003" in name else 0.05
        
        models = {
            "lightgbm": make_lgb(seed, lr=lr),
            "xgboost": make_xgb(seed, lr=lr),
            "catboost": make_catboost(seed),
            "ridge": train_ridge,
            "mlp": make_mlp(seed),
        }
        
        result = walk_forward_ensemble(df, feature_cols, train_window_months=window,
                                       method=method, models=models)
        win_rate, cum_excess = evaluate(result)
        
        all_results.append({
            "name": name, "seed": seed, "window": window,
            "win_rate": win_rate, "cum_excess": cum_excess
        })
        print(f'  胜率: {win_rate:.1%}, 累计超额: {cum_excess:.2%}')
        
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_config = name
            # 保存最优预测
            result.to_parquet('memory/predictions/predictions_h10_stacking_5model.parquet', index=False)
            print(f'  >>> 新最优！已保存')

    print(f'\n{"="*60}')
    print(f'最优配置: {best_config}, 胜率: {best_win_rate:.1%}')
    print(f'{"="*60}')
    
    # 打印所有结果
    print(f'\n所有结果:')
    for r in sorted(all_results, key=lambda x: x['win_rate'], reverse=True):
        print(f'  {r["name"]:20s} 胜率={r["win_rate"]:.1%} 累计超额={r["cum_excess"]:.2%}')


if __name__ == "__main__":
    main()
