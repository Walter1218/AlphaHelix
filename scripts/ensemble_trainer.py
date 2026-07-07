"""
AlphaHelix 多模型集成训练器

集成 LightGBM + XGBoost + CatBoost + Ridge，用简单平均或 stacking 组合预测。

用法：
    python scripts/ensemble_trainer.py --dataset memory/dataset/features_h10_composite.parquet --horizon 10
"""
import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import tushare_call
from model_trainer import load_dataset, get_feature_cols, _make_rank_label, _compute_group_counts

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

DATASET_DIR = Path("memory/dataset")
MODEL_DIR = Path("memory/models")
PRED_DIR = Path("memory/predictions")


def train_lightgbm(X_tr, y_tr, X_val, y_val, feature_cols, **kwargs):
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)
    params = {
        "objective": "regression", "metric": "mse", "verbosity": -1,
        "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
        "colsample_bytree": 0.8, "seed": 42, "num_leaves": 31,
    }
    model = lgb.train(params, dtrain, num_boost_round=300,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(30, verbose=False)])
    return {"model": model, "feature_cols": feature_cols, "best_iteration": model.best_iteration}


def train_xgboost(X_tr, y_tr, X_val, y_val, feature_cols, **kwargs):
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
        "colsample_bytree": 0.8, "seed": 42,
    }
    model = xgb.train(params, dtrain, num_boost_round=300,
                      evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False)
    return {"model": model, "feature_cols": feature_cols}


def train_catboost(X_tr, y_tr, X_val, y_val, feature_cols, **kwargs):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(
        iterations=300, learning_rate=0.05, depth=6,
        subsample=0.8, random_seed=42, verbose=0,
        early_stopping_rounds=30,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    return {"model": model, "feature_cols": feature_cols}


def train_ridge(X_tr, y_tr, X_val, y_val, feature_cols, **kwargs):
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    return {"model": model, "feature_cols": feature_cols}


def train_mlp(X_tr, y_tr, X_val, y_val, feature_cols, **kwargs):
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                         random_state=42, early_stopping=True)
    model.fit(X_tr, y_tr)
    return {"model": model, "feature_cols": feature_cols}


def predict_model(model_dict, X, model_type):
    model = model_dict["model"]
    if model_type == "lightgbm":
        return model.predict(X, num_iteration=model_dict.get("best_iteration"))
    elif model_type == "xgboost":
        return model.predict(xgb.DMatrix(X, feature_names=model_dict["feature_cols"]))
    elif model_type == "catboost":
        return model.predict(X)
    else:
        return model.predict(X)


def walk_forward_ensemble(df, feature_cols, train_window_months=6,
                          models=None, method="average"):
    """
    walk-forward 集成预测。
    
    method:
        - "average": 简单平均所有模型预测
        - "stacking": 用 Ridge 组合模型预测
    """
    df = df.sort_values("date").copy()
    df["year_month"] = df["date"].dt.to_period("M")
    months = sorted(df["year_month"].unique())
    
    if models is None:
        models = {
            "lightgbm": train_lightgbm,
            "xgboost": train_xgboost,
            "catboost": train_catboost,
            "ridge": train_ridge,
        }
    
    all_preds = []
    
    for i, test_month in enumerate(months):
        train_months = months[max(0, i - train_window_months):i]
        if len(train_months) < 3:
            continue
        
        train_df = df[df["year_month"].isin(train_months)]
        test_df = df[df["year_month"] == test_month]
        if train_df.empty or test_df.empty:
            continue
        
        # 训练/验证切分
        val_month = train_months[-1]
        tr_df = train_df[train_df["year_month"] != val_month].sort_values("date")
        val_df = train_df[train_df["year_month"] == val_month].sort_values("date")
        
        X_tr = tr_df[feature_cols].values
        y_tr = tr_df["excess_return"].values
        X_val = val_df[feature_cols].values
        y_val = val_df["excess_return"].values
        X_test = test_df[feature_cols].values
        
        # 训练所有模型
        trained_models = {}
        model_preds = {}
        
        for name, train_fn in models.items():
            try:
                model = train_fn(X_tr, y_tr, X_val, y_val, feature_cols)
                trained_models[name] = model
                pred = predict_model(model, X_test, name)
                model_preds[name] = pred
            except Exception as e:
                print(f"  {test_month} {name} failed: {e}")
                continue
        
        if not model_preds:
            continue
        
        # 组合预测
        if method == "average":
            preds = np.mean(list(model_preds.values()), axis=0)
        elif method == "stacking":
            # 用验证集预测训练 stacking meta-learner
            val_preds = {}
            for name, model in trained_models.items():
                val_preds[name] = predict_model(model, X_val, name)
            val_X = np.column_stack(list(val_preds.values()))
            
            meta = Ridge(alpha=1.0)
            meta.fit(val_X, y_val)
            
            test_X = np.column_stack(list(model_preds.values()))
            preds = meta.predict(test_X)
        else:
            preds = np.mean(list(model_preds.values()), axis=0)
        
        pred = test_df[["date", "ts_code", "stock_return", "benchmark_return", "excess_return", "industry"]].copy()
        pred["predicted"] = preds
        
        # 记录各模型单独预测（诊断用）
        for name, p in model_preds.items():
            pred[f"pred_{name}"] = p
        
        all_preds.append(pred)
        print(f"  {test_month}: {len(test_df)} stocks, {len(model_preds)} models")
    
    if not all_preds:
        return pd.DataFrame()
    
    return pd.concat(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--train-window-months", type=int, default=6)
    parser.add_argument("--method", choices=["average", "stacking"], default="average")
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    
    df = load_dataset(args.horizon, args.dataset)
    feature_cols = get_feature_cols(df)
    print(f"[ensemble] Dataset: {len(df)} rows, features: {len(feature_cols)}")
    
    result = walk_forward_ensemble(
        df, feature_cols,
        train_window_months=args.train_window_months,
        method=args.method,
    )
    
    if result.empty:
        print("[ensemble] No predictions generated")
        return
    
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"predictions_h{args.horizon}_ensemble_{args.method}.parquet"
    output_path = PRED_DIR / output_name
    result.to_parquet(output_path, index=False)
    print(f"[ensemble] Saved {len(result)} predictions to {output_path}")
    
    # 输出各模型单独表现（诊断）
    print("\n=== Individual Model Performance ===")
    pred_cols = [c for c in result.columns if c.startswith("pred_")]
    for col in pred_cols:
        name = col.replace("pred_", "")
        ic = result.groupby("date").apply(
            lambda g: g[col].corr(g["excess_return"], method="spearman")
        ).mean()
        print(f"  {name}: Mean IC = {ic:.4f}")


if __name__ == "__main__":
    main()
