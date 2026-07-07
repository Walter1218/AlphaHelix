"""
AlphaHelix 两阶段模型

Stage 1: 二分类（涨/跌）→ 召回
Stage 2: 回归（预测收益大小）→ 排序

用法：
    python scripts/two_stage_ensemble.py --dataset memory/dataset/features_h10_composite.parquet --horizon 10
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
from model_trainer import load_dataset, get_feature_cols

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = None
    CatBoostRegressor = None

DATASET_DIR = Path("memory/dataset")
MODEL_DIR = Path("memory/models")
PRED_DIR = Path("memory/predictions")


def train_classifier(X_tr, y_tr, X_val, y_val, feature_cols, model_type="lightgbm"):
    """Stage 1: 训练分类器"""
    if model_type == "lightgbm":
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)
        params = {
            "objective": "binary", "metric": "auc", "verbosity": -1,
            "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
            "colsample_bytree": 0.8, "seed": 42, "num_leaves": 31,
        }
        model = lgb.train(params, dtrain, num_boost_round=300,
                          valid_sets=[dval], callbacks=[lgb.early_stopping(30, verbose=False)])
        return {"model": model, "type": "lightgbm", "best_iteration": model.best_iteration}
    elif model_type == "xgboost":
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
        params = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
            "colsample_bytree": 0.8, "seed": 42,
        }
        model = xgb.train(params, dtrain, num_boost_round=300,
                          evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False)
        return {"model": model, "type": "xgboost", "feature_cols": feature_cols}
    elif model_type == "catboost":
        model = CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6,
            subsample=0.8, random_seed=42, verbose=0,
            early_stopping_rounds=30,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        return {"model": model, "type": "catboost"}
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_regressor(X_tr, y_tr, X_val, y_val, feature_cols, model_type="lightgbm"):
    """Stage 2: 训练回归器"""
    if model_type == "lightgbm":
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)
        params = {
            "objective": "regression", "metric": "mse", "verbosity": -1,
            "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
            "colsample_bytree": 0.8, "seed": 42, "num_leaves": 31,
        }
        model = lgb.train(params, dtrain, num_boost_round=300,
                          valid_sets=[dval], callbacks=[lgb.early_stopping(30, verbose=False)])
        return {"model": model, "type": "lightgbm", "best_iteration": model.best_iteration}
    elif model_type == "xgboost":
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
        params = {
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
            "colsample_bytree": 0.8, "seed": 42,
        }
        model = xgb.train(params, dtrain, num_boost_round=300,
                          evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False)
        return {"model": model, "type": "xgboost", "feature_cols": feature_cols}
    elif model_type == "catboost":
        model = CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=6,
            subsample=0.8, random_seed=42, verbose=0,
            early_stopping_rounds=30,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        return {"model": model, "type": "catboost"}
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_model(model_dict, X):
    """预测"""
    model = model_dict["model"]
    mtype = model_dict["type"]
    if mtype == "lightgbm":
        return model.predict(X, num_iteration=model_dict.get("best_iteration"))
    elif mtype == "xgboost":
        return model.predict(xgb.DMatrix(X, feature_names=model_dict.get("feature_cols", [])))
    elif mtype == "catboost":
        return model.predict(X)
    else:
        return model.predict(X)


def walk_forward_two_stage(df, feature_cols, train_window_months=6,
                           recall_ratio=0.5):
    """
    walk-forward 两阶段模型。
    
    Stage 1: 二分类（涨/跌）→ 召回 recall_ratio 的股票
    Stage 2: 回归（预测收益大小）→ 排序
    """
    df = df.sort_values("date").copy()
    df["year_month"] = df["date"].dt.to_period("M")
    months = sorted(df["year_month"].unique())
    
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
        
        # Stage 1: 二分类（涨/跌）
        y_tr_bin = (y_tr > 0).astype(int)
        y_val_bin = (y_val > 0).astype(int)
        
        clf_lgb = train_classifier(X_tr, y_tr_bin, X_val, y_val_bin, feature_cols, "lightgbm")
        clf_xgb = train_classifier(X_tr, y_tr_bin, X_val, y_val_bin, feature_cols, "xgboost")
        
        prob_lgb = predict_model(clf_lgb, X_test)
        prob_xgb = predict_model(clf_xgb, X_test)
        prob_avg = (prob_lgb + prob_xgb) / 2
        
        # 选 recall_ratio 的股票
        n_recall = int(len(X_test) * recall_ratio)
        recall_idx = np.argsort(prob_avg)[-n_recall:]
        
        if len(recall_idx) < 10:
            continue
        
        # Stage 2: 回归（只在 recall 池里训练）
        X_tr_recall = X_tr[y_tr_bin == 1]  # 只用正样本训练
        y_tr_recall = y_tr[y_tr_bin == 1]
        X_val_recall = X_val[y_val_bin == 1]
        y_val_recall = y_val[y_val_bin == 1]
        
        if len(X_tr_recall) < 10 or len(X_val_recall) < 10:
            # 如果正样本太少，用全部样本
            X_tr_recall, y_tr_recall = X_tr, y_tr
            X_val_recall, y_val_recall = X_val, y_val
        
        reg_lgb = train_regressor(X_tr_recall, y_tr_recall, X_val_recall, y_val_recall, feature_cols, "lightgbm")
        reg_xgb = train_regressor(X_tr_recall, y_tr_recall, X_val_recall, y_val_recall, feature_cols, "xgboost")
        
        # 预测 recall 池
        X_recall = X_test[recall_idx]
        pred_lgb = predict_model(reg_lgb, X_recall)
        pred_xgb = predict_model(reg_xgb, X_recall)
        pred_avg = (pred_lgb + pred_xgb) / 2
        
        # 最终得分 = Stage1 概率 * Stage2 预测
        final_score = prob_avg[recall_idx] * pred_avg
        
        # 构建预测结果
        pred = test_df.iloc[recall_idx][["date", "ts_code", "stock_return", "benchmark_return", "excess_return", "industry"]].copy()
        pred["predicted"] = final_score
        pred["prob_up"] = prob_avg[recall_idx]
        pred["pred_return"] = pred_avg
        
        all_preds.append(pred)
        print(f"  {test_month}: recall={len(recall_idx)}, selected={len(pred)}")
    
    if not all_preds:
        return pd.DataFrame()
    
    return pd.concat(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--train-window-months", type=int, default=6)
    parser.add_argument("--recall-ratio", type=float, default=0.5)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    
    df = load_dataset(args.horizon, args.dataset)
    feature_cols = get_feature_cols(df)
    print(f"[two_stage] Dataset: {len(df)} rows, features: {len(feature_cols)}")
    
    result = walk_forward_two_stage(
        df, feature_cols,
        train_window_months=args.train_window_months,
        recall_ratio=args.recall_ratio,
    )
    
    if result.empty:
        print("[two_stage] No predictions generated")
        return
    
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"predictions_h{args.horizon}_two_stage.parquet"
    output_path = PRED_DIR / output_name
    result.to_parquet(output_path, index=False)
    print(f"[two_stage] Saved {len(result)} predictions to {output_path}")


if __name__ == "__main__":
    main()
