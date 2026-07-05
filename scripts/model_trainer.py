"""
AlphaHelix GBDT 预测模型训练

- 读取 build_dataset.py 生成的 parquet 数据集
- 滚动训练 + walk-forward 预测
- 输出预测结果与特征重要性

目标变量：未来 H 日相对沪深300 的超额收益 excess_return
"""
import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 支持 lightgbm 或 xgboost
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _trace import trace_event, new_run

DATASET_DIR = Path("memory/dataset")
MODEL_DIR = Path("memory/models")
PRED_DIR = Path("memory/predictions")


def load_dataset(horizon: int, dataset_path: str = None):
    if dataset_path:
        path = Path(dataset_path)
    else:
        path = DATASET_DIR / f"features_h{horizon}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run build_dataset.py first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"stock_return", "benchmark_return", "excess_return"}
    return [c for c in numeric if c not in exclude]


def train_gbdt(X_train, y_train, X_val, y_val, feature_cols,
               model_type: str = "lightgbm",
               num_rounds: int = 500,
               early_stopping: int = 30,
               target: str = "excess_return"):
    if model_type == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm not installed")
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
        }
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_rounds,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
    elif model_type == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_rounds,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model


def walk_forward_predict(df: pd.DataFrame, feature_cols: list,
                         train_window_months: int = 12,
                         model_type: str = "lightgbm",
                         target: str = "excess_return") -> pd.DataFrame:
    """
    滚动训练 + walk-forward 预测。

    每月用过去 N 个月的数据训练模型，预测下一个月所有样本。
    """
    df = df.sort_values("date").copy()
    df["year_month"] = df["date"].dt.to_period("M")
    months = sorted(df["year_month"].unique())

    all_preds = []
    for i, test_month in enumerate(months):
        # 找到训练窗口：test_month 之前的 train_window_months 个月
        train_months = months[max(0, i - train_window_months):i]
        if len(train_months) < 3:
            continue

        train_df = df[df["year_month"].isin(train_months)]
        test_df = df[df["year_month"] == test_month]
        if train_df.empty or test_df.empty:
            continue

        # 训练/验证切分：训练月最后一个月作为验证
        val_month = train_months[-1]
        tr_df = train_df[train_df["year_month"] != val_month]
        val_df = train_df[train_df["year_month"] == val_month]

        X_tr = tr_df[feature_cols].values
        y_tr = tr_df[target].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target].values

        try:
            model = train_gbdt(X_tr, y_tr, X_val, y_val, feature_cols, model_type=model_type, target=target)
        except Exception as e:
            print(f"[model_trainer] train failed for {test_month}: {e}")
            continue

        X_test = test_df[feature_cols].values
        if model_type == "xgboost":
            dtest = xgb.DMatrix(X_test)
            preds = model.predict(dtest)
        else:
            preds = model.predict(X_test, num_iteration=model.best_iteration)

        pred_df = test_df[["date", "ts_code", "excess_return", "stock_return", "benchmark_return", "industry"]].copy()
        pred_df["predicted"] = preds
        pred_df["train_end_month"] = str(train_months[-1])
        all_preds.append(pred_df)

    if not all_preds:
        return pd.DataFrame()
    return pd.concat(all_preds, ignore_index=True)


def simple_split_predict(df: pd.DataFrame, feature_cols: list,
                         train_end: str = "20241231",
                         model_type: str = "lightgbm",
                         target: str = "excess_return") -> pd.DataFrame:
    """简单切分：用 2024 及之前训练，预测 2025 及之后。用于快速验证。"""
    train_mask = df["date"] <= pd.to_datetime(train_end, format="%Y%m%d")
    train_df = df[train_mask]
    test_df = df[~train_mask]

    # 训练集中再切 20% 做验证
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_val = int(len(train_df) * 0.2)
    tr_df = train_df.iloc[n_val:]
    val_df = train_df.iloc[:n_val]

    X_tr = tr_df[feature_cols].values
    y_tr = tr_df[target].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target].values

    model = train_gbdt(X_tr, y_tr, X_val, y_val, feature_cols, model_type=model_type, target=target)

    X_test = test_df[feature_cols].values
    if model_type == "xgboost":
        dtest = xgb.DMatrix(X_test)
        preds = model.predict(dtest)
    else:
        preds = model.predict(X_test, num_iteration=model.best_iteration)

    pred_df = test_df[["date", "ts_code", "excess_return", "stock_return", "benchmark_return", "industry"]].copy()
    pred_df["predicted"] = preds
    return pred_df, model


def save_model(model, feature_cols: list, path: str, model_type: str = "lightgbm"):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if model_type == "lightgbm":
        model.save_model(path)
    elif model_type == "xgboost":
        model.save_model(path)
    # 保存特征名
    meta_path = path.replace(".txt", "_meta.json")
    if meta_path == path:
        meta_path = path + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols, "model_type": model_type}, f, ensure_ascii=False, indent=2)


def load_saved_model(path: str, model_type: str = "lightgbm"):
    if model_type == "lightgbm":
        return lgb.Booster(model_file=path)
    elif model_type == "xgboost":
        return xgb.Booster(model_file=path)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True, help="Target horizon in days")
    parser.add_argument("--mode", choices=["walkforward", "split"], default="walkforward")
    parser.add_argument("--model-type", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--train-end", default="20241231", help="For split mode")
    parser.add_argument("--train-window-months", type=int, default=12)
    parser.add_argument("--target", choices=["excess_return", "stock_return"], default="excess_return")
    parser.add_argument("--dataset", default=None, help="Path to parquet dataset (default: memory/dataset/features_h{horizon}.parquet)")
    args = parser.parse_args()

    df = load_dataset(args.horizon, args.dataset)
    feature_cols = get_feature_cols(df)
    print(f"[model_trainer] Dataset: {len(df)} rows, features: {feature_cols}")

    if args.mode == "walkforward":
        pred_df = walk_forward_predict(df, feature_cols,
                                       train_window_months=args.train_window_months,
                                       model_type=args.model_type,
                                       target=args.target)
        output_name = f"predictions_h{args.horizon}_walkforward_{args.target}.parquet"
    else:
        pred_df, model = simple_split_predict(df, feature_cols,
                                              train_end=args.train_end,
                                              model_type=args.model_type,
                                              target=args.target)
        # 保存 split 模式下的模型
        model_path = str(MODEL_DIR / f"gbdt_h{args.horizon}_split_{args.target}.{args.model_type}.txt")
        save_model(model, feature_cols, model_path, model_type=args.model_type)
        output_name = f"predictions_h{args.horizon}_split_{args.target}.parquet"

    if pred_df.empty:
        print("[model_trainer] No predictions generated")
        return

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PRED_DIR / output_name
    pred_df.to_parquet(output_path, index=False)
    print(f"[model_trainer] Saved {len(pred_df)} predictions to {output_path}")


if __name__ == "__main__":
    main()
