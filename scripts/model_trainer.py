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
import macro_timing

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


def _compute_group_counts(df: pd.DataFrame, id_col: str = "date") -> np.ndarray:
    """为 LightGBM LambdaRank 计算 query group 大小（按日期分组）。"""
    return df.groupby(id_col).size().values


def _make_rank_label(df: pd.DataFrame, target: str, n_bins: int = 5) -> np.ndarray:
    """把连续目标转换为每个查询内的整数 relevance 标签（0~n_bins-1），用于 LambdaRank。"""
    df = df.copy()
    df["_rank_pct"] = df.groupby("date")[target].rank(pct=True, method="first")
    df["_label"] = (df["_rank_pct"] * n_bins).clip(upper=n_bins - 1).astype(int)
    return df["_label"].values


def train_gbdt(X_train, y_train, X_val, y_val, feature_cols,
               model_type: str = "lightgbm",
               num_rounds: int = 500,
               early_stopping: int = 30,
               target: str = "excess_return",
               objective: str = "regression",
               train_group: np.ndarray = None,
               val_group: np.ndarray = None):
    if model_type == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm not installed")
        if objective == "lambdarank":
            train_data = lgb.Dataset(X_train, label=y_train, group=train_group)
            valid_data = lgb.Dataset(X_val, label=y_val, group=val_group, reference=train_data)
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5, 10, 20],
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "seed": 42,
            }
        else:
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
        if objective == "lambdarank":
            raise ValueError("lambdarank not supported for xgboost in this trainer")
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
                         target: str = "excess_return",
                         objective: str = "regression") -> pd.DataFrame:
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
        tr_df = train_df[train_df["year_month"] != val_month].sort_values("date")
        val_df = train_df[train_df["year_month"] == val_month].sort_values("date")

        X_tr = tr_df[feature_cols].values
        X_val = val_df[feature_cols].values

        if objective == "lambdarank":
            y_tr = _make_rank_label(tr_df, target)
            y_val = _make_rank_label(val_df, target)
        else:
            y_tr = tr_df[target].values
            y_val = val_df[target].values

        kwargs = {"model_type": model_type, "target": target, "objective": objective}
        if objective == "lambdarank":
            kwargs["train_group"] = _compute_group_counts(tr_df)
            kwargs["val_group"] = _compute_group_counts(val_df)

        try:
            model = train_gbdt(X_tr, y_tr, X_val, y_val, feature_cols, **kwargs)
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


def assign_regime(df: pd.DataFrame, macro_dataset: str,
                  pos_thr: float = 0.3, neg_thr: float = -0.3) -> pd.DataFrame:
    """为每个样本按当日宏观状态打上 regime 标签。"""
    df = df.copy()
    df = macro_timing.load_macro_features(df, macro_dataset)
    df["regime_score"] = df.apply(macro_timing.compute_regime_score, axis=1)
    conditions = [df["regime_score"] > pos_thr, df["regime_score"] < neg_thr]
    choices = ["strong", "weak"]
    df["regime"] = np.select(conditions, choices, default="neutral")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def _train_model_for_fold(tr_df, val_df, feature_cols, model_type, target, objective):
    """辅助函数：为当前 fold 训练一个模型（支持回归/LambdaRank）。"""
    X_tr = tr_df[feature_cols].values
    X_val = val_df[feature_cols].values

    if objective == "lambdarank":
        y_tr = _make_rank_label(tr_df, target)
        y_val = _make_rank_label(val_df, target)
    else:
        y_tr = tr_df[target].values
        y_val = val_df[target].values

    kwargs = {"model_type": model_type, "target": target, "objective": objective}
    if objective == "lambdarank":
        kwargs["train_group"] = _compute_group_counts(tr_df)
        kwargs["val_group"] = _compute_group_counts(val_df)

    return train_gbdt(X_tr, y_tr, X_val, y_val, feature_cols, **kwargs)


def walk_forward_predict_by_regime(df: pd.DataFrame, feature_cols: list,
                                   macro_dataset: str,
                                   train_window_months: int = 12,
                                   model_type: str = "lightgbm",
                                   target: str = "excess_return",
                                   objective: str = "regression",
                                   pos_thr: float = 0.3,
                                   neg_thr: float = -0.3,
                                   min_train_groups: int = 10) -> pd.DataFrame:
    """
    按宏观 regime 分模型滚动训练 + walk-forward 预测。

    每个 fold 训练：
    - 一个全局 fallback 模型；
    - 每个在训练窗口中出现且样本数足够的 regime（strong/neutral/weak）一个专用模型。
    预测时根据测试日期的 regime 选择对应模型，缺失则回退到全局模型。
    """
    df = assign_regime(df, macro_dataset, pos_thr, neg_thr)
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

        val_month = train_months[-1]

        # 全局 fallback 模型
        tr_global = train_df[train_df["year_month"] != val_month].sort_values("date")
        val_global = train_df[train_df["year_month"] == val_month].sort_values("date")
        try:
            global_model = _train_model_for_fold(tr_global, val_global, feature_cols,
                                                  model_type, target, objective)
        except Exception as e:
            print(f"[model_trainer] global model failed for {test_month}: {e}")
            continue

        # 每个 regime 训练专用模型
        regime_models = {}
        for regime in ["strong", "neutral", "weak"]:
            tr_reg = train_df[(train_df["year_month"] != val_month) &
                              (train_df["regime"] == regime)].sort_values("date")
            val_reg = train_df[(train_df["year_month"] == val_month) &
                               (train_df["regime"] == regime)].sort_values("date")
            n_groups = tr_reg["date"].nunique()
            if n_groups < min_train_groups or val_reg.empty:
                continue
            try:
                regime_models[regime] = _train_model_for_fold(
                    tr_reg, val_reg, feature_cols, model_type, target, objective
                )
                print(f"[model_trainer] {test_month} regime={regime}: trained on {n_groups} dates")
            except Exception as e:
                print(f"[model_trainer] regime={regime} failed for {test_month}: {e}")

        # 预测：按日期选择对应 regime 模型
        pred_parts = []
        for regime, grp in test_df.groupby("regime"):
            model = regime_models.get(regime, global_model)
            X_test = grp[feature_cols].values
            if model_type == "xgboost":
                dtest = xgb.DMatrix(X_test)
                preds = model.predict(dtest)
            else:
                preds = model.predict(X_test, num_iteration=model.best_iteration)
            pred_df = grp[["date", "ts_code", "excess_return", "stock_return",
                           "benchmark_return", "industry", "regime"]].copy()
            pred_df["predicted"] = preds
            pred_df["train_end_month"] = str(train_months[-1])
            pred_parts.append(pred_df)

        if pred_parts:
            all_preds.append(pd.concat(pred_parts, ignore_index=True))

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
    parser.add_argument("--objective", choices=["regression", "lambdarank"], default="regression",
                        help="训练目标：回归 或 LambdaRank（仅 LightGBM）")
    parser.add_argument("--dataset", default=None, help="Path to parquet dataset (default: memory/dataset/features_h{horizon}.parquet)")
    args = parser.parse_args()

    df = load_dataset(args.horizon, args.dataset)
    feature_cols = get_feature_cols(df)
    print(f"[model_trainer] Dataset: {len(df)} rows, features: {feature_cols}")

    if args.mode == "walkforward":
        pred_df = walk_forward_predict(df, feature_cols,
                                       train_window_months=args.train_window_months,
                                       model_type=args.model_type,
                                       target=args.target,
                                       objective=args.objective)
        output_name = f"predictions_h{args.horizon}_walkforward_{args.target}_{args.objective}.parquet"
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
