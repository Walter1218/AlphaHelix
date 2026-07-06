"""
两阶段选股框架：召回 + 排序

- 召回层：用二分类模型判断“是否值得进入候选池”，输出 recall_prob；
- 排序层：在召回池中再用一个二分类模型判断“跑赢基准的概率”，输出 rank_prob（0~1）；
- 最终持仓按 rank_prob 取 top-N。

召回层可以用规则 + 轻量分类模型，排序层专注于方向胜率。
"""
import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_trainer import load_dataset, get_feature_cols, _train_model_for_fold

PRED_DIR = Path("memory/predictions")


def make_binary_target(df: pd.DataFrame, target: str = "excess_return",
                       quantile: float = None) -> pd.Series:
    """
    构造二分类标签。
    - quantile=None: excess_return > 0 为正样本；
    - quantile=0.2: 当日排名前 20% 为正样本（更严格）。
    """
    if quantile is None:
        return (df[target] > 0).astype(int)
    else:
        return (df.groupby("date")[target].rank(pct=True) >= (1 - quantile)).astype(int)


def walk_forward_two_stage(df: pd.DataFrame, feature_cols: list,
                           train_window_months: int = 12,
                           model_type: str = "lightgbm",
                           recall_quantile: float = None,
                           rank_quantile: float = None,
                           recall_ratio: float = 0.3,
                           min_recall_samples: int = 200) -> pd.DataFrame:
    """
    滚动训练两阶段模型。

    每个 fold：
    1. 用全部训练数据训练召回二分类模型；
    2. 在训练数据上取 recall_prob 最高的 recall_ratio 比例作为召回池；
    3. 在召回池上训练排序二分类模型；
    4. 对测试集：先预测 recall_prob，保留召回部分，再用排序模型预测 rank_prob。
    """
    df = df.sort_values("date").copy()
    df["year_month"] = df["date"].dt.to_period("M")
    df["is_win"] = make_binary_target(df, "excess_return", quantile=None)
    months = sorted(df["year_month"].unique())

    all_preds = []
    for i, test_month in enumerate(months):
        train_months = months[max(0, i - train_window_months):i]
        if len(train_months) < 3:
            continue

        train_df = df[df["year_month"].isin(train_months)].copy()
        test_df = df[df["year_month"] == test_month].copy()
        if train_df.empty or test_df.empty:
            continue

        val_month = train_months[-1]
        tr_global = train_df[train_df["year_month"] != val_month].sort_values("date")
        val_global = train_df[train_df["year_month"] == val_month].sort_values("date")

        # ---- 召回层 ----
        y_tr_recall = make_binary_target(tr_global, "excess_return", recall_quantile).values
        y_val_recall = make_binary_target(val_global, "excess_return", recall_quantile).values
        try:
            recall_model = _train_model_for_fold(
                tr_global.assign(is_win=y_tr_recall),
                val_global.assign(is_win=y_val_recall),
                feature_cols, model_type, "is_win", "binary"
            )
        except Exception as e:
            print(f"[two_stage] recall train failed for {test_month}: {e}")
            continue

        # 在训练集上预测召回概率，确定召回阈值
        X_train = train_df[feature_cols].values
        if model_type == "xgboost":
            import xgboost as xgb
            recall_train_prob = recall_model.predict(xgb.DMatrix(X_train))
        else:
            recall_train_prob = recall_model.predict(X_train, num_iteration=recall_model.best_iteration)
        train_df["recall_prob"] = recall_train_prob
        # 按日期取 top recall_ratio 作为召回池
        train_df["recall_rank"] = train_df.groupby("date")["recall_prob"].rank(pct=True, ascending=False)
        recalled_train = train_df[train_df["recall_rank"] <= recall_ratio].copy()
        if len(recalled_train) < min_recall_samples:
            print(f"[two_stage] {test_month}: too few recalled train samples ({len(recalled_train)}), skip rank model")
            recalled_train = train_df.copy()

        # ---- 排序层 ----
        tr_rank = recalled_train[recalled_train["year_month"] != val_month].sort_values("date")
        val_rank = recalled_train[recalled_train["year_month"] == val_month].sort_values("date")
        if tr_rank.empty or val_rank.empty:
            print(f"[two_stage] {test_month}: rank train/val empty, use recall scores")
            rank_model = recall_model
        else:
            y_tr_rank = make_binary_target(tr_rank, "excess_return", rank_quantile).values
            y_val_rank = make_binary_target(val_rank, "excess_return", rank_quantile).values
            try:
                rank_model = _train_model_for_fold(
                    tr_rank.assign(is_win=y_tr_rank),
                    val_rank.assign(is_win=y_val_rank),
                    feature_cols, model_type, "is_win", "binary"
                )
            except Exception as e:
                print(f"[two_stage] rank train failed for {test_month}: {e}")
                rank_model = recall_model

        # ---- 预测测试集 ----
        X_test = test_df[feature_cols].values
        if model_type == "xgboost":
            import xgboost as xgb
            recall_test_prob = recall_model.predict(xgb.DMatrix(X_test))
            rank_test_prob = rank_model.predict(xgb.DMatrix(X_test))
        else:
            recall_test_prob = recall_model.predict(X_test, num_iteration=recall_model.best_iteration)
            rank_test_prob = rank_model.predict(X_test, num_iteration=rank_model.best_iteration)

        test_df["recall_prob"] = recall_test_prob
        test_df["rank_prob"] = rank_test_prob
        test_df["predicted"] = test_df["rank_prob"]
        # 未召回的股票 rank_prob 清 0，不参与排序
        test_df["recall_rank"] = test_df.groupby("date")["recall_prob"].rank(pct=True, ascending=False)
        test_df.loc[test_df["recall_rank"] > recall_ratio, "predicted"] = 0.0

        pred_df = test_df[["date", "ts_code", "excess_return", "stock_return",
                           "benchmark_return", "industry", "recall_prob",
                           "rank_prob", "predicted"]].copy()
        pred_df["train_end_month"] = str(train_months[-1])
        all_preds.append(pred_df)

    if not all_preds:
        return pd.DataFrame()
    return pd.concat(all_preds, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="dataset parquet")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--model-type", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--train-window-months", type=int, default=12)
    parser.add_argument("--recall-quantile", type=float, default=None,
                        help="召回层正样本定义：前 quantile 比例，None 表示 excess_return>0")
    parser.add_argument("--rank-quantile", type=float, default=None,
                        help="排序层正样本定义：前 quantile 比例，None 表示 excess_return>0")
    parser.add_argument("--recall-ratio", type=float, default=0.3,
                        help="召回池占当日股票比例，默认 30%")
    parser.add_argument("--min-recall-samples", type=int, default=200)
    args = parser.parse_args()

    df = load_dataset(args.horizon, args.dataset)
    feature_cols = get_feature_cols(df)
    print(f"[two_stage] Dataset: {len(df)} rows, features: {len(feature_cols)}")

    pred_df = walk_forward_two_stage(
        df, feature_cols,
        train_window_months=args.train_window_months,
        model_type=args.model_type,
        recall_quantile=args.recall_quantile,
        rank_quantile=args.rank_quantile,
        recall_ratio=args.recall_ratio,
        min_recall_samples=args.min_recall_samples,
    )

    if pred_df.empty:
        print("[two_stage] No predictions generated")
        return

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PRED_DIR / f"predictions_h{args.horizon}_twostage_binary_{args.model_type}.parquet"
    pred_df.to_parquet(output_path, index=False)
    print(f"[two_stage] Saved {len(pred_df)} predictions to {output_path}")

    # 简单诊断
    print("\n=== Two-stage diagnostic ===")
    print(f"Recall pool avg size per date: {pred_df[pred_df['predicted'] > 0].groupby('date').size().mean():.0f}")
    top20 = pred_df.groupby("date").apply(lambda g: g.sort_values("predicted", ascending=False).head(20), include_groups=False)
    if not top20.empty:
        print(f"Top20 avg excess: {top20['excess_return'].mean():+.4f}")
        print(f"Top20 positive ratio: {(top20['excess_return'] > 0).mean():.2%}")


if __name__ == "__main__":
    main()
