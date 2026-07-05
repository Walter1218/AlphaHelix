"""
AlphaHelix GBDT 模型重训练脚本

用于生产环境定期重训练：读取最新 dataset，用全部历史数据训练一个模型，
保存到 memory/models/ 供 screen.py --use-gbdt 实时调用。

注意：本脚本训练时只能用训练截止日期之前的数据；训练完成后才能用于未来日期的选股。
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_trainer import (
    load_dataset, get_feature_cols, train_gbdt, save_model,
    _make_rank_label, _compute_group_counts,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="parquet 数据集路径（默认 memory/dataset/features_h10.parquet）")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--train-end", default=None, help="训练截止日期 YYYYMMDD；默认使用数据集最后一天")
    parser.add_argument("--model-type", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--target", choices=["excess_return", "stock_return"], default="excess_return")
    parser.add_argument("--objective", choices=["regression", "lambdarank"], default="regression",
                        help="训练目标：回归 或 LambdaRank（仅 LightGBM）")
    parser.add_argument("--output-name", default=None, help="模型文件名，默认 gbdt_latest_h{horizon}.{model_type}.txt")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="从训练集中划出验证集的比例")
    args = parser.parse_args()

    df = load_dataset(args.horizon, args.dataset)
    feature_cols = get_feature_cols(df)

    if args.train_end:
        train_end = pd.to_datetime(args.train_end, format="%Y%m%d")
        train_df = df[df["date"] <= train_end].copy()
    else:
        train_df = df.copy()

    if train_df.empty:
        raise ValueError("训练集为空，请检查 dataset 和 train-end 参数")

    # 按日期排序后，尾部 val-ratio 作为验证集（避免随机抽样导致数据泄露）
    train_df = train_df.sort_values("date").reset_index(drop=True)
    n_val = int(len(train_df) * args.val_ratio)
    if n_val < 100:
        n_val = min(100, len(train_df) // 5)

    tr_df = train_df.iloc[:-n_val].sort_values("date")
    val_df = train_df.iloc[-n_val:].sort_values("date")

    X_tr = tr_df[feature_cols].values
    X_val = val_df[feature_cols].values

    if args.objective == "lambdarank":
        y_tr = _make_rank_label(tr_df, args.target)
        y_val = _make_rank_label(val_df, args.target)
    else:
        y_tr = tr_df[args.target].values
        y_val = val_df[args.target].values

    print(f"[retrain_gbdt] Training on {len(train_df)} samples, features={len(feature_cols)}")
    print(f"[retrain_gbdt] Train={len(tr_df)}, Val={len(val_df)}, target={args.target}, objective={args.objective}")

    kwargs = {"model_type": args.model_type, "target": args.target, "objective": args.objective}
    if args.objective == "lambdarank":
        kwargs["train_group"] = _compute_group_counts(tr_df)
        kwargs["val_group"] = _compute_group_counts(val_df)

    model = train_gbdt(X_tr, y_tr, X_val, y_val, feature_cols, **kwargs)

    model_dir = Path("memory/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"gbdt_latest_h{args.horizon}.{args.model_type}.txt"
    model_path = model_dir / output_name
    save_model(model, feature_cols, str(model_path), model_type=args.model_type)

    # 输出特征重要性（LightGBM 可用）
    try:
        if args.model_type == "lightgbm":
            import lightgbm as lgb
            booster = model if hasattr(model, "feature_importance") else lgb.Booster(model_file=str(model_path))
            imp = booster.feature_importance(importance_type="gain")
            print("\n=== Top 10 Feature Importance (gain) ===")
            for name, score in sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:10]:
                print(f"  {name:30s} {score:.1f}")
    except Exception as e:
        print(f"[retrain_gbdt] feature importance skipped: {e}")

    print(f"\n[retrain_gbdt] Saved model to {model_path}")


if __name__ == "__main__":
    main()
