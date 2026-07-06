"""
召回层过滤实验

在已有 composite 数据集上应用召回规则，重新训练 GBDT 回归模型并输出预测。
目的是验证：通过质量/波动率/市值等规则过滤后，排序模型的 top-20 胜率是否提升。

用法：
python scripts/recall_filter_trainer.py \
  --dataset memory/dataset/features_h10_composite.parquet \
  --filters roe:0.2:1 profit_growth:0.2:1 volatility_20:0:0.8 total_mv:0.2:1
"""
import sys
import os
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_trainer import load_dataset, get_feature_cols, walk_forward_predict

PRED_DIR = Path("memory/predictions")


def parse_filters(raw: str) -> dict:
    """
    解析命令行过滤规则。
    格式: col:min:max,col2:min:max
    用 'none' 表示不限制该侧。
    """
    filters = {}
    if not raw:
        return filters
    for part in raw.split(","):
        tokens = part.split(":")
        if len(tokens) != 3:
            raise ValueError(f"Invalid filter: {part}")
        col, mn, mx = tokens
        bounds = {}
        if mn.lower() != "none":
            bounds["min"] = float(mn)
        if mx.lower() != "none":
            bounds["max"] = float(mx)
        filters[col.strip()] = bounds
    return filters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--model-type", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--train-window-months", type=int, default=12)
    parser.add_argument("--target", default="excess_return")
    parser.add_argument("--filters", default="",
                        help="召回过滤规则，例如 roe:0.2:1,volatility_20:0:0.8")
    parser.add_argument("--output-name", default=None,
                        help="输出文件名，默认自动生成")
    args = parser.parse_args()

    df = load_dataset(args.horizon, args.dataset)
    feature_cols = get_feature_cols(df)
    filters = parse_filters(args.filters)
    print(f"[recall_filter_trainer] Filters: {filters}")

    pred_df = walk_forward_predict(
        df, feature_cols,
        train_window_months=args.train_window_months,
        model_type=args.model_type,
        target=args.target,
        objective="regression",
        recall_filters=filters,
    )

    if pred_df.empty:
        print("[recall_filter_trainer] No predictions generated")
        return

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        output_path = PRED_DIR / args.output_name
    else:
        tag = args.filters.replace(":", "_").replace(",", "-") or "nofilter"
        output_path = PRED_DIR / f"predictions_h{args.horizon}_recall_{tag}.parquet"
    pred_df.to_parquet(output_path, index=False)
    print(f"[recall_filter_trainer] Saved {len(pred_df)} predictions to {output_path}")

    # 简单诊断
    top20 = pred_df.groupby("date").apply(lambda g: g.sort_values("predicted", ascending=False).head(20), include_groups=False)
    if not top20.empty:
        avg_excess = top20["excess_return"].mean()
        pos_ratio = (top20["excess_return"] > 0).mean()
        print(f"\nDiagnostic: top20 avg excess={avg_excess:+.4f}, positive ratio={pos_ratio:.2%}")


if __name__ == "__main__":
    main()
