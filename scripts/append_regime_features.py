"""
在 composite-only 数据集上追加 regime / 宏观交互特征。

输入：
- 基础数据集（如 memory/dataset/features_h10_composite.parquet）
- 宏观数据集（如 memory/dataset/features_h10_composite_phase3.parquet）

追加字段：
- regime_score: 综合北向+融资融券的宏观情绪分数 [-1, 1]
- regime_strong / regime_neutral / regime_weak: one-hot regime 标签
- northbound_net_20d_zscore, margin_change_5d, margin_change_20d: 原始宏观特征
- 以及若干显式交互项（动量/质量/价值 × regime_score）

输出新的 parquet，供 model_trainer.py 直接训练。
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import macro_timing


def append_regime_features(base_path: str, macro_path: str, output_path: str,
                           pos_thr: float = 0.3, neg_thr: float = -0.3):
    base = pd.read_parquet(base_path)
    base["date"] = pd.to_datetime(base["date"], format="%Y%m%d")

    # 计算 regime_score 和标签
    df = macro_timing.load_macro_features(base, macro_path)
    df["regime_score"] = df.apply(macro_timing.compute_regime_score, axis=1)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    conditions = [df["regime_score"] > pos_thr, df["regime_score"] < neg_thr]
    choices = ["strong", "weak"]
    df["regime"] = np.select(conditions, choices, default="neutral")

    # 追加宏观原始特征
    macro_cols = ["northbound_net_20d_zscore", "margin_change_5d", "margin_change_20d"]
    for c in macro_cols:
        if c in df.columns:
            base[c] = df[c].values

    base["regime_score"] = df["regime_score"].values
    for reg in ["strong", "neutral", "weak"]:
        base[f"regime_{reg}"] = (df["regime"] == reg).astype(int).values

    # 显式交互项：让模型更容易学到风格切换
    interaction_candidates = ["mom_20", "mom_60", "roe", "net_mf_ratio",
                              "relative_strength", "quality_growth",
                              "smart_money_per_risk", "defensive_quality"]
    for c in interaction_candidates:
        if c in base.columns:
            base[f"{c}_x_regime"] = base[c].values * base["regime_score"].values

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(output_path, index=False)
    print(f"[append_regime_features] Saved {len(base)} rows with {len(base.columns)} cols to {output_path}")
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="memory/dataset/features_h10_composite.parquet")
    parser.add_argument("--macro", default="memory/dataset/features_h10_composite_phase3.parquet")
    parser.add_argument("--output", default="memory/dataset/features_h10_composite_regime.parquet")
    parser.add_argument("--pos-thr", type=float, default=0.3)
    parser.add_argument("--neg-thr", type=float, default=-0.3)
    args = parser.parse_args()

    append_regime_features(args.base, args.macro, args.output, args.pos_thr, args.neg_thr)


if __name__ == "__main__":
    main()
