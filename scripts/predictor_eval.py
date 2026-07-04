"""
AlphaHelix 预测模型评估

计算：
- rank IC（截面 Spearman 相关）
- ICIR
- IC 序列统计
- 分位数收益（按预测得分分 5/10 组）
- 分组收益单调性
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PRED_DIR = Path("memory/predictions")


def compute_rank_ic(df: pd.DataFrame) -> pd.DataFrame:
    """逐期计算 rank IC。"""
    ic_records = []
    for date, g in df.groupby("date"):
        if len(g) < 5:
            continue
        ic, pval = stats.spearmanr(g["predicted"], g["excess_return"])
        ic_records.append({
            "date": date,
            "rank_ic": ic,
            "p_value": pval,
            "n": len(g),
        })
    return pd.DataFrame(ic_records)


def compute_quantile_returns(df: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """按预测得分分 n_quantiles 组，计算每组平均 excess_return。"""
    df = df.copy()
    df["quantile"] = df.groupby("date")["predicted"].transform(
        lambda x: pd.qcut(x.rank(method="first"), n_quantiles, labels=False, duplicates="drop")
    )
    return df.groupby("quantile").agg(
        avg_excess=("excess_return", "mean"),
        avg_return=("stock_return", "mean"),
        count=("excess_return", "count"),
    ).reset_index()


def evaluate(pred_path: str):
    df = pd.read_parquet(pred_path)
    if df.empty:
        print("[predictor_eval] Empty predictions")
        return

    # 总体 rank IC
    overall_ic, overall_p = stats.spearmanr(df["predicted"], df["excess_return"])

    # 逐期 rank IC
    ic_df = compute_rank_ic(df)
    mean_ic = ic_df["rank_ic"].mean()
    std_ic = ic_df["rank_ic"].std()
    ir = mean_ic / std_ic if std_ic > 0 else 0
    positive_ratio = (ic_df["rank_ic"] > 0).mean()

    # 分位数收益
    q5 = compute_quantile_returns(df, n_quantiles=5)
    q10 = compute_quantile_returns(df, n_quantiles=10)

    print("\n=== Predictor Evaluation ===")
    print(f"Predictions: {len(df):,} rows, dates: {df['date'].nunique()}")
    print(f"Overall rank IC: {overall_ic:.4f} (p={overall_p:.4f})")
    print(f"Mean rank IC: {mean_ic:.4f}")
    print(f"IC std: {std_ic:.4f}")
    print(f"ICIR: {ir:.4f}")
    print(f"Positive IC ratio: {positive_ratio:.2%}")

    print("\n--- Quintile Returns ---")
    print(q5.to_string(index=False))

    print("\n--- Decile Returns ---")
    print(q10.to_string(index=False))

    # 保存 IC 序列
    ic_path = Path(pred_path).parent / f"{Path(pred_path).stem}_ic.parquet"
    ic_df.to_parquet(ic_path, index=False)
    print(f"\n[predictor_eval] Saved IC series to {ic_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", required=True, help="Path to predictions parquet")
    args = parser.parse_args()
    evaluate(args.pred_path)


if __name__ == "__main__":
    main()
