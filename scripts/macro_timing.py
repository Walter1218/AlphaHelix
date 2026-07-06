"""
宏观择时工具函数

提供基于北向资金和融资融券的 regime_score / position_scale 计算。
可与 portfolio_backtest.py 或 apply_market_timing.py 共用，避免循环导入。
"""
from pathlib import Path

import numpy as np
import pandas as pd


def load_macro_features(pred_df: pd.DataFrame, macro_dataset: str = None) -> pd.DataFrame:
    """加载宏观特征并与预测表按 date 合并。"""
    if macro_dataset and Path(macro_dataset).exists():
        df = pd.read_parquet(macro_dataset)
    else:
        raise FileNotFoundError(f"Macro dataset not found: {macro_dataset}")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.strftime("%Y%m%d")

    macro_cols = ["date", "margin_total_balance", "margin_change_5d", "margin_change_20d",
                  "northbound_net_today", "northbound_net_5d_sum", "northbound_net_20d_sum",
                  "northbound_net_20d_zscore", "northbound_net_5d_vs_20d"]
    macro_cols = [c for c in macro_cols if c in df.columns]
    macro = df[macro_cols].drop_duplicates(subset=["date"])
    return pred_df.merge(macro, on="date", how="left")


def compute_regime_score(row: pd.Series) -> float:
    """基于北向和融资融券计算 -1~1 的 regime 分数。"""
    nb = row.get("northbound_net_20d_zscore", 0)
    margin = row.get("margin_change_5d", 0)

    nb_score = np.clip(nb / 2.0, -1, 1) if pd.notna(nb) else 0.0
    margin_score = np.clip(margin / 0.05, -1, 1) if pd.notna(margin) else 0.0

    return 0.6 * nb_score + 0.4 * margin_score


def compute_position_scales(pred_df: pd.DataFrame, macro_dataset: str = None,
                            date_col: str = "date",
                            regime_threshold: float = None) -> dict:
    """
    返回 date -> position_scale 的字典。
    若未提供 macro_dataset，则所有日期 scale=1.0。

    regime_threshold: 若指定，当 regime_score <= threshold 时空仓（scale=0），
    否则满仓（scale=1）。未指定时使用连续缩放 scale = clip(1 + regime_score, 0, 1)。
    """
    if not macro_dataset:
        return {}

    df = load_macro_features(pred_df, macro_dataset)
    df["regime_score"] = df.apply(compute_regime_score, axis=1)
    if regime_threshold is not None:
        df["position_scale"] = (df["regime_score"] > regime_threshold).astype(float)
    else:
        df["position_scale"] = (1 + df["regime_score"]).clip(0, 1)

    # 同一日期所有股票共享一个 scale，取第一条
    scale_map = (
        df.groupby(date_col)["position_scale"]
        .first()
        .to_dict()
    )
    return {str(d): float(v) for d, v in scale_map.items()}
