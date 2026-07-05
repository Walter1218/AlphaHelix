"""
AlphaHelix 特征工程工具

- 截面 rank 标准化
- 行业/市值中性化
- 截尾（winsorize）
- 构建模型输入特征向量

所有操作仅使用当前截面的已知信息，无未来函数。
"""
import sys
import os
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def rank_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    对指定列做截面 rank 标准化，输出到 [0, 1]。
    缺失值填充为 0.5。
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df[col].rank(pct=True, na_option="keep")
        df[col] = df[col].fillna(0.5)
    return df


def winsorize_features(df: pd.DataFrame, cols: List[str],
                       lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    对指定列做截面截尾，避免极端值影响模型。
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df[col] = df[col].clip(lo, hi)
    return df


def neutralize_features(df: pd.DataFrame, feature_cols: List[str],
                        group_col: str = "industry",
                        size_col: str = "total_mv") -> pd.DataFrame:
    """
    对数值因子做行业/市值中性化。

    方法：对每个因子，在截面上用行业和 log(市值) 做线性回归，取残差。
    行业为 categorical，内部做 one-hot；缺失行业归为 "未知"。
    """
    df = df.copy()
    if group_col not in df.columns:
        df[group_col] = "未知"
    df[group_col] = df[group_col].fillna("未知").astype(str)

    if size_col in df.columns:
        df["_log_size"] = np.log1p(pd.to_numeric(df[size_col], errors="coerce").fillna(df[size_col].median()))
    else:
        df["_log_size"] = 0.0

    # one-hot 行业
    dummies = pd.get_dummies(df[group_col], prefix="_ind", drop_first=True)
    dummy_cols = dummies.columns.tolist()
    df = pd.concat([df, dummies], axis=1)

    X_cols = ["_log_size"] + dummy_cols
    X = df[X_cols].fillna(0).values

    for col in feature_cols:
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce").values
        valid = ~np.isnan(y)
        if valid.sum() < len(X_cols) + 5:
            df[col] = np.where(valid, y, 0.0)
            continue
        try:
            # 最小二乘：beta = (X'X)^-1 X'y
            Xv = X[valid]
            yv = y[valid]
            beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
            pred = X @ beta
            residual = y - pred
            df[col] = np.where(valid, residual, 0.0)
        except Exception:
            df[col] = np.where(valid, y, 0.0)

    df = df.drop(columns=["_log_size"] + dummy_cols, errors="ignore")
    return df


def build_numeric_features(df: pd.DataFrame,
                           feature_cols: Optional[List[str]] = None,
                           neutralize: bool = True,
                           rank: bool = True,
                           winsorize: bool = True) -> pd.DataFrame:
    """
    把原始因子 DataFrame 转换为模型可用的数值特征。

    处理顺序：截尾 → 中性化 → rank。
    """
    if feature_cols is None:
        feature_cols = [
            "mom_5", "mom_20", "mom_60", "mom_120",
            "risk_adj_mom", "relative_strength",
            "ep", "bp", "sp", "dv_ratio",
            "roe", "revenue_growth", "profit_growth", "ocf_growth",
            "net_mf_ratio", "net_mf_divergence",
            "sector_momentum", "relative_to_sector", "sector_breadth",
            "forecast_type_score", "forecast_pchange_mid", "express_diluted_roe",
            "reversal_score", "amount_ratio_5d", "volatility_20", "liquidity", "total_mv",
            # Phase 2
            "margin_total_balance", "margin_financing_ratio",
            "northbound_net", "northbound_net_5d",
            "top_list_flag", "top_list_net_amount", "top_list_amount_rate", "top_list_turnover_rate", "top_list_pct_change",
            "days_to_disclosure", "days_since_disclosure",
            # Composite factors
            "defensive_quality", "smart_money_per_risk", "quality_growth", "value_quality",
            "earnings_surprise_momentum", "growth_consistency", "risk_adj_momentum_20",
        ]
    # 只保留存在的列
    feature_cols = [c for c in feature_cols if c in df.columns]

    df = df.copy()
    if winsorize:
        df = winsorize_features(df, feature_cols)
    if neutralize:
        df = neutralize_features(df, feature_cols)
    if rank:
        df = rank_features(df, feature_cols)
    return df


def add_composite_factors(df: pd.DataFrame) -> pd.DataFrame:
    """添加 factor_miner 筛选出的高 IC 复合因子。输入应为原始特征 DataFrame。"""
    df = df.copy()
    expressions = {
        "defensive_quality": "roe / (volatility_20 + 1e-9)",
        "smart_money_per_risk": "net_mf_ratio / (volatility_20 + 1e-9)",
        "quality_growth": "roe * profit_growth",
        "value_quality": "dv_ratio * roe",
        "earnings_surprise_momentum": "(forecast_pchange_mid - mom_20) / (volatility_20 + 1e-9)",
        "growth_consistency": "revenue_growth * profit_growth * ocf_growth",
        "risk_adj_momentum_20": "mom_20 / (volatility_20 + 1e-9)",
    }
    for name, expr in expressions.items():
        df[name] = pd.eval(expr, local_dict=df, engine="python")
        df[name] = df[name].replace([np.inf, -np.inf], np.nan)
        lo = df[name].quantile(0.01)
        hi = df[name].quantile(0.99)
        df[name] = df[name].clip(lo, hi)
    return df


def extract_feature_vector(row: dict, feature_cols: List[str]) -> np.ndarray:
    """从一行记录中提取特征向量，缺失填 0。"""
    values = []
    for col in feature_cols:
        v = row.get(col)
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            values.append(0.0)
    return np.array(values, dtype=float)


if __name__ == "__main__":
    # simple sanity check
    df = pd.DataFrame({
        "ts_code": ["a", "b", "c", "d"],
        "industry": ["x", "x", "y", "y"],
        "total_mv": [100, 200, 300, 400],
        "mom_20": [0.1, 0.2, np.nan, -0.1],
        "ep": [0.05, 0.1, 0.08, 0.12],
    })
    out = build_numeric_features(df)
    print(out[["ts_code", "mom_20", "ep"]])
