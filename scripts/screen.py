"""
AlphaHelix 因子初筛脚本
支持多策略：momentum_value_hybrid、quality_growth、contrarian、event_driven。
基于 Tushare 数据计算动量、估值、质量、资金、流动性、事件、反转、行业相对强度因子，输出候选股票池。
"""
import sys
import os
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 把 scripts 目录加入路径以导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import tushare_call, get_trade_date_before, is_st_historical, concurrent_map
from market_regime import classify_regime, regime_to_strategy
from _trace import trace_event

warnings.filterwarnings("ignore")

# 默认值
MIN_AVG_AMOUNT = 50000
MAX_VOLATILITY = 0.07
LIST_DAYS_MIN = 120
UNIVERSE_SAMPLE = int(os.environ.get("AH_UNIVERSE_SAMPLE", 400))
PASS1_TOP_K = int(os.environ.get("AH_PASS1_TOP_K", 80))      # 第一轮初筛后进入第二轮深度计算的候选数
MIN_ROE = 0.05        # ROE 过滤阈值
MAX_SECTOR_PCT = 0.40 # 单一行业权重上限
SKIP_ST_CHECK = os.environ.get("AH_SKIP_ST_CHECK", "").lower() in ("1", "true", "yes")

# 内存缓存：按 trade_date 缓存截面数据，避免反复读缓存文件
_daily_cache: dict = {}
_moneyflow_cache: dict = {}


def _load_daily_by_date(date: str) -> pd.DataFrame:
    """加载某交易日的全市场日线（优先内存缓存）。"""
    if date not in _daily_cache:
        df = tushare_call("daily", {"trade_date": date})
        if not df.empty:
            df["trade_date"] = df["trade_date"].astype(str)
            df["ts_code"] = df["ts_code"].astype(str)
        _daily_cache[date] = df
    return _daily_cache[date]


def _load_moneyflow_by_date(date: str) -> pd.DataFrame:
    """加载某交易日的全市场资金流向（优先内存缓存）。"""
    if date not in _moneyflow_cache:
        df = tushare_call("moneyflow", {"trade_date": date})
        if not df.empty:
            df["trade_date"] = df["trade_date"].astype(str)
            df["ts_code"] = df["ts_code"].astype(str)
        _moneyflow_cache[date] = df
    return _moneyflow_cache[date]


def load_daily_window(end_date: str, days: int) -> pd.DataFrame:
    """加载 [end_date-days+1, end_date] 的日线窗口并合并。"""
    dates = []
    d = end_date
    collected = 0
    while collected < days:
        dates.append(d)
        d = get_trade_date_before(d, days=1)
        collected += 1
    dates.reverse()
    dfs = concurrent_map(_load_daily_by_date, dates)
    valid = [df for df in dfs if not df.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


def load_moneyflow_window(end_date: str, days: int) -> pd.DataFrame:
    """加载 [end_date-days+1, end_date] 的资金流向窗口并合并。"""
    dates = []
    d = end_date
    collected = 0
    while collected < days:
        dates.append(d)
        d = get_trade_date_before(d, days=1)
        collected += 1
    dates.reverse()
    dfs = concurrent_map(_load_moneyflow_by_date, dates)
    valid = [df for df in dfs if not df.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


STRATEGIES = {
    "momentum_value_hybrid": {
        "description": "趋势向上时，买入低估值高动量蓝筹",
        "pass1": {
            "filters": {"min_amount": 50000, "max_volatility": 0.07},
            "weights": {
                "mom_20": 0.30, "mom_60": 0.20,
                "ep": 0.15, "bp": 0.15,
                "size": 0.10, "liquidity": 0.10,
            },
        },
        "pass2": {
            "filters": {"min_roe": 0.05},
            "weights": {
                "mom_20": 0.25, "mom_60": 0.10,
                "ep": 0.10, "bp": 0.10, "sp": 0.05,
                "roe": 0.10, "profit_growth": 0.05, "revenue_growth": 0.05,
                "net_mf_5d": 0.10, "net_mf_ratio": 0.05,
                "liquidity": 0.05,
            },
        },
    },
    "quality_growth": {
        "description": "财报季或震荡市，买入高 ROE + 高成长标的",
        "pass1": {
            "filters": {"min_amount": 50000, "max_volatility": 0.08},
            "weights": {
                "mom_20": 0.20, "mom_60": 0.10,
                "ep": 0.10, "bp": 0.10,
                "size": 0.05, "liquidity": 0.15,
                "sp": 0.15, "dividend": 0.15,
            },
        },
        "pass2": {
            "filters": {"min_roe": 0.08, "min_profit_growth": 0.10},
            "weights": {
                "roe": 0.25, "profit_growth": 0.20, "revenue_growth": 0.10, "ocf_growth": 0.10,
                "ep": 0.10, "bp": 0.05,
                "net_mf_5d": 0.05, "net_mf_ratio": 0.05,
                "liquidity": 0.05, "mom_20": 0.05,
            },
        },
    },
    "contrarian": {
        "description": "大盘急跌后，买入超跌但基本面稳健标的",
        "pass1": {
            "filters": {"min_amount": 30000, "max_volatility": 0.10, "max_mom_60": 0.05},
            "weights": {
                "bp": 0.18, "ep": 0.13, "sp": 0.08,
                "dividend": 0.08, "liquidity": 0.08,
                "mom_20": -0.10, "mom_60": -0.05,
                "mom_5": 0.05, "reversal_score": 0.15,
                "sector_momentum": 0.05, "relative_to_sector": 0.05,
            },
        },
        "pass2": {
            "filters": {"min_roe": 0.03},
            "weights": {
                "bp": 0.18, "ep": 0.13, "roe": 0.08, "profit_growth": 0.08,
                "net_mf_5d": 0.10,
                "mom_20": -0.10, "mom_60": -0.05,
                "mom_5": 0.05, "reversal_score": 0.10,
                "sector_momentum": 0.05, "relative_to_sector": 0.05,
                "liquidity": 0.05,
            },
        },
    },
    "event_driven": {
        "description": "基于业绩预告/快报超预期与短期资金共振的事件驱动策略",
        "pass1": {
            "filters": {"min_amount": 50000, "max_volatility": 0.10},
            "weights": {
                "mom_20": 0.15, "mom_60": 0.05,
                "ep": 0.10, "bp": 0.10,
                "liquidity": 0.15, "size": 0.05,
                "net_mf_ratio": 0.10,
                "sector_momentum": 0.05,
            },
        },
        "pass2": {
            "filters": {},
            "weights": {
                "forecast_type_score": 0.28,
                "forecast_pchange_mid": 0.18,
                "express_diluted_roe": 0.12,
                "net_mf_5d": 0.10,
                "net_mf_ratio": 0.10,
                "mom_20": 0.05,
                "ep": 0.05,
                "liquidity": 0.05,
                "sector_momentum": 0.05,
            },
        },
    },
}


def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def rank_fill(s: pd.Series) -> pd.Series:
    """百分位排名并填充缺失值为 0.5。"""
    return s.rank(pct=True).fillna(0.5)


def compute_price_factors(df_daily: pd.DataFrame) -> dict:
    if df_daily is None or len(df_daily) < 60:
        return {}

    df = df_daily.sort_values("trade_date").reset_index(drop=True)
    close = df["close"].astype(float)
    amount = df["amount"].astype(float)
    returns = close.pct_change().dropna()

    mom_5 = (close.iloc[-1] / close.iloc[-5]) - 1
    mom_20 = (close.iloc[-1] / close.iloc[-20]) - 1
    mom_60 = (close.iloc[-1] / close.iloc[-60]) - 1
    volatility_20 = returns.tail(20).std()
    avg_amount_20 = amount.tail(20).mean()
    avg_amount_5 = amount.tail(5).mean()
    amount_ratio_5d = avg_amount_5 / avg_amount_20 if avg_amount_20 > 0 else 1.0

    # 反转打分：短期跌幅越深、近期相对 20 日放量，分数越高
    reversal_score = -(mom_20) * amount_ratio_5d

    return {
        "mom_5": mom_5,
        "mom_20": mom_20,
        "mom_60": mom_60,
        "volatility_20": volatility_20,
        "avg_amount_20": avg_amount_20,
        "amount_ratio_5d": amount_ratio_5d,
        "reversal_score": reversal_score,
    }


def fetch_fina_factors(ts_code: str, trade_date: str) -> dict:
    """获取最近已披露财务指标因子，严格校验 ann_date <= trade_date，避免财报穿越。失败返回空 dict。"""
    try:
        df = tushare_call("fina_indicator", {"ts_code": ts_code})
        if df.empty:
            return {}
        df["ann_date"] = df["ann_date"].fillna("").astype(str)
        df = df[df["ann_date"] <= trade_date]
        if df.empty:
            return {}
        row = df.sort_values("ann_date", ascending=False).iloc[0]
        return {
            "roe": safe_float(row.get("roe"), np.nan),
            "roa": safe_float(row.get("roa"), np.nan),
            "grossprofit_margin": safe_float(row.get("grossprofit_margin"), np.nan),
            "revenue_growth": safe_float(row.get("tr_yoy"), np.nan),
            "profit_growth": safe_float(row.get("netprofit_yoy"), np.nan),
            "ocf_growth": safe_float(row.get("ocf_yoy"), np.nan),
        }
    except Exception:
        return {}


def fetch_fund_factors(ts_code: str, start_date: str, end_date: str) -> dict:
    """获取资金流向因子，失败返回空 dict（保留兼容旧调用）。"""
    try:
        df = tushare_call("moneyflow", {"ts_code": ts_code, "start_date": start_date, "end_date": end_date})
        if df.empty or len(df) < 5:
            return {}
        df = df.sort_values("trade_date")
        df["net_mf_amount"] = pd.to_numeric(df.get("net_mf_amount", 0), errors="coerce").fillna(0)
        amount_cols = [c for c in df.columns if "_amount" in c and c != "net_mf_amount"]
        df["gross_amount"] = df[amount_cols].sum(axis=1)
        net_mf_5d = df["net_mf_amount"].tail(5).sum()
        net_mf_20d = df["net_mf_amount"].tail(20).sum()
        gross_5d = df["gross_amount"].tail(5).sum()
        net_mf_ratio = net_mf_5d / gross_5d if gross_5d > 0 else 0
        return {
            "net_mf_5d": net_mf_5d,
            "net_mf_20d": net_mf_20d,
            "net_mf_ratio": net_mf_ratio,
        }
    except Exception:
        return {}


def compute_fund_factors_from_window(ts_code: str, df_mf_window: pd.DataFrame) -> dict:
    """从预加载的资金流窗口中计算资金因子。"""
    try:
        df = df_mf_window[df_mf_window["ts_code"] == ts_code].sort_values("trade_date")
        if df.empty or len(df) < 5:
            return {}
        df["net_mf_amount"] = pd.to_numeric(df.get("net_mf_amount", 0), errors="coerce").fillna(0)
        amount_cols = [c for c in df.columns if "_amount" in c and c != "net_mf_amount"]
        df["gross_amount"] = df[amount_cols].sum(axis=1)
        net_mf_5d = df["net_mf_amount"].tail(5).sum()
        net_mf_20d = df["net_mf_amount"].tail(20).sum()
        gross_5d = df["gross_amount"].tail(5).sum()
        net_mf_ratio = net_mf_5d / gross_5d if gross_5d > 0 else 0
        return {
            "net_mf_5d": net_mf_5d,
            "net_mf_20d": net_mf_20d,
            "net_mf_ratio": net_mf_ratio,
        }
    except Exception:
        return {}


FORECAST_TYPE_SCORE = {
    "预增": 2.0,
    "略增": 1.0,
    "扭亏": 2.0,
    "续盈": 0.5,
    "预减": -1.0,
    "略减": -0.5,
    "首亏": -2.0,
    "续亏": -1.5,
    "不确定": 0.0,
    "减亏": 0.5,
    "增亏": -0.5,
}

# 业绩预告/快报 freshness：超过该天数的旧公告视为失效，不参与打分
EVENT_LOOKBACK_DAYS = 120


def _event_within_window(ann_date: str, trade_date: str, window_days: int = EVENT_LOOKBACK_DAYS) -> bool:
    """判断公告日期是否在 trade_date 前 window_days 天内。"""
    try:
        ann = datetime.strptime(ann_date, "%Y%m%d")
        base = datetime.strptime(trade_date, "%Y%m%d")
        return (base - ann).days <= window_days
    except Exception:
        return False


def fetch_forecast_factor(ts_code: str, trade_date: str) -> dict:
    """获取业绩预告因子，严格校验 ann_date <= trade_date 且处于有效窗口内。失败返回空 dict。"""
    try:
        df = tushare_call("forecast", {"ts_code": ts_code})
        if df.empty:
            return {}
        df["ann_date"] = df["ann_date"].fillna("").astype(str)
        df = df[df["ann_date"] <= trade_date]
        df = df[df["ann_date"].apply(lambda d: _event_within_window(d, trade_date, EVENT_LOOKBACK_DAYS))]
        if df.empty:
            return {}
        row = df.sort_values("ann_date", ascending=False).iloc[0]
        type_score = FORECAST_TYPE_SCORE.get(str(row.get("type")), 0.0)
        pmin = safe_float(row.get("p_change_min"), np.nan)
        pmax = safe_float(row.get("p_change_max"), np.nan)
        pchange_mid = (pmin + pmax) / 2.0 if pd.notna(pmin) and pd.notna(pmax) else np.nan
        return {
            "forecast_type_score": type_score,
            "forecast_pchange_mid": pchange_mid,
        }
    except Exception:
        return {}


def fetch_express_factor(ts_code: str, trade_date: str) -> dict:
    """获取业绩快报因子，严格校验 ann_date <= trade_date 且处于有效窗口内。失败返回空 dict。"""
    try:
        df = tushare_call("express", {"ts_code": ts_code})
        if df.empty:
            return {}
        df["ann_date"] = df["ann_date"].fillna("").astype(str)
        df = df[df["ann_date"] <= trade_date]
        df = df[df["ann_date"].apply(lambda d: _event_within_window(d, trade_date, EVENT_LOOKBACK_DAYS))]
        if df.empty:
            return {}
        row = df.sort_values("ann_date", ascending=False).iloc[0]
        diluted_roe = safe_float(row.get("diluted_roe"), np.nan)
        diluted_eps = safe_float(row.get("diluted_eps"), np.nan)
        return {
            "express_diluted_roe": diluted_roe,
            "express_diluted_eps": diluted_eps,
        }
    except Exception:
        return {}


def build_universe(date: str) -> pd.DataFrame:
    """构建股票池：剔除次新股，合并 daily_basic。"""
    df_basic = tushare_call("stock_basic", {"exchange": "", "list_status": "L"})
    df_basic = df_basic[df_basic["ts_code"].str.endswith((".SH", ".SZ"))]

    df_basic["list_dt"] = pd.to_datetime(df_basic["list_date"], format="%Y%m%d")
    cutoff = datetime.strptime(date, "%Y%m%d") - pd.Timedelta(days=LIST_DAYS_MIN)
    df_basic = df_basic[df_basic["list_dt"] <= cutoff]

    df_daily_basic = tushare_call("daily_basic", {"trade_date": date})
    df = df_basic.merge(df_daily_basic, on="ts_code", how="inner")

    df["total_mv"] = pd.to_numeric(df.get("total_mv", 0), errors="coerce").fillna(0)
    df["pe"] = pd.to_numeric(df.get("pe", np.nan), errors="coerce")
    df["pb"] = pd.to_numeric(df.get("pb", np.nan), errors="coerce")
    df["ps"] = pd.to_numeric(df.get("ps", np.nan), errors="coerce")
    df["dv_ratio"] = pd.to_numeric(df.get("dv_ratio", np.nan), errors="coerce")

    df = df.sort_values("total_mv", ascending=False).head(UNIVERSE_SAMPLE)
    return df


def _factor_series(df: pd.DataFrame, factor: str) -> pd.Series:
    """根据因子名返回标准化的打分序列。"""
    if factor == "ep":
        return rank_fill(1 / df["pe"].replace(0, np.nan))
    if factor == "bp":
        return rank_fill(1 / df["pb"].replace(0, np.nan))
    if factor == "sp":
        return rank_fill(1 / df["ps"].replace(0, np.nan))
    if factor == "dividend":
        return rank_fill(df["dv_ratio"])
    if factor == "size":
        return rank_fill(df["total_mv"])
    if factor == "liquidity":
        return rank_fill(df["avg_amount_20"])
    if factor in df.columns:
        return rank_fill(df[factor])
    return pd.Series([0.5] * len(df), index=df.index)


def compute_weighted_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    """按权重计算综合得分。"""
    score = pd.Series(0.0, index=df.index)
    for factor, w in weights.items():
        score += w * _factor_series(df, factor)
    return score


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """应用 pass 级别的过滤条件。"""
    if "min_amount" in filters:
        df = df[df["avg_amount_20"] >= filters["min_amount"]]
    if "max_volatility" in filters:
        df = df[df["volatility_20"] <= filters["max_volatility"]]
    if "min_mom_20" in filters:
        df = df[df["mom_20"] >= filters["min_mom_20"]]
    if "max_mom_20" in filters:
        df = df[df["mom_20"] <= filters["max_mom_20"]]
    if "min_mom_60" in filters:
        df = df[df["mom_60"] >= filters["min_mom_60"]]
    if "max_mom_60" in filters:
        df = df[df["mom_60"] <= filters["max_mom_60"]]
    if "min_roe" in filters:
        df = df[df["roe"] >= filters["min_roe"]]
    if "min_profit_growth" in filters:
        df = df[df["profit_growth"] >= filters["min_profit_growth"]]
    if "min_revenue_growth" in filters:
        df = df[df["revenue_growth"] >= filters["min_revenue_growth"]]
    return df


def add_sector_factors(df: pd.DataFrame, group_col: str = "industry") -> pd.DataFrame:
    """基于当日截面计算行业相对强度因子。

    注意：行业分类来自 `stock_basic`，为当前分类。历史回测中若股票行业发生过变更，
    报告中的行业分布可能与历史真实分布存在偏差，因此该因子仅作为辅助参考。
    """
    if df.empty or group_col not in df.columns:
        return df

    df = df.copy()
    df[group_col] = df[group_col].fillna("未知")

    # 行业动量：行业内成分股平均 mom_20
    sector_mom = df.groupby(group_col)["mom_20"].transform("mean")
    df["sector_momentum"] = sector_mom
    df["relative_to_sector"] = df["mom_20"] - sector_mom

    # 行业内 5 日动量与放量比均值
    if "mom_5" in df.columns:
        df["sector_mom5"] = df.groupby(group_col)["mom_5"].transform("mean")
    if "amount_ratio_5d" in df.columns:
        df["sector_amount_ratio"] = df.groupby(group_col)["amount_ratio_5d"].transform("mean")

    # 行业数量过滤：样本不足 3 只的行业标记为 NaN，避免单一股票误导行业均值
    sector_counts = df[group_col].map(df[group_col].value_counts())
    min_count = 3
    df.loc[sector_counts < min_count, "sector_momentum"] = np.nan
    df.loc[sector_counts < min_count, "relative_to_sector"] = np.nan
    if "sector_mom5" in df.columns:
        df.loc[sector_counts < min_count, "sector_mom5"] = np.nan
    if "sector_amount_ratio" in df.columns:
        df.loc[sector_counts < min_count, "sector_amount_ratio"] = np.nan

    return df


def _pass1_row(row: dict, date: str, df_daily_window: pd.DataFrame) -> dict:
    """处理单只股票的 pass1 因子。供并发调用。"""
    ts_code = row["ts_code"]
    try:
        df_price = df_daily_window[df_daily_window["ts_code"] == ts_code].sort_values("trade_date")
        if df_price.empty or date not in df_price["trade_date"].astype(str).values:
            return None
        factors = compute_price_factors(df_price)
        if not factors:
            return None
        return {
            "ts_code": ts_code,
            "name": row.get("name", ""),
            "industry": row.get("industry", ""),
            "mom_5": factors["mom_5"],
            "mom_20": factors["mom_20"],
            "mom_60": factors["mom_60"],
            "volatility_20": factors["volatility_20"],
            "avg_amount_20": factors["avg_amount_20"],
            "amount_ratio_5d": factors["amount_ratio_5d"],
            "reversal_score": factors["reversal_score"],
            "pe": safe_float(row.get("pe"), np.nan),
            "pb": safe_float(row.get("pb"), np.nan),
            "ps": safe_float(row.get("ps"), np.nan),
            "dv_ratio": safe_float(row.get("dv_ratio"), np.nan),
            "total_mv": safe_float(row.get("total_mv"), 0),
        }
    except Exception:
        return None


def pass1_screen(df_universe: pd.DataFrame, date: str, config: dict) -> pd.DataFrame:
    """第一轮：用价格 + 估值因子快速筛选，保留 PASS1_TOP_K。"""
    df_daily_window = load_daily_window(date, days=90)
    if df_daily_window.empty:
        return pd.DataFrame()

    rows = [row.to_dict() for _, row in df_universe.iterrows()]
    results = concurrent_map(lambda r: _pass1_row(r, date, df_daily_window), rows)
    records = [r for r in results if r is not None]

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    filters = config.get("filters", {})
    df = apply_filters(df, filters)
    if df.empty:
        return df

    # 历史 ST/*ST/退市 过滤
    if not SKIP_ST_CHECK:
        mask = ~df["ts_code"].apply(lambda x: is_st_historical(x, date))
        df = df[mask]

    # 行业相对强度（基于 pass1 截面）
    df = add_sector_factors(df)

    df["pass1_score"] = compute_weighted_score(df, config["weights"])
    return df.sort_values("pass1_score", ascending=False).head(PASS1_TOP_K)


def _pass2_row(row: dict, date: str, df_mf_window: pd.DataFrame) -> dict:
    """处理单个 pass1 候选的第二轮因子补充。供并发调用。"""
    ts_code = row["ts_code"]
    try:
        fina = fetch_fina_factors(ts_code, date)
        fund = compute_fund_factors_from_window(ts_code, df_mf_window)
        forecast = fetch_forecast_factor(ts_code, date)
        express = fetch_express_factor(ts_code, date)
        rec = dict(row)
        rec.update(fina)
        rec.update(fund)
        rec.update(forecast)
        rec.update(express)
        return rec
    except Exception:
        return None


def pass2_enrich(df_pass1: pd.DataFrame, date: str, config: dict) -> pd.DataFrame:
    """第二轮：对 top 候选补充财务 + 资金因子，重新打分。"""
    if df_pass1.empty:
        return df_pass1

    df_mf_window = load_moneyflow_window(date, days=25)

    rows = [row.to_dict() for _, row in df_pass1.iterrows()]
    results = concurrent_map(lambda r: _pass2_row(r, date, df_mf_window), rows)
    records = [r for r in results if r is not None]

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    for col in ["pe", "pb", "ps", "dv_ratio", "roe", "revenue_growth", "profit_growth", "ocf_growth",
                "net_mf_5d", "net_mf_20d", "net_mf_ratio", "avg_amount_20", "total_mv",
                "mom_5", "amount_ratio_5d", "reversal_score",
                "sector_momentum", "relative_to_sector", "sector_mom5", "sector_amount_ratio",
                "forecast_type_score", "forecast_pchange_mid", "express_diluted_roe", "express_diluted_eps"]:
        if col not in df.columns:
            df[col] = np.nan

    df = apply_filters(df, config.get("filters", {}))
    if df.empty:
        return df

    df["total_score"] = compute_weighted_score(df, config["weights"])
    return df.sort_values("total_score", ascending=False)


def cap_sector_weight(df: pd.DataFrame, top_n: int, max_pct: float = MAX_SECTOR_PCT) -> pd.DataFrame:
    """限制单一行业入选数量，避免过度集中。"""
    if df.empty or "industry" not in df.columns:
        return df

    df = df.copy()
    df["industry"] = df["industry"].fillna("未知")

    max_per_sector = max(1, int(top_n * max_pct))
    sector_counts = {}
    kept = []
    for _, row in df.iterrows():
        if len(kept) >= top_n:
            break
        sector = row["industry"]
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        kept.append(row)
    return pd.DataFrame(kept)


def load_dynamic_weights(strategy: str, regime: str = None) -> dict:
    """从 memory/weights/ 加载动态权重；不存在则返回 None。

    Args:
        strategy: 策略名
        regime: 可选，regime 名。若提供，优先加载 {strategy}_{regime}_rolling.json，
                不存在则回退到 {strategy}_latest.json。

    注意：回测模式下（AH_BACKTEST_MODE=1）不加载动态权重，避免 latest.json
    包含未来数据导致时间穿越。walk-forward 在线学习应通过显式 override 传入权重。
    """
    if os.environ.get("AH_BACKTEST_MODE", "").lower() in ("1", "true", "yes"):
        return None

    paths = []
    if regime:
        paths.append(Path("memory/weights") / f"{strategy}_{regime}_rolling.json")
    paths.append(Path("memory/weights") / f"{strategy}_latest.json")

    for path in paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("weights")
            except Exception:
                continue
    return None


def screen(date: str, strategy: str, top_n: int = 50, return_full: bool = False,
           pass1_weights_override: dict = None, pass2_weights_override: dict = None):
    """通用选股入口。支持 strategy='regime' 自动按市场状态切换策略；支持动态权重和权重覆盖。

    Args:
        return_full: 为 True 时返回 (top_n_records, full_df_pass2)，供离线优化使用。
        pass1_weights_override: 可选，直接覆盖 pass1 权重。
        pass2_weights_override: 可选，直接覆盖 pass2 权重。
    """
    actual_strategy = strategy
    regime_info = None
    if strategy == "regime":
        regime_info = classify_regime(date)
        actual_strategy = regime_to_strategy(regime_info["regime"], available=list(STRATEGIES.keys()))

    if actual_strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {actual_strategy}. Available: {list(STRATEGIES.keys())} or 'regime'")

    config = {k: (v.copy() if isinstance(v, dict) else v) for k, v in STRATEGIES[actual_strategy].items()}

    # 权重加载优先级：显式覆盖 > regime 滚动权重 > 策略最新权重 > 硬编码
    if pass1_weights_override is not None:
        config["pass1"]["weights"] = pass1_weights_override
    if pass2_weights_override is not None:
        config["pass2"]["weights"] = pass2_weights_override

    dynamic = None
    if pass1_weights_override is None or pass2_weights_override is None:
        regime_name = regime_info.get("regime") if regime_info else None
        dynamic = load_dynamic_weights(actual_strategy, regime=regime_name)
        if dynamic:
            for phase in ("pass1", "pass2"):
                if phase in dynamic and phase in config and "weights" in config[phase]:
                    # 只有未被显式覆盖的才使用动态权重
                    if (phase == "pass1" and pass1_weights_override is None) or \
                       (phase == "pass2" and pass2_weights_override is None):
                        config[phase]["weights"] = dynamic[phase]

    trace_event(
        "screen.start",
        {
            "inputs": {
                "date": date,
                "strategy": strategy,
                "actual_strategy": actual_strategy,
                "top_n": top_n,
                "regime_info": regime_info,
                "dynamic_weights_loaded": dynamic is not None,
                "pass1_weights": config["pass1"].get("weights"),
                "pass2_weights": config["pass2"].get("weights"),
                "pass1_filters": config["pass1"].get("filters"),
                "pass2_filters": config["pass2"].get("filters"),
            }
        },
        date=date,
        strategy=actual_strategy,
    )

    df_universe = build_universe(date)
    df_pass1 = pass1_screen(df_universe, date, config["pass1"])
    df_pass2 = pass2_enrich(df_pass1, date, config["pass2"])

    # 行业数量集中度截断（基于当前行业分类，仅作辅助参考）
    df_result = cap_sector_weight(df_pass2, top_n, max_pct=MAX_SECTOR_PCT)

    if df_result.empty:
        trace_event(
            "screen.end",
            {"outputs": {"candidates": 0}, "metadata": {"error": "empty result"}},
            date=date,
            strategy=actual_strategy,
        )
        return ([], df_pass2) if return_full else []

    output_cols = [
        "ts_code", "name", "industry", "total_score",
        "mom_5", "mom_20", "mom_60", "pe", "pb", "ps", "dv_ratio",
        "roe", "revenue_growth", "profit_growth", "ocf_growth",
        "net_mf_5d", "net_mf_20d", "net_mf_ratio",
        "avg_amount_20", "amount_ratio_5d", "volatility_20", "total_mv",
        "reversal_score", "sector_momentum", "relative_to_sector",
        "forecast_type_score", "forecast_pchange_mid",
        "express_diluted_roe", "express_diluted_eps",
    ]
    output_cols = [c for c in output_cols if c in df_result.columns]
    records = df_result[output_cols].to_dict(orient="records")

    trace_event(
        "screen.end",
        {
            "outputs": {
                "candidates": len(records),
                "pass2_pool_size": len(df_pass2),
                "top_picks": [{"ts_code": r["ts_code"], "name": r.get("name"), "score": r.get("total_score")} for r in records[:top_n]],
            },
            "metadata": {
                "universe_size": len(df_universe),
                "pass1_pool_size": len(df_pass1),
            },
        },
        date=date,
        strategy=actual_strategy,
    )

    if return_full:
        return records, df_pass2
    return records


def screen_momentum_value_hybrid(date: str, top_n: int = 50) -> list:
    return screen(date, "momentum_value_hybrid", top_n)


def main():
    strategy = sys.argv[1] if len(sys.argv) > 1 else "momentum_value_hybrid"
    date = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime("%Y%m%d")
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    candidates = screen(date, strategy, top_n)
    print(json.dumps(candidates, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
