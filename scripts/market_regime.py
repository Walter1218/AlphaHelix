"""
AlphaHelix 市场状态判断模块
基于沪深300 日线判断当前市场 regime，用于策略切换和仓位控制。
"""
import sys
import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import tushare_call, get_trade_date_before

BENCHMARK = "000300.SH"
MIN_DAYS = 60


def fetch_index_data(trade_date: str, days: int = 90) -> pd.DataFrame:
    """获取沪深300 截至 trade_date 的日线数据。"""
    start_date = get_trade_date_before(trade_date, days=days)
    df = tushare_call("index_daily", {"ts_code": BENCHMARK, "start_date": start_date, "end_date": trade_date})
    if df.empty:
        return df
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    # Tushare pct_chg 字段为百分比数值（如 -2.14），需除以 100 转为小数
    df["pct_chg"] = pd.to_numeric(df.get("pct_chg", 0), errors="coerce") / 100.0
    df = df.dropna(subset=["close"])
    return df


def classify_regime(trade_date: str) -> dict:
    """
    判断市场 regime。

    规则（基于沪深300 近 60 个交易日）：
    1. 高波动：20 日年化波动率 > 30% → high_vol
    2. 下跌趋势：20 日收益 < -8% 或近 20 日最大单日跌幅 < -5% → trend_down
    3. 趋势向上：20 日均线斜率 > 0，20 日收益 > 5%，20 日波动率 < 20% → trend_up
    4. 其他 → range
    """
    df = fetch_index_data(trade_date, days=90)
    if len(df) < MIN_DAYS:
        return {"trade_date": trade_date, "regime": "range", "reason": "insufficient_data", "metrics": {}}

    close = df["close"]
    recent = df.tail(20)

    ma20 = recent["close"].mean()
    ma20_prev = df.iloc[-25:-5]["close"].mean() if len(df) >= 25 else recent["close"].iloc[0]
    ma20_slope = (ma20 - ma20_prev) / ma20_prev if ma20_prev > 0 else 0.0

    ret_20d = (recent["close"].iloc[-1] / recent["close"].iloc[0]) - 1
    volatility_20d = recent["pct_chg"].std()
    annual_vol = volatility_20d * np.sqrt(252)
    max_daily_drop = recent["pct_chg"].min()

    reason = ""
    if annual_vol > 0.30:
        regime = "high_vol"
        reason = f"annual_vol={annual_vol:.1%} > 30%"
    elif ret_20d < -0.08 or max_daily_drop < -0.05:
        regime = "trend_down"
        reason = f"ret_20d={ret_20d:.1%}, max_drop={max_daily_drop:.1%}"
    elif ma20_slope > 0 and ret_20d > 0.05 and annual_vol < 0.20:
        regime = "trend_up"
        reason = f"ma20_slope={ma20_slope:.2%}, ret_20d={ret_20d:.1%}, vol={annual_vol:.1%}"
    else:
        regime = "range"
        reason = f"ma20_slope={ma20_slope:.2%}, ret_20d={ret_20d:.1%}, vol={annual_vol:.1%}"

    return {
        "trade_date": trade_date,
        "regime": regime,
        "reason": reason,
        "metrics": {
            "ma20": round(ma20, 4),
            "ma20_slope": round(float(ma20_slope), 6),
            "ret_20d": round(float(ret_20d), 6),
            "volatility_20d": round(float(volatility_20d), 6),
            "annual_vol": round(float(annual_vol), 6),
            "max_daily_drop": round(float(max_daily_drop), 6),
        },
    }


def regime_to_strategy(regime: str, available: list = None) -> str:
    """根据 regime 推荐策略。

    逻辑（2026-07-03 更新，基于 walk-forward 验证）：
    - 趋势向上/震荡：使用 event_driven，因为事件驱动（forecast/express）在两个主要回测区间均跑赢 momentum_value_hybrid。
    - 趋势下跌：使用 contrarian，捕捉超跌反弹。
    - 高波动：使用 quality_growth，高 ROE/高成长标的防御性相对更强。
    """
    mapping = {
        "trend_up": "event_driven",
        "range": "event_driven",
        "trend_down": "contrarian",
        "high_vol": "quality_growth",
    }
    strategy = mapping.get(regime, "event_driven")
    if available and strategy not in available:
        return available[0]
    return strategy


def main():
    parser = argparse.ArgumentParser(description="AlphaHelix market regime classifier")
    parser.add_argument("date", nargs="?", default=datetime.now().strftime("%Y%m%d"), help="Trade date YYYYMMDD")
    args = parser.parse_args()
    result = classify_regime(args.date)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
