"""
AlphaHelix 熊市/急跌防御模块

根据市场 regime 与近期沪深300 最大回撤，计算当期股票仓位比例。
仓位比例 = 1.0 表示满仓；0.0 表示空仓（全现金）。

设计原则：
- 不预测下跌，只在已有公开信息（regime、已实现回撤）触发防御；
- 规则简单、可解释、参数化，便于回测调优；
- 现金流收益 = 0，组合收益按仓位比例缩放。
"""
import sys
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import tushare_call, get_trade_date_before
from market_regime import classify_regime


def max_drawdown(prices: pd.Series) -> float:
    """计算价格序列的最大回撤（负值）。"""
    if prices.empty or len(prices) < 2:
        return 0.0
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return float(drawdown.min())


def get_index_drawdown(trade_date: str, days: int = 60, retries: int = 3) -> float:
    """获取 trade_date 前 days 个交易日内沪深300 的最大回撤（带重试）。"""
    start_date = get_trade_date_before(trade_date, days=days)
    for attempt in range(retries):
        try:
            df = tushare_call("index_daily", {"ts_code": "000300.SH", "start_date": start_date, "end_date": trade_date})
            if df.empty or len(df) < 2:
                return 0.0
            df = df.sort_values("trade_date")
            df["close"] = pd.to_numeric(df["close"], errors="coerce").dropna()
            return max_drawdown(df["close"])
        except Exception as e:
            if attempt == retries - 1:
                print(f"[market_defense] get_index_drawdown failed after {retries} attempts for {trade_date} window={days}: {e}", file=sys.stderr)
                return 0.0
            time.sleep(0.5 * (attempt + 1))


def get_defensive_position(trade_date: str,
                           regime_info: dict = None,
                           base_ratio: dict = None,
                           drawdown_windows: dict = None) -> float:
    """
    计算防御仓位比例。

    Args:
        trade_date: 选股日 YYYYMMDD
        regime_info: market_regime.classify_regime 的输出；为 None 时自动计算
        base_ratio: 各 regime 基础仓位，如 {"range": 1.0, "trend_up": 1.0, "trend_down": 0.3, "high_vol": 0.5}
        drawdown_windows: 回撤触发减仓的窗口与阈值，如 {20: -0.05, 60: -0.10}

    Returns:
        仓位比例，范围 [0.0, 1.0]
    """
    if base_ratio is None:
        base_ratio = {"range": 1.0, "trend_up": 1.0, "trend_down": 0.3, "high_vol": 0.5}
    if drawdown_windows is None:
        drawdown_windows = {20: -0.05, 60: -0.10}

    if regime_info is None:
        regime_info = classify_regime(trade_date)
    regime = regime_info.get("regime", "range") if regime_info else "range"

    ratio = base_ratio.get(regime, 1.0)
    ratio = max(0.0, min(1.0, ratio))

    for window, threshold in sorted(drawdown_windows.items()):
        dd = get_index_drawdown(trade_date, days=window)
        if dd <= threshold:
            ratio *= 0.5

    return max(0.0, min(1.0, ratio))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute defensive position ratio")
    parser.add_argument("--date", required=True, help="Trade date YYYYMMDD")
    args = parser.parse_args()
    pos = get_defensive_position(args.date)
    print(f"{args.date}: position_ratio={pos:.2f}")
