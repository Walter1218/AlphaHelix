"""
AlphaHelix 选股评估脚本
读取选股 JSON 快照，计算持有期实际收益、相对沪深300超额收益、最大回撤等指标。
"""
import sys
import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import tushare_call, get_trade_date_after

BENCHMARK = "000300.SH"


def load_snapshot(date: str) -> dict:
    path = Path("memory/stock") / f"{date}.json"
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_index(ts_code: str) -> bool:
    return ts_code in {BENCHMARK, "000001.SH", "399001.SZ", "399006.SZ"}


def get_close_price(ts_code: str, date: str) -> float:
    api = "index_daily" if _is_index(ts_code) else "daily"
    df = tushare_call(api, {"ts_code": ts_code, "trade_date": date})
    if df.empty:
        raise ValueError(f"No price data for {ts_code} on {date}")
    return float(df.iloc[0]["close"])


def get_price_series(ts_code: str, start_date: str, end_date: str) -> pd.Series:
    api = "index_daily" if _is_index(ts_code) else "daily"
    df = tushare_call(api, {"ts_code": ts_code, "start_date": start_date, "end_date": end_date})
    if df.empty:
        return pd.Series(dtype=float)
    df = df.sort_values("trade_date").set_index("trade_date")
    return pd.to_numeric(df["close"], errors="coerce")


def max_drawdown(prices: pd.Series) -> float:
    if prices.empty or len(prices) < 2:
        return 0.0
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return float(drawdown.min())


def evaluate(date: str, horizon: int = 20) -> dict:
    snapshot = load_snapshot(date)
    picks = snapshot.get("picks", [])
    if not picks:
        raise ValueError(f"No picks found in snapshot {date}")

    exit_date = get_trade_date_after(date, days=horizon)
    benchmark_entry = get_close_price(BENCHMARK, date)
    benchmark_exit = get_close_price(BENCHMARK, exit_date)
    benchmark_return = (benchmark_exit / benchmark_entry) - 1

    results = []
    for pick in picks:
        ts_code = pick["ts_code"]
        try:
            entry_price = get_close_price(ts_code, date)
            exit_price = get_close_price(ts_code, exit_date)
            price_series = get_price_series(ts_code, date, exit_date)

            abs_return = (exit_price / entry_price) - 1
            excess_return = abs_return - benchmark_return
            mdd = max_drawdown(price_series)

            results.append({
                "ts_code": ts_code,
                "name": pick.get("name", ""),
                "rank": pick.get("rank", 0),
                "confidence": pick.get("confidence", "medium"),
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "abs_return": round(abs_return, 6),
                "excess_return": round(excess_return, 6),
                "max_drawdown": round(mdd, 6),
            })
        except Exception as e:
            results.append({
                "ts_code": ts_code,
                "name": pick.get("name", ""),
                "rank": pick.get("rank", 0),
                "confidence": pick.get("confidence", "medium"),
                "error": str(e),
            })

    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"date": date, "horizon": horizon, "error": "No valid price data for any pick"}

    returns = np.array([r["abs_return"] for r in valid])
    direction_accuracy = float(np.mean(returns > 0))
    portfolio_return = float(np.mean(returns))
    excess_return = portfolio_return - benchmark_return

    top3 = [r for r in results if r.get("rank", 999) <= 3 and "error" not in r]
    top3_hit_rate = float(np.mean([r["abs_return"] > 0 for r in top3])) if top3 else 0.0

    # 组合最大回撤：等权组合净值序列（按交易日对齐）
    portfolio_df = None
    for r in valid:
        ts_code = r["ts_code"]
        try:
            series = get_price_series(ts_code, date, exit_date)
            if series.empty:
                continue
            normalized = series / series.iloc[0]
            df_piece = normalized.to_frame(ts_code)
            if portfolio_df is None:
                portfolio_df = df_piece
            else:
                portfolio_df = portfolio_df.join(df_piece, how="outer")
        except Exception:
            continue

    if portfolio_df is not None and not portfolio_df.empty:
        portfolio_values = portfolio_df.mean(axis=1, skipna=True).dropna()
        portfolio_mdd = max_drawdown(portfolio_values)
    else:
        portfolio_mdd = 0.0

    # 置信度相关性（将 high/medium/low 映射为 1/0.5/0）
    confidence_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
    confidences = [confidence_map.get(r.get("confidence", "medium"), 0.5) for r in valid]
    if len(valid) >= 2:
        confidence_correlation = float(np.corrcoef(confidences, [r["abs_return"] for r in valid])[0, 1])
        if np.isnan(confidence_correlation):
            confidence_correlation = 0.0
    else:
        confidence_correlation = 0.0

    return {
        "date": date,
        "exit_date": exit_date,
        "horizon": horizon,
        "benchmark": BENCHMARK,
        "benchmark_return": round(benchmark_return, 6),
        "portfolio_return": round(portfolio_return, 6),
        "excess_return": round(excess_return, 6),
        "direction_accuracy": round(direction_accuracy, 4),
        "top3_hit_rate": round(top3_hit_rate, 4),
        "portfolio_max_drawdown": round(portfolio_mdd, 6),
        "confidence_correlation": round(confidence_correlation, 4),
        "details": results,
    }


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y%m%d")
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    result = evaluate(date, horizon)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
