"""
AlphaHelix 选股评估脚本
读取选股 JSON 快照，计算持有期实际收益、相对沪深300超额收益、最大回撤等指标。
"""
import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import tushare_call, get_trade_date_after, get_trade_calendar
from _trace import trace_event

BENCHMARK = "000300.SH"

# 按 trade_date 缓存截面数据，避免 evaluate 时对每只股票单独请求 start/end
_DAILY_CROSS_SECTION: dict = {}
_INDEX_CROSS_SECTION: dict = {}


def load_daily_cross_section(trade_date: str) -> pd.DataFrame:
    """加载某交易日的全市场日线截面（含缓存）。"""
    if trade_date not in _DAILY_CROSS_SECTION:
        df = tushare_call("daily", {"trade_date": trade_date})
        if not df.empty:
            df["trade_date"] = df["trade_date"].astype(str)
            df["ts_code"] = df["ts_code"].astype(str)
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
        _DAILY_CROSS_SECTION[trade_date] = df
    return _DAILY_CROSS_SECTION[trade_date]


def load_index_cross_section(trade_date: str) -> pd.DataFrame:
    """加载某交易日的指数日线截面（含缓存）。"""
    if trade_date not in _INDEX_CROSS_SECTION:
        df = tushare_call("index_daily", {"trade_date": trade_date})
        if not df.empty:
            df["trade_date"] = df["trade_date"].astype(str)
            df["ts_code"] = df["ts_code"].astype(str)
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
        _INDEX_CROSS_SECTION[trade_date] = df
    return _INDEX_CROSS_SECTION[trade_date]


def _close_from_cross_section(ts_code: str, trade_date: str) -> float:
    """从截面缓存中查某只股票/指数的收盘价。"""
    if _is_index(ts_code):
        df = load_index_cross_section(trade_date)
    else:
        df = load_daily_cross_section(trade_date)
    if df.empty:
        raise ValueError(f"No cross-section data on {trade_date}")
    row = df[df["ts_code"] == ts_code]
    if row.empty:
        raise ValueError(f"No price data for {ts_code} on {trade_date}")
    return float(row.iloc[0]["close"])


def load_snapshot(date: str) -> dict:
    path = Path("memory/stock") / f"{date}.json"
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_index(ts_code: str) -> bool:
    return ts_code in {BENCHMARK, "000001.SH", "399001.SZ", "399006.SZ"}


def get_close_price(ts_code: str, date: str) -> float:
    return _close_from_cross_section(ts_code, date)


def get_price_series(ts_code: str, start_date: str, end_date: str) -> pd.Series:
    """通过截面缓存拼装某股票在 [start_date, end_date] 的价格序列。"""
    cal = get_trade_calendar("SSE", start_date, end_date)
    cal = cal[cal["is_open"].astype(int) == 1].sort_values("cal_date")
    dates = cal["cal_date"].astype(str).tolist()
    if not dates:
        return pd.Series(dtype=float)

    values = {}
    for d in dates:
        try:
            values[d] = _close_from_cross_section(ts_code, d)
        except Exception:
            continue
    if not values:
        return pd.Series(dtype=float)
    return pd.Series(values).sort_index()


def max_drawdown(prices: pd.Series) -> float:
    if prices.empty or len(prices) < 2:
        return 0.0
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return float(drawdown.min())


def evaluate(date: str, horizon: int = 20, score_field: str = None) -> dict:
    snapshot = load_snapshot(date)
    picks = snapshot.get("picks", [])
    if not picks:
        raise ValueError(f"No picks found in snapshot {date}")

    exit_date = get_trade_date_after(date, days=horizon)
    benchmark_entry = get_close_price(BENCHMARK, date)
    benchmark_exit = get_close_price(BENCHMARK, exit_date)
    benchmark_return = (benchmark_exit / benchmark_entry) - 1

    # 按 score_field 计算权重；缺失或无效时退化为等权
    weights = None
    if score_field:
        raw_scores = []
        for pick in picks:
            try:
                raw_scores.append(float(pick.get(score_field, 0)))
            except (TypeError, ValueError):
                raw_scores.append(0.0)
        raw_scores = np.array(raw_scores)
        min_score = raw_scores.min()
        shifted = raw_scores - min_score + 1e-9
        weights = shifted / shifted.sum()

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

    # 权重：默认等权，若提供 score_field 则按得分加权
    if weights is not None and len(weights) == len(valid):
        w = weights[:len(valid)]
        w = w / w.sum()
    else:
        w = np.ones(len(valid)) / len(valid)
    portfolio_return = float(np.sum(w * returns))
    excess_return = portfolio_return - benchmark_return

    top3 = [r for r in results if r.get("rank", 999) <= 3 and "error" not in r]
    top3_hit_rate = float(np.mean([r["abs_return"] > 0 for r in top3])) if top3 else 0.0

    # 组合最大回撤：加权组合净值序列（按交易日对齐）
    portfolio_df = None
    for i, r in enumerate(valid):
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
        # 按权重加权各股票净值
        weight_map = {r["ts_code"]: w[i] for i, r in enumerate(valid) if r["ts_code"] in portfolio_df.columns}
        aligned_weights = np.array([weight_map.get(col, 0) for col in portfolio_df.columns])
        aligned_weights = aligned_weights / aligned_weights.sum()
        portfolio_values = portfolio_df.multiply(aligned_weights, axis=1).sum(axis=1, skipna=True).dropna()
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

    result = {
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

    trace_event(
        "evaluate",
        {
            "inputs": {"date": date, "horizon": horizon, "benchmark": BENCHMARK},
            "outputs": {
                "benchmark_return": result["benchmark_return"],
                "portfolio_return": result["portfolio_return"],
                "excess_return": result["excess_return"],
                "direction_accuracy": result["direction_accuracy"],
                "top3_hit_rate": result["top3_hit_rate"],
                "portfolio_max_drawdown": result["portfolio_max_drawdown"],
                "confidence_correlation": result["confidence_correlation"],
                "details_count": len(results),
                "valid_count": len(valid),
            },
        },
        date=date,
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="AlphaHelix 选股评估")
    parser.add_argument("date", nargs="?", default=datetime.now().strftime("%Y%m%d"),
                        help="选股快照日期 YYYYMMDD")
    parser.add_argument("horizon", nargs="?", type=int, default=20,
                        help="持有期交易日数")
    parser.add_argument("--score-field", default=None,
                        help="使用指定字段作为权重计算组合收益（例如 gbdt_score）")
    args = parser.parse_args()

    result = evaluate(args.date, args.horizon, score_field=args.score_field)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
