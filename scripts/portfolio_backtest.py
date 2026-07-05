"""
AlphaHelix 组合回测（基于 GBDT 预测得分）

输入：model_trainer 生成的 predictions parquet
输出：组合净值、超额收益、方向准确率、换手率、交易成本、止损触发次数

再平衡规则：
- 每个预测日，取预测得分最高的 max_positions 只股票；
- 等权持有到下一再平衡日；
- 换仓时扣除佣金、印花税、滑点；
- 可设置行业集中度上限；
- 可设置个股止损线，触发止损当日收盘离场。
"""
import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import get_close_price
from _tushare_utils import tushare_call, get_trade_calendar

# 全市场每日收盘价缓存，避免反复加载
_close_cache: dict = {}


def _warm_close_cache(dates):
    """预热所有交易日的收盘价缓存。"""
    for d in dates:
        d_str = d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d)
        if d_str in _close_cache:
            continue
        try:
            df = tushare_call("daily", {"trade_date": d_str})
            if not df.empty:
                df["ts_code"] = df["ts_code"].astype(str)
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                for _, row in df.iterrows():
                    _close_cache[(d_str, row["ts_code"])] = float(row["close"])
        except Exception:
            continue


def _get_cached_close(ts_code: str, date: str) -> float:
    key = (str(date), str(ts_code))
    if key not in _close_cache:
        # 兜底：调用 evaluate.get_close_price
        from evaluate import get_close_price
        return get_close_price(ts_code, date)
    return _close_cache[key]


def _get_cached_price_series(ts_code: str, start_date: str, end_date: str) -> pd.Series:
    """基于已缓存的截面数据构造个股价格序列。"""
    values = {}
    for (d, code), price in _close_cache.items():
        if code == ts_code and start_date <= d <= end_date:
            values[d] = price
    if not values:
        return pd.Series(dtype=float)
    return pd.Series(values).sort_index()


def apply_sector_cap(df_day, max_positions, max_sector_pct=0.4):
    """按行业数量做集中度截断。"""
    if df_day.empty or "industry" not in df_day.columns:
        return df_day
    max_per_sector = max(1, int(max_positions * max_sector_pct))
    df_day = df_day.copy()
    df_day["industry"] = df_day["industry"].fillna("未知")
    counts = {}
    kept = []
    for _, row in df_day.iterrows():
        if len(kept) >= max_positions:
            break
        sec = row["industry"]
        if counts.get(sec, 0) >= max_per_sector:
            continue
        counts[sec] = counts.get(sec, 0) + 1
        kept.append(row)
    return pd.DataFrame(kept)


def simulate_position(code, shares, entry_price, stop_price, start_date, end_date,
                      commission=0.0002, stamp_tax=0.001, slippage=0.001):
    """模拟单只股票在持有期内的表现，触发止损则提前离场。

    返回：(final_cash, hit_stop, days_held, exit_date)
    """
    try:
        series = _get_cached_price_series(code, start_date, end_date)
    except Exception:
        return 0.0, False, 0, end_date

    if series.empty:
        return 0.0, False, 0, end_date

    # 跳过起始日（entry_price 已在该日买入）
    future = series.loc[series.index > start_date]
    if future.empty:
        # 没有后续交易日，按 end_date 收盘价卖出
        exit_price = series.iloc[-1]
        proceeds = shares * exit_price * (1 - slippage)
        cost = proceeds * (commission + stamp_tax)
        return proceeds - cost, False, 1, end_date

    for d, price in future.items():
        if price <= stop_price:
            # 触发止损，当日收盘离场
            effective_price = price * (1 - slippage)
            proceeds = shares * effective_price
            cost = proceeds * (commission + stamp_tax)
            return proceeds - cost, True, len(series.loc[series.index <= d]), d

    # 持有到期
    exit_price = series.iloc[-1]
    effective_price = exit_price * (1 - slippage)
    proceeds = shares * effective_price
    cost = proceeds * (commission + stamp_tax)
    return proceeds - cost, False, len(future), future.index[-1]


def run_backtest(pred_path, max_positions=10, max_sector_pct=0.4,
                 commission=0.0002, stamp_tax=0.001, slippage=0.001,
                 pred_threshold=None, stop_loss_pct=None):
    df = pd.read_parquet(pred_path)
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    # 预热收盘价缓存：覆盖回测区间所有交易日
    if dates:
        start_cache = dates[0].strftime("%Y%m%d")
        end_cache = (dates[-1] + pd.Timedelta(days=30)).strftime("%Y%m%d")
        cal = get_trade_calendar("SSE", start_cache, end_cache)
        trade_dates = pd.to_datetime(cal[cal["is_open"].astype(int) == 1]["cal_date"])
        print(f"[portfolio_backtest] Warming close price cache for {len(trade_dates)} trade dates...")
        _warm_close_cache(trade_dates)
        print(f"[portfolio_backtest] Cache ready: {len(_close_cache)} price points")

    nav = 1.0
    cash = 1.0
    positions = {}  # ts_code -> shares
    stops = {}      # ts_code -> stop_price
    current_holdings = set()
    records = []

    for i, t in enumerate(dates[:-1]):
        t_next = dates[i + 1]
        t_str = t.strftime("%Y%m%d")
        t_next_str = t_next.strftime("%Y%m%d")

        day_df = df[df["date"] == t].sort_values("predicted", ascending=False)
        if pred_threshold is not None:
            day_df = day_df[day_df["predicted"] >= pred_threshold]
        day_df = apply_sector_cap(day_df, max_positions, max_sector_pct)
        new_holdings = set(day_df["ts_code"].head(max_positions).tolist())

        # 计算当前持仓市值（按 t 日收盘价）
        portfolio_value = cash
        prices = {}
        for code in current_holdings:
            try:
                price = _get_cached_close(code, t_str)
                prices[code] = price
                portfolio_value += positions[code] * price
            except Exception:
                pass

        # 卖出不在新持仓中的旧股
        sold = current_holdings - new_holdings
        for code in sold:
            if code not in prices:
                continue
            proceeds = positions[code] * prices[code] * (1 - slippage)
            cost = proceeds * (commission + stamp_tax)
            cash += proceeds - cost
            del positions[code]
            stops.pop(code, None)

        # 买入新股/调整持仓
        new_prices = {}
        for code in new_holdings:
            try:
                new_prices[code] = _get_cached_close(code, t_str)
            except Exception:
                pass

        n = len([c for c in new_holdings if c in new_prices])
        if n > 0:
            target_value = portfolio_value / n
            for code in new_holdings:
                if code not in new_prices:
                    continue
                old_value = positions.get(code, 0) * new_prices[code]
                delta = target_value - old_value
                if delta > 0:
                    cost = delta * (commission + slippage)
                    cash -= delta + cost
                    add_shares = delta * (1 - slippage) / new_prices[code]
                    positions[code] = positions.get(code, 0) + add_shares
                    if stop_loss_pct is not None:
                        stops[code] = new_prices[code] * (1 - stop_loss_pct)
                elif delta < 0:
                    sell_value = -delta
                    cost = sell_value * (commission + slippage + stamp_tax)
                    cash += sell_value - cost
                    reduce_shares = sell_value / new_prices[code]
                    positions[code] = positions.get(code, 0) - reduce_shares
                    if abs(positions[code]) < 1e-9:
                        del positions[code]
                        stops.pop(code, None)
                    elif stop_loss_pct is not None and code in stops:
                        # 重新按最新成本价设止损（简化处理）
                        stops[code] = new_prices[code] * (1 - stop_loss_pct)

        # 期初市值（用于计算本期收益率）
        period_start_value = cash + sum(
            positions[code] * new_prices.get(code, prices.get(code, 0))
            for code in list(positions.keys())
        )
        if period_start_value <= 0:
            period_start_value = 1e-9

        # 持有期内逐只模拟，支持止损
        final_invested = 0.0
        hit_count = 0
        for code, shares in list(positions.items()):
            stop_price = stops.get(code, 0.0)
            proceeds, hit, _, _ = simulate_position(
                code, shares,
                new_prices.get(code, prices.get(code, 0)),
                stop_price,
                t_str, t_next_str,
                commission=commission, stamp_tax=stamp_tax, slippage=slippage,
            )
            final_invested += proceeds
            if hit:
                hit_count += 1

        # 期末总市值 = 未 invested 现金 + 个股离场现金
        period_end_value = cash + final_invested

        try:
            bench_t = _get_cached_close("000300.SH", t_str)
            bench_t1 = _get_cached_close("000300.SH", t_next_str)
            bench_ret = bench_t1 / bench_t - 1
        except Exception:
            bench_ret = 0.0

        port_ret = period_end_value / period_start_value - 1
        turnover = (len(sold) + len(new_holdings - current_holdings)) / (2 * max_positions) if max_positions > 0 else 0.0

        nav *= (1 + port_ret)
        excess = port_ret - bench_ret

        records.append({
            "date": t_str,
            "next_date": t_next_str,
            "holdings": len(positions),
            "portfolio_return": port_ret,
            "benchmark_return": bench_ret,
            "excess_return": excess,
            "turnover": turnover,
            "nav": nav,
            "stop_hits": hit_count,
        })

        current_holdings = set(positions.keys())

    summary = {
        "periods": len(records),
        "avg_portfolio_return": float(np.mean([r["portfolio_return"] for r in records])),
        "avg_excess_return": float(np.mean([r["excess_return"] for r in records])),
        "cumulative_portfolio_return": float(nav - 1),
        "cumulative_excess_return": float(nav - 1) - float(np.prod([1 + r["benchmark_return"] for r in records]) - 1),
        "avg_turnover": float(np.mean([r["turnover"] for r in records])),
        "win_rate_excess": float(np.mean([r["excess_return"] > 0 for r in records])),
        "avg_stop_hits": float(np.mean([r["stop_hits"] for r in records])),
        "records": records,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", required=True)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--max-sector-pct", type=float, default=0.4)
    parser.add_argument("--commission", type=float, default=0.0002)
    parser.add_argument("--stamp-tax", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--pred-threshold", type=float, default=None)
    parser.add_argument("--stop-loss-pct", type=float, default=None,
                        help="个股止损比例，例如 0.05 表示跌 5% 止损")
    args = parser.parse_args()

    summary = run_backtest(
        args.pred_path,
        max_positions=args.max_positions,
        max_sector_pct=args.max_sector_pct,
        commission=args.commission,
        stamp_tax=args.stamp_tax,
        slippage=args.slippage,
        pred_threshold=args.pred_threshold,
        stop_loss_pct=args.stop_loss_pct,
    )

    print("\n=== Portfolio Backtest Summary ===")
    print(f"Periods: {summary['periods']}")
    print(f"Avg portfolio return: {summary['avg_portfolio_return']:+.2%}")
    print(f"Avg excess return: {summary['avg_excess_return']:+.2%}")
    print(f"Cumulative portfolio return: {summary['cumulative_portfolio_return']:+.2%}")
    print(f"Cumulative excess return: {summary['cumulative_excess_return']:+.2%}")
    print(f"Win rate (excess > 0): {summary['win_rate_excess']:.1%}")
    print(f"Avg turnover: {summary['avg_turnover']:+.1%}")
    if summary.get("avg_stop_hits") is not None:
        print(f"Avg stop-loss hits / period: {summary['avg_stop_hits']:.2f}")

    output_path = Path(args.pred_path).parent / f"{Path(args.pred_path).stem}_portfolio.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[portfolio_backtest] Saved summary to {output_path}")


if __name__ == "__main__":
    main()
