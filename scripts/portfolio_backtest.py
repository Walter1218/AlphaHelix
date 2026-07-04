"""
AlphaHelix 组合回测（基于 GBDT 预测得分）

输入：model_trainer 生成的 predictions parquet
输出：组合净值、超额收益、方向准确率、换手率、交易成本

再平衡规则：
- 每个预测日，取预测得分最高的 max_positions 只股票；
- 等权持有到下一再平衡日；
- 换仓时扣除佣金、印花税、滑点；
- 可设置行业集中度上限。
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


def compute_period_return(holdings, start_date, end_date):
    """计算持仓等权收益。"""
    rets = []
    for code in holdings:
        try:
            p0 = get_close_price(code, start_date)
            p1 = get_close_price(code, end_date)
            rets.append(p1 / p0 - 1)
        except Exception:
            continue
    if not rets:
        return 0.0, []
    return float(np.mean(rets)), rets


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


def run_backtest(pred_path, max_positions=10, max_sector_pct=0.4,
                 commission=0.0002, stamp_tax=0.001, slippage=0.001,
                 pred_threshold=None):
    df = pd.read_parquet(pred_path)
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    nav = 1.0
    cash = 1.0
    positions = {}  # ts_code -> shares
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

        # 计算当前持仓市值
        portfolio_value = cash
        prices = {}
        for code in current_holdings:
            try:
                price = get_close_price(code, t_str)
                prices[code] = price
                portfolio_value += positions[code] * price
            except Exception:
                pass

        # 卖出
        sold = current_holdings - new_holdings
        for code in sold:
            if code not in prices:
                continue
            proceeds = positions[code] * prices[code]
            cost = proceeds * (commission + slippage + stamp_tax)
            cash += proceeds - cost
            del positions[code]

        # 买入/继续持有
        new_prices = {}
        for code in new_holdings:
            try:
                new_prices[code] = get_close_price(code, t_str)
            except Exception:
                pass

        # 等权分配目标市值
        n = len([c for c in new_holdings if c in new_prices])
        if n > 0:
            target_value = portfolio_value / n
            for code in new_holdings:
                if code not in new_prices:
                    continue
                # 若已持有，调整到目标市值
                old_value = positions.get(code, 0) * new_prices[code]
                delta = target_value - old_value
                if delta > 0:
                    cost = delta * (commission + slippage)
                    cash -= delta + cost
                    positions[code] = positions.get(code, 0) + (delta / new_prices[code])
                elif delta < 0:
                    sell_value = -delta
                    cost = sell_value * (commission + slippage + stamp_tax)
                    cash += sell_value - cost
                    positions[code] = positions.get(code, 0) + (delta / new_prices[code])
                    if abs(positions[code]) < 1e-9:
                        del positions[code]

        # 持有期收益
        holdings_list = list(positions.keys())
        port_ret, _ = compute_period_return(holdings_list, t_str, t_next_str)

        # 更新持仓市值和现金
        new_portfolio_value = cash
        for code, shares in positions.items():
            try:
                p1 = get_close_price(code, t_next_str)
                new_portfolio_value += shares * p1
            except Exception:
                pass

        try:
            bench_t = get_close_price("000300.SH", t_str)
            bench_t1 = get_close_price("000300.SH", t_next_str)
            bench_ret = bench_t1 / bench_t - 1
        except Exception:
            bench_ret = 0.0

        turnover = (len(sold) + len(new_holdings - current_holdings)) / (2 * max_positions) if max_positions > 0 else 0.0

        nav *= (1 + port_ret)
        excess = port_ret - bench_ret

        records.append({
            "date": t_str,
            "next_date": t_next_str,
            "holdings": len(holdings_list),
            "portfolio_return": port_ret,
            "benchmark_return": bench_ret,
            "excess_return": excess,
            "turnover": turnover,
            "nav": nav,
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
    args = parser.parse_args()

    summary = run_backtest(
        args.pred_path,
        max_positions=args.max_positions,
        max_sector_pct=args.max_sector_pct,
        commission=args.commission,
        stamp_tax=args.stamp_tax,
        slippage=args.slippage,
        pred_threshold=args.pred_threshold,
    )

    print("\n=== Portfolio Backtest Summary ===")
    print(f"Periods: {summary['periods']}")
    print(f"Avg portfolio return: {summary['avg_portfolio_return']:+.2%}")
    print(f"Avg excess return: {summary['avg_excess_return']:+.2%}")
    print(f"Cumulative portfolio return: {summary['cumulative_portfolio_return']:+.2%}")
    print(f"Cumulative excess return: {summary['cumulative_excess_return']:+.2%}")
    print(f"Win rate (excess > 0): {summary['win_rate_excess']:.1%}")
    print(f"Avg turnover: {summary['avg_turnover']:+.1%}")

    output_path = Path(args.pred_path).parent / f"{Path(args.pred_path).stem}_portfolio.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[portfolio_backtest] Saved summary to {output_path}")


if __name__ == "__main__":
    main()
