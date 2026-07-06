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
from _tushare_utils import tushare_call, get_trade_calendar, get_trade_date_before

# 全市场每日收盘价缓存，避免反复加载
_close_cache: dict = {}
# ts_code -> {date: close} 的反向索引，用于快速构造个股时间序列
_close_by_code: dict = {}

# 每日总市值缓存（来自 daily_basic.total_mv），用于行业权重约束和市值中性化
_mv_cache: dict = {}


def _update_close_index(d: str, ts_code: str, price: float):
    """同步更新全键缓存和按代码索引。"""
    _close_cache[(d, ts_code)] = float(price)
    if ts_code not in _close_by_code:
        _close_by_code[ts_code] = {}
    _close_by_code[ts_code][d] = float(price)


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
                    _update_close_index(d_str, row["ts_code"], float(row["close"]))
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
    series_dict = _close_by_code.get(ts_code, {})
    values = {d: p for d, p in series_dict.items() if start_date <= d <= end_date}
    if not values:
        return pd.Series(dtype=float)
    return pd.Series(values).sort_index()


def _warm_mv_cache(dates):
    """预热每日总市值缓存（daily_basic.total_mv）。"""
    for d in dates:
        d_str = d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d)
        if d_str in _mv_cache:
            continue
        try:
            df = tushare_call("daily_basic", {"trade_date": d_str})
            if not df.empty:
                df["ts_code"] = df["ts_code"].astype(str)
                df["total_mv"] = pd.to_numeric(df["total_mv"], errors="coerce")
                _mv_cache[d_str] = {
                    row["ts_code"]: float(row["total_mv"])
                    for _, row in df.iterrows()
                    if pd.notna(row["total_mv"])
                }
        except Exception:
            continue


def _get_cached_mv(ts_code: str, date: str) -> float:
    """获取个股某日总市值（亿元），缺失时兜底调用 daily_basic。"""
    d_str = str(date)
    code = str(ts_code)
    mv_map = _mv_cache.get(d_str)
    if mv_map and code in mv_map:
        return mv_map[code]
    try:
        df = tushare_call("daily_basic", {"ts_code": code, "trade_date": d_str})
        if not df.empty:
            mv = float(pd.to_numeric(df["total_mv"], errors="coerce").iloc[0])
            if pd.notna(mv):
                _mv_cache.setdefault(d_str, {})[code] = mv
                return mv
    except Exception:
        pass
    return np.nan


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


def _neutralize_scores_by_market_cap(df_day: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """
    用当日截面回归去除预测得分中的市值暴露。

    residual = predicted - (alpha + beta * log(total_mv))
    返回按 residual 排序后的 df，predicted 列被替换为 residual。
    """
    df = df_day.copy()
    if "total_mv" not in df.columns:
        df["total_mv"] = df["ts_code"].apply(lambda c: _get_cached_mv(c, date_str))
    df = df[df["total_mv"].notna() & (df["total_mv"] > 0)].copy()
    if len(df) < 5:
        return df

    log_mv = np.log(df["total_mv"].astype(float))
    y = df["predicted"].astype(float).values
    x = log_mv.values
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)
    cov = np.nanmean((x - x_mean) * (y - y_mean))
    var = np.nanvar(x)
    if var <= 1e-12:
        return df
    beta = cov / var
    alpha = y_mean - beta * x_mean
    df["predicted_raw"] = df["predicted"]
    df["predicted"] = y - (alpha + beta * x)
    return df.sort_values("predicted", ascending=False)


def apply_sector_weight_cap(df_day, max_positions, max_sector_weight=0.4, date_str=None):
    """
    按行业市值权重做集中度截断。

    在按预测得分排序后，贪心选择股票，保证每个行业的总市值权重不超过
    max_sector_weight。若某只股票加入后会突破行业权重上限，则跳过。
    """
    if df_day.empty or "industry" not in df_day.columns:
        return df_day
    df_day = df_day.copy()
    df_day["industry"] = df_day["industry"].fillna("未知")

    # 尝试补充总市值列
    if "total_mv" not in df_day.columns and date_str is not None:
        df_day["total_mv"] = df_day["ts_code"].apply(lambda c: _get_cached_mv(c, date_str))

    kept = []
    sector_mv = {}
    total_mv = 0.0
    fallback_used = False
    for _, row in df_day.iterrows():
        if len(kept) >= max_positions:
            break
        sec = row["industry"]
        mv = row.get("total_mv", np.nan)
        if not np.isfinite(mv) or mv <= 0:
            mv = 1.0
            fallback_used = True
        # 预测加入后检查行业权重：允许进入新行业，已有行业不得超过上限
        new_sector_mv = sector_mv.get(sec, 0.0) + mv
        new_total_mv = total_mv + mv
        if sector_mv.get(sec, 0.0) > 0 and new_sector_mv / new_total_mv > max_sector_weight + 1e-9:
            continue
        kept.append(row)
        sector_mv[sec] = new_sector_mv
        total_mv = new_total_mv

    if kept and fallback_used:
        # 缺失市值的股票按等权 fallback，对权重约束不精确，仅做提示
        pass
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


def _get_recent_volatility(code: str, end_date: str, window: int = 20) -> float:
    """基于缓存收盘价计算个股最近 window 个交易日的收益率波动率。"""
    try:
        series_dict = _close_by_code.get(code, {})
        # 只取小于等于 end_date 的交易日，避免 look-ahead
        valid_dates = sorted(d for d in series_dict.keys() if d <= end_date)
        if len(valid_dates) < window // 2:
            return np.nan
        prices = pd.Series({d: series_dict[d] for d in valid_dates})
        returns = prices.sort_index().pct_change().dropna()
        if len(returns) < 3:
            return np.nan
        # 用最近 window 个交易日的收益率
        return float(returns.tail(window).std())
    except Exception:
        return np.nan


def _compute_weights(day_df: pd.DataFrame, prices: dict, scheme: str = "equal") -> dict:
    """
    计算目标持仓权重。

    scheme:
    - equal: 等权
    - score: 按预测得分（非负）加权
    - risk_parity: 按波动率倒数加权
    - score_risk: 得分 / 波动率 加权
    """
    codes = [c for c in day_df["ts_code"].tolist() if c in prices]
    if not codes:
        return {}

    sub = day_df[day_df["ts_code"].isin(codes)].copy()
    sub = sub.sort_values("predicted", ascending=False).head(len(codes))
    scores = sub.set_index("ts_code")["predicted"]

    if scheme == "equal":
        raw = {c: 1.0 for c in codes}
    elif scheme == "score":
        raw = {c: max(0.0, float(scores.get(c, 0.0))) for c in codes}
    elif scheme in ("risk_parity", "score_risk"):
        end_date = sub["date"].iloc[0].strftime("%Y%m%d") if "date" in sub.columns else None
        vols = {}
        for c in codes:
            vol = _get_recent_volatility(c, end_date) if end_date else np.nan
            vols[c] = vol if vol and vol > 0 else np.nan
        median_vol = np.nanmedian(list(vols.values())) if any(np.isfinite(list(vols.values()))) else 1e-6
        raw = {}
        for c in codes:
            vol = vols.get(c, median_vol)
            if not np.isfinite(vol) or vol <= 0:
                vol = median_vol
            if scheme == "risk_parity":
                raw[c] = 1.0 / vol
            else:
                raw[c] = max(0.0, float(scores.get(c, 0.0))) / vol
    else:
        raise ValueError(f"Unknown weight scheme: {scheme}")

    total = sum(raw.values())
    if total <= 0:
        return {c: 1.0 / len(codes) for c in codes}
    return {c: v / total for c, v in raw.items()}


def run_backtest(pred_path, max_positions=10, max_sector_pct=0.4,
                 max_sector_weight: float = 1.0,
                 commission=0.0002, stamp_tax=0.001, slippage=0.001,
                 pred_threshold=None, stop_loss_pct=None,
                 weight_scheme: str = "equal",
                 neutralize_market_cap: bool = False):
    df = pd.read_parquet(pred_path)
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    # 预热收盘价缓存：覆盖回测区间所有交易日（若已缓存则跳过）
    if dates:
        start_cache = dates[0].strftime("%Y%m%d")
        end_cache = (dates[-1] + pd.Timedelta(days=30)).strftime("%Y%m%d")
        cal = get_trade_calendar("SSE", start_cache, end_cache)
        trade_dates = pd.to_datetime(cal[cal["is_open"].astype(int) == 1]["cal_date"])
        needed = {d.strftime("%Y%m%d") for d in trade_dates}
        cached = {d for d, _ in _close_cache.keys()}
        missing = needed - cached
        if missing:
            print(f"[portfolio_backtest] Warming close price cache for {len(missing)} missing trade dates...")
            _warm_close_cache([pd.to_datetime(d) for d in sorted(missing)])
            print(f"[portfolio_backtest] Cache ready: {len(_close_cache)} price points")
        else:
            print(f"[portfolio_backtest] Using existing cache: {len(_close_cache)} price points")

        # 若启用行业市值权重约束或市值中性化，同步预热总市值缓存
        if max_sector_weight < 1.0 or neutralize_market_cap:
            mv_needed = needed - set(_mv_cache.keys())
            if mv_needed:
                print(f"[portfolio_backtest] Warming market-cap cache for {len(mv_needed)} missing trade dates...")
                _warm_mv_cache([pd.to_datetime(d) for d in sorted(mv_needed)])
                print(f"[portfolio_backtest] MV cache ready: {sum(len(v) for v in _mv_cache.values())} entries")
            else:
                print(f"[portfolio_backtest] Using existing MV cache: {sum(len(v) for v in _mv_cache.values())} entries")

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
        if neutralize_market_cap:
            day_df = _neutralize_scores_by_market_cap(day_df, t_str)
        day_df = apply_sector_cap(day_df, max_positions, max_sector_pct)
        if max_sector_weight < 1.0:
            day_df = apply_sector_weight_cap(day_df, max_positions, max_sector_weight, date_str=t_str)
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

        valid_codes = [c for c in new_holdings if c in new_prices]
        n = len(valid_codes)
        weights = {}
        if n > 0:
            weights = _compute_weights(
                day_df[day_df["ts_code"].isin(valid_codes)],
                new_prices,
                scheme=weight_scheme,
            )
            for code in new_holdings:
                if code not in new_prices:
                    continue
                weight = weights.get(code, 1.0 / n)
                target_value = portfolio_value * weight
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
        "max_sector_weight": max_sector_weight,
        "neutralize_market_cap": neutralize_market_cap,
        "records": records,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", required=True)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--max-sector-pct", type=float, default=0.4)
    parser.add_argument("--max-sector-weight", type=float, default=1.0,
                        help="行业市值权重上限，例如 0.4 表示单行业不超过 40%；1.0 表示不启用")
    parser.add_argument("--neutralize-market-cap", action="store_true",
                        help="选股前对预测得分做市值中性化（截面回归去除 log(总市值) 暴露）")
    parser.add_argument("--commission", type=float, default=0.0002)
    parser.add_argument("--stamp-tax", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--pred-threshold", type=float, default=None)
    parser.add_argument("--stop-loss-pct", type=float, default=None,
                        help="个股止损比例，例如 0.05 表示跌 5% 止损")
    parser.add_argument("--weight-scheme", type=str, default="equal",
                        choices=["equal", "score", "risk_parity", "score_risk"],
                        help="持仓权重方案：equal 等权，score 按预测得分，risk_parity 按波动率倒数，score_risk 结合得分和风险")
    args = parser.parse_args()

    summary = run_backtest(
        args.pred_path,
        max_positions=args.max_positions,
        max_sector_pct=args.max_sector_pct,
        max_sector_weight=args.max_sector_weight,
        commission=args.commission,
        stamp_tax=args.stamp_tax,
        slippage=args.slippage,
        pred_threshold=args.pred_threshold,
        stop_loss_pct=args.stop_loss_pct,
        weight_scheme=args.weight_scheme,
        neutralize_market_cap=args.neutralize_market_cap,
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
