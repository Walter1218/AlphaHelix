"""
全市场基础特征数据集构建

不经过 screen.py 的 PASS1/PASS2 筛选，直接从 daily / daily_basic 构造：
- 所有有交易的 A 股（剔除 ST/退市、流动性过差）；
- 基础量价/估值特征；
- 未来 H 日收益与超额收益。

输出：memory/dataset/features_h{H}_full.parquet
供模型在全市场分布上训练，避免只在 top-80 候选池里学习。
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import tushare_call, get_trade_calendar, get_trade_date_after, is_st_historical
from feature_engineering import build_numeric_features
from evaluate import get_close_price

DATASET_DIR = Path("memory/dataset")

# 交易日收盘价缓存
_close_cache: dict = {}


def _warm_close_cache(dates: list):
    """预热所有交易日的收盘价缓存。"""
    for d in dates:
        d_str = d if isinstance(d, str) else d.strftime("%Y%m%d")
        if (d_str, "any") in _close_cache or (d_str, "000001.SZ") in _close_cache:
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


def _get_close(ts_code: str, date: str) -> float:
    key = (date, ts_code)
    if key in _close_cache:
        return _close_cache[key]
    try:
        price = get_close_price(ts_code, date)
        _close_cache[key] = float(price)
        return float(price)
    except Exception:
        return np.nan


def _build_price_series(ts_code: str, dates: list) -> pd.Series:
    """用缓存构造个股价格序列。"""
    values = {}
    for d in dates:
        d_str = d if isinstance(d, str) else d.strftime("%Y%m%d")
        p = _get_close(ts_code, d_str)
        if not np.isnan(p):
            values[d_str] = p
    return pd.Series(values).sort_index()


def _get_rebalance_dates(start_date: str, end_date: str, delta_days: int) -> list:
    cal = get_trade_calendar("SSE", start_date, end_date)
    cal = cal[cal["is_open"].astype(int) == 1].copy()
    dates = sorted(cal["cal_date"].astype(str).tolist())
    rebalance = []
    for i in range(0, len(dates), delta_days):
        t = dates[i]
        try:
            t_next = get_trade_date_after(t, days=delta_days)
        except Exception:
            break
        if t_next > end_date:
            break
        rebalance.append(t)
    return rebalance


def build_full_dataset(start_date: str, end_date: str, horizon: int,
                       delta_days: int = 5,
                       min_turnover_rate: float = 0.001):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATASET_DIR / f"features_h{horizon}_full.parquet"

    rebalance = _get_rebalance_dates(start_date, end_date, delta_days)
    print(f"[build_full] {len(rebalance)} rebalance dates")

    # 预热所需收盘价：lookback 120 天 + 未来 horizon 天
    all_trade_dates = get_trade_calendar("SSE",
                                          (pd.to_datetime(start_date) - pd.Timedelta(days=180)).strftime("%Y%m%d"),
                                          (pd.to_datetime(end_date) + pd.Timedelta(days=30)).strftime("%Y%m%d"))
    all_trade_dates = sorted(all_trade_dates[all_trade_dates["is_open"].astype(int) == 1]["cal_date"].astype(str).tolist())
    print(f"[build_full] Warming close cache for {len(all_trade_dates)} trade dates...")
    _warm_close_cache(all_trade_dates)
    print(f"[build_full] Cache ready: {len(_close_cache)} price points")

    # 加载股票基础信息（行业）
    stock_basic = tushare_call("stock_basic", {"exchange": "", "list_status": "L"})
    stock_basic["ts_code"] = stock_basic["ts_code"].astype(str)
    industry_map = stock_basic.set_index("ts_code")["industry"].to_dict() if "industry" in stock_basic.columns else {}

    rows = []
    for idx, t in enumerate(rebalance, 1):
        print(f"[build_full] [{idx}/{len(rebalance)}] {t} ...", end=" ", flush=True)
        try:
            t_exit = get_trade_date_after(t, days=horizon)
            bench_t = _get_close("000300.SH", t)
            bench_exit = _get_close("000300.SH", t_exit)
            if np.isnan(bench_t) or np.isnan(bench_exit):
                print("no bench")
                continue
            bench_ret = bench_exit / bench_t - 1

            daily = tushare_call("daily", {"trade_date": t})
            basic = tushare_call("daily_basic", {"trade_date": t})
            if daily.empty or basic.empty:
                print("empty")
                continue

            daily["ts_code"] = daily["ts_code"].astype(str)
            basic["ts_code"] = basic["ts_code"].astype(str)
            if "close" in basic.columns:
                basic = basic.drop(columns=["close"])
            df = daily.merge(basic, on="ts_code", how="inner")

            # 基础过滤：有交易、非 ST/退市、流动性足够
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["turnover_rate"] = pd.to_numeric(df["turnover_rate"], errors="coerce")
            df = df[df["close"].notna() & (df["close"] > 0)]
            df = df[df["turnover_rate"].notna() & (df["turnover_rate"] >= min_turnover_rate)]
            if not SKIP_ST_CHECK:
                df = df[~df["ts_code"].apply(lambda x: is_st_historical(x, t))]

            if df.empty:
                print("0 after filter")
                continue

            # 计算动量/波动率（用缓存价格序列）
            date_idx = {d: i for i, d in enumerate(all_trade_dates)}
            t_pos = date_idx[t]

            mom5_list, mom20_list, mom60_list, mom120_list, vol_list = [], [], [], [], []
            amt5_list, amt20_list, rev_list = [], [], []
            for code in df["ts_code"]:
                series = _build_price_series(code, all_trade_dates[:t_pos + 1])
                if len(series) < 25:
                    mom5_list.append(np.nan); mom20_list.append(np.nan); mom60_list.append(np.nan)
                    mom120_list.append(np.nan); vol_list.append(np.nan)
                    amt5_list.append(np.nan); amt20_list.append(np.nan); rev_list.append(np.nan)
                    continue
                closes = series.values
                mom5_list.append(closes[-1] / closes[-6] - 1 if len(closes) >= 6 else np.nan)
                mom20_list.append(closes[-1] / closes[-21] - 1 if len(closes) >= 21 else np.nan)
                mom60_list.append(closes[-1] / closes[-61] - 1 if len(closes) >= 61 else np.nan)
                mom120_list.append(closes[-1] / closes[-121] - 1 if len(closes) >= 121 else np.nan)
                rets = pd.Series(closes[-21:]).pct_change().dropna()
                vol_list.append(rets.std() if len(rets) > 1 else np.nan)
                # 成交额需要 daily 的 amount；这里用 close * vol 近似
                # 用过去 5/20 日收盘价均值作为成交额代理
                amt5_list.append(np.mean(closes[-5:]))
                amt20_list.append(np.mean(closes[-20:]))
                high20 = np.max(closes[-20:])
                low20 = np.min(closes[-20:])
                rev_list.append((closes[-1] - high20) / (high20 - low20 + 1e-9) if high20 > low20 else 0.0)

            df["mom_5"] = mom5_list
            df["mom_20"] = mom20_list
            df["mom_60"] = mom60_list
            df["mom_120"] = mom120_list
            df["volatility_20"] = vol_list
            df["amount_proxy_5d"] = amt5_list
            df["amount_proxy_20d"] = amt20_list
            df["amount_ratio_5d"] = df["amount_proxy_5d"] / (df["amount_proxy_20d"] + 1e-9)
            df["reversal_score"] = rev_list

            # 估值特征
            df["pe"] = pd.to_numeric(df.get("pe"), errors="coerce")
            df["pb"] = pd.to_numeric(df.get("pb"), errors="coerce")
            df["ps"] = pd.to_numeric(df.get("ps"), errors="coerce")
            df["dv_ratio"] = pd.to_numeric(df.get("dv_ratio"), errors="coerce")
            df["total_mv"] = pd.to_numeric(df.get("total_mv"), errors="coerce")
            df["ep"] = 1.0 / df["pe"]
            df["bp"] = 1.0 / df["pb"]
            df["sp"] = 1.0 / df["ps"]
            df["liquidity"] = df["turnover_rate"]

            # 行业与相对强度
            df["industry"] = df["ts_code"].map(industry_map).fillna("未知")
            bench_mom20 = _get_close("000300.SH", t) / _get_close("000300.SH",
                                                                  all_trade_dates[max(0, t_pos - 20)]) - 1
            df["relative_strength"] = df["mom_20"] / (bench_mom20 + 1e-9)

            # 行业动量等
            df["sector_momentum"] = df.groupby("industry")["mom_20"].transform("mean")
            df["relative_to_sector"] = df["mom_20"] - df["sector_momentum"]
            df["sector_breadth"] = df.groupby("industry")["mom_20"].transform(
                lambda x: (x > 0).mean())

            # 风险调整动量
            df["risk_adj_mom"] = df["mom_20"] / (df["volatility_20"] + 1e-9)

            # 过滤缺失关键特征
            needed = ["mom_20", "volatility_20", "ep", "bp", "total_mv"]
            df = df.dropna(subset=needed)
            if df.empty:
                print("0 after feature dropna")
                continue

            # 构建模型特征（截尾、中性化、rank）
            feature_cols = ["mom_5", "mom_20", "mom_60", "mom_120",
                            "risk_adj_mom", "relative_strength",
                            "ep", "bp", "sp", "dv_ratio",
                            "sector_momentum", "relative_to_sector", "sector_breadth",
                            "reversal_score", "amount_ratio_5d", "volatility_20", "liquidity", "total_mv"]
            df = build_numeric_features(df, feature_cols=feature_cols, neutralize=True, rank=True, winsorize=True)

            n_valid = 0
            for _, row in df.iterrows():
                code = row["ts_code"]
                p_t = _get_close(code, t)
                p_exit = _get_close(code, t_exit)
                if np.isnan(p_t) or np.isnan(p_exit):
                    continue
                stock_ret = p_exit / p_t - 1
                excess_ret = stock_ret - bench_ret
                rec = {
                    "date": t,
                    "exit_date": t_exit,
                    "ts_code": code,
                    "stock_return": stock_ret,
                    "benchmark_return": bench_ret,
                    "excess_return": excess_ret,
                    "industry": row.get("industry"),
                }
                for col in feature_cols:
                    rec[col] = row[col]
                rows.append(rec)
                n_valid += 1

            print(f"{n_valid} samples")
        except Exception as e:
            print(f"ERROR: {e}")

        if idx % 10 == 0 and rows:
            df_save = pd.DataFrame(rows)
            df_save.to_parquet(output_path, index=False)
            print(f"[build_full] Saved {len(df_save)} rows to {output_path}")

    if rows:
        df_save = pd.DataFrame(rows)
        df_save.to_parquet(output_path, index=False)
        print(f"[build_full] Saved {len(df_save)} rows to {output_path}")
    else:
        print("[build_full] No rows generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--delta-days", type=int, default=5)
    parser.add_argument("--min-turnover-rate", type=float, default=0.001)
    parser.add_argument("--skip-st-check", action="store_true", default=True)
    args = parser.parse_args()

    global SKIP_ST_CHECK
    SKIP_ST_CHECK = args.skip_st_check

    build_full_dataset(args.start, args.end, args.horizon,
                       delta_days=args.delta_days,
                       min_turnover_rate=args.min_turnover_rate)
