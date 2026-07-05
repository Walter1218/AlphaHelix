"""
Phase 3：增加另类数据 engineered 特征

新增：
- 宏观融资融券：margin_total_balance（当日总和）, margin_change_5d, margin_change_20d
- 宏观北向：northbound_net_today, northbound_net_5d_sum, northbound_net_20d_sum,
            northbound_net_20d_zscore, northbound_net_5d_vs_20d
- 龙虎榜：5日/10日出现次数、累计净流入、平均 amount_rate / turnover_rate / pct_change
- 披露日：days_to_disclosure / days_since_disclosure 的 rank 版本（已在 Phase 2 v2 里做，这里保留 raw）

宏观特征不经过截面 neutralize/rank，保留为日期维度原始值，让树模型按市场状态做 regime split。
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import (
    fetch_margin_daily, fetch_northbound_daily, fetch_top_list,
    get_trade_date_before,
)


def _date_to_str(d):
    return pd.to_datetime(d).strftime("%Y%m%d")


def build_margin_features(dates: list) -> pd.DataFrame:
    """为每个交易日构建融资融券余额及变化率（宏观）。"""
    rows = []
    for d in dates:
        d_str = _date_to_str(d)
        try:
            df = fetch_margin_daily(d_str)
            if df.empty:
                continue
            total = pd.to_numeric(df.get("rzrqye"), errors="coerce").sum()
            rows.append({"date": d_str, "margin_total_balance": total})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["date", "margin_total_balance", "margin_change_5d", "margin_change_20d"])

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["margin_total_balance"] = out["margin_total_balance"].astype(float)
    out["margin_change_5d"] = out["margin_total_balance"] / out["margin_total_balance"].shift(5) - 1
    out["margin_change_20d"] = out["margin_total_balance"] / out["margin_total_balance"].shift(20) - 1
    return out


def build_northbound_features(dates: list) -> pd.DataFrame:
    """为每个交易日构建北向资金特征（宏观）。"""
    # 先拉取宽窗口，避免逐日重复调用
    if not dates:
        return pd.DataFrame()
    dates = sorted([_date_to_str(d) for d in dates])
    start = get_trade_date_before(dates[0], days=30)
    end = dates[-1]

    parts = []
    d = end
    # 往回拉最多 60 个交易日，覆盖 start
    collected = 0
    target_dates = set(dates)
    current = end
    all_dates = []
    while collected < 80:
        all_dates.append(current)
        if current <= start:
            break
        try:
            current = get_trade_date_before(current, days=1)
        except Exception:
            break
        collected += 1

    for d in all_dates:
        try:
            df = fetch_northbound_daily(d)
            if df.empty:
                continue
            net = pd.to_numeric(df.get("north_money"), errors="coerce").sum()
            parts.append({"date": d, "northbound_net": net})
        except Exception:
            continue

    if not parts:
        return pd.DataFrame(columns=["date", "northbound_net_today", "northbound_net_5d_sum",
                                     "northbound_net_20d_sum", "northbound_net_20d_zscore",
                                     "northbound_net_5d_vs_20d"])

    nb = pd.DataFrame(parts).sort_values("date").reset_index(drop=True)
    nb["northbound_net"] = nb["northbound_net"].astype(float)

    def window_sum(col, n):
        return nb[col].rolling(window=n, min_periods=1).sum()

    def window_mean_std(col, n):
        return nb[col].rolling(window=n, min_periods=2).mean(), nb[col].rolling(window=n, min_periods=2).std()

    nb["northbound_net_today"] = nb["northbound_net"]
    nb["northbound_net_5d_sum"] = window_sum("northbound_net", 5)
    nb["northbound_net_20d_sum"] = window_sum("northbound_net", 20)
    mean20, std20 = window_mean_std("northbound_net", 20)
    nb["northbound_net_20d_zscore"] = (nb["northbound_net"] - mean20) / (std20.replace(0, np.nan))
    nb["northbound_net_5d_vs_20d"] = nb["northbound_net_5d_sum"] / nb["northbound_net_20d_sum"].replace(0, np.nan)

    nb = nb[nb["date"].isin(target_dates)].copy()
    return nb[["date", "northbound_net_today", "northbound_net_5d_sum", "northbound_net_20d_sum",
               "northbound_net_20d_zscore", "northbound_net_5d_vs_20d"]]


def build_top_list_window_features(date: str, window: int) -> pd.DataFrame:
    """基于过去 window 个交易日龙虎榜数据聚合个股特征。"""
    parts = []
    for offset in range(1, window + 1):
        try:
            d = get_trade_date_before(date, days=offset)
        except Exception:
            continue
        try:
            df = fetch_top_list(d)
            if df.empty:
                continue
            df["ts_code"] = df["ts_code"].astype(str)
            for col in ["net_amount", "amount_rate", "turnover_rate", "pct_change"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.drop_duplicates(subset=["ts_code"], keep="first")
            df["trade_date"] = d
            parts.append(df[["trade_date", "ts_code", "net_amount", "amount_rate", "turnover_rate", "pct_change"]])
        except Exception:
            continue

    if not parts:
        cols = ["ts_code",
                f"top_list_count_{window}d", f"top_list_net_amount_{window}d",
                f"top_list_amount_rate_{window}d", f"top_list_turnover_rate_{window}d", f"top_list_pct_change_{window}d"]
        return pd.DataFrame(columns=cols)

    all_df = pd.concat(parts, ignore_index=True)
    agg = all_df.groupby("ts_code").agg(
        top_list_count=("ts_code", "size"),
        top_list_net_amount=("net_amount", "sum"),
        top_list_amount_rate=("amount_rate", "mean"),
        top_list_turnover_rate=("turnover_rate", "mean"),
        top_list_pct_change=("pct_change", "mean"),
    ).reset_index()
    agg = agg.rename(columns={
        "top_list_count": f"top_list_count_{window}d",
        "top_list_net_amount": f"top_list_net_amount_{window}d",
        "top_list_amount_rate": f"top_list_amount_rate_{window}d",
        "top_list_turnover_rate": f"top_list_turnover_rate_{window}d",
        "top_list_pct_change": f"top_list_pct_change_{window}d",
    })
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="memory/dataset/features_h10_composite_phase2.parquet")
    parser.add_argument("--output", default="memory/dataset/features_h10_composite_phase3.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    dates = sorted(df["date"].unique())
    print(f"[append_phase3] {len(df)} rows, {len(dates)} dates")

    # 宏观特征
    print("[append_phase3] building margin features...")
    margin_df = build_margin_features(dates)
    print("[append_phase3] building northbound features...")
    nb_df = build_northbound_features(dates)

    # 龙虎榜 5d / 10d
    print("[append_phase3] building top_list 5d features...")
    top5_parts = []
    for i, d in enumerate(dates):
        if i % 20 == 0:
            print(f"[append_phase3] top_list 5d {i}/{len(dates)} {d}")
        tdf = build_top_list_window_features(d, 5)
        if not tdf.empty:
            tdf["date"] = d
            top5_parts.append(tdf)
    top5_df = pd.concat(top5_parts, ignore_index=True) if top5_parts else pd.DataFrame()

    print("[append_phase3] building top_list 10d features...")
    top10_parts = []
    for i, d in enumerate(dates):
        if i % 20 == 0:
            print(f"[append_phase3] top_list 10d {i}/{len(dates)} {d}")
        tdf = build_top_list_window_features(d, 10)
        if not tdf.empty:
            tdf["date"] = d
            top10_parts.append(tdf)
    top10_df = pd.concat(top10_parts, ignore_index=True) if top10_parts else pd.DataFrame()

    merged = df.copy()
    for extra in [margin_df, nb_df]:
        if not extra.empty:
            merged = merged.merge(extra, on="date", how="left")
    for extra in [top5_df, top10_df]:
        if not extra.empty:
            merged = merged.merge(extra, on=["date", "ts_code"], how="left")

    # 填充个股缺失为 0（未上榜）
    for col in merged.columns:
        if col.startswith("top_list_count_") or col.startswith("top_list_net_amount_") or \
           col.startswith("top_list_amount_rate_") or col.startswith("top_list_turnover_rate_") or \
           col.startswith("top_list_pct_change_"):
            merged[col] = merged[col].fillna(0)

    print(f"[append_phase3] final shape {merged.shape}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    print(f"[append_phase3] saved to {args.output}")


if __name__ == "__main__":
    main()
