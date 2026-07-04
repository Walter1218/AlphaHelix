"""
Phase 2 v2：精炼另类数据特征，避免宏观变量污染截面模型。

新增特征：
- 披露日：days_to_disclosure, days_since_disclosure, disclosure_near(<=5), disclosure_very_near(<=2), since_disclosure_lt10
- 龙虎榜：过去 20 个交易日出现次数、累计净流入、平均 amount_rate / turnover_rate / pct_change

不对融资融券/北向资金做个股截面特征（它们是市场层面变量），避免 neutralize 后产生伪截面波动。
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 加载 .env 中的 TUSHARE_TOKEN
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import (
    fetch_top_list, fetch_disclosure_schedule, get_trade_date_before,
)
from feature_engineering import build_numeric_features


def build_top_list_window_features(date: str, window: int = 20) -> pd.DataFrame:
    """基于过去 window 个交易日龙虎榜数据，为每只股票聚合特征。"""
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
            df = df.rename(columns={
                "net_amount": "net_amount",
                "amount_rate": "amount_rate",
                "turnover_rate": "turnover_rate",
                "pct_change": "pct_change",
            })
            df = df.drop_duplicates(subset=["ts_code"], keep="first")
            df["trade_date"] = d
            parts.append(df[["trade_date", "ts_code", "net_amount", "amount_rate", "turnover_rate", "pct_change"]])
        except Exception:
            continue

    if not parts:
        return pd.DataFrame(columns=["ts_code", "top_list_count_20d", "top_list_net_amount_20d",
                                     "top_list_amount_rate_20d", "top_list_turnover_rate_20d", "top_list_pct_change_20d"])

    all_df = pd.concat(parts, ignore_index=True)
    agg = all_df.groupby("ts_code").agg(
        top_list_count_20d=("ts_code", "size"),
        top_list_net_amount_20d=("net_amount", "sum"),
        top_list_amount_rate_20d=("amount_rate", "mean"),
        top_list_turnover_rate_20d=("turnover_rate", "mean"),
        top_list_pct_change_20d=("pct_change", "mean"),
    ).reset_index()
    return agg


def build_disclosure_features(dates: list, ts_codes: list) -> pd.DataFrame:
    """基于预约披露时间表，用 merge_asof 快速为所有 date+ts_code 计算披露日特征。"""
    years = sorted({pd.to_datetime(d).year for d in dates} | {pd.to_datetime(d).year - 1 for d in dates})
    disc = fetch_disclosure_schedule(years=years)
    if disc.empty:
        return pd.DataFrame(columns=["date", "ts_code", "days_to_disclosure", "days_since_disclosure"])

    disc["ts_code"] = disc["ts_code"].astype(str)
    disc["pre_date"] = pd.to_datetime(disc["pre_date"], errors="coerce")
    disc["ann_date"] = pd.to_datetime(disc["ann_date"], errors="coerce")
    disc = disc[disc["ts_code"].isin(ts_codes)].copy()

    target = pd.DataFrame({"date": pd.to_datetime(dates)})
    target["key"] = 1
    stocks = pd.DataFrame({"ts_code": ts_codes})
    stocks["key"] = 1
    grid = target.merge(stocks, on="key").drop(columns=["key"])

    pre = disc.dropna(subset=["pre_date"]).sort_values("pre_date")
    grid_pre = pd.merge_asof(
        grid.sort_values("date"),
        pre[["ts_code", "pre_date"]].sort_values("pre_date"),
        left_on="date", right_on="pre_date", by="ts_code", direction="forward",
    )
    grid_pre["days_to_disclosure"] = (grid_pre["pre_date"] - grid_pre["date"]).dt.days

    ann = disc.dropna(subset=["ann_date"]).sort_values("ann_date")
    grid_ann = pd.merge_asof(
        grid.sort_values("date"),
        ann[["ts_code", "ann_date"]].sort_values("ann_date"),
        left_on="date", right_on="ann_date", by="ts_code", direction="backward",
    )
    grid_ann["days_since_disclosure"] = (grid_ann["date"] - grid_ann["ann_date"]).dt.days

    out = grid_pre[["date", "ts_code", "days_to_disclosure"]].merge(
        grid_ann[["date", "ts_code", "days_since_disclosure"]], on=["date", "ts_code"], how="outer")
    out["date"] = out["date"].dt.strftime("%Y%m%d")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="memory/dataset/features_h10.parquet")
    parser.add_argument("--output", default="memory/dataset/features_h10_phase2_v2.parquet")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    dates = sorted(df["date"].unique())
    ts_codes = df["ts_code"].astype(str).unique().tolist()
    print(f"[append_phase2_v2] {len(df)} rows, {len(dates)} dates, {len(ts_codes)} stocks")

    # 龙虎榜窗口特征
    top_parts = []
    for i, d in enumerate(dates):
        if i % 10 == 0:
            print(f"[append_phase2_v2] top_list window {i}/{len(dates)} {d}")
        tdf = build_top_list_window_features(d, window=args.window)
        if not tdf.empty:
            tdf["date"] = d
            top_parts.append(tdf)
    top_df = pd.concat(top_parts, ignore_index=True) if top_parts else pd.DataFrame()

    # 披露日特征
    print("[append_phase2_v2] building disclosure features...")
    disc_df = build_disclosure_features(dates, ts_codes)

    # 合并
    merged = df.copy()
    if not top_df.empty:
        merged = merged.merge(top_df, on=["date", "ts_code"], how="left")
    if not disc_df.empty:
        merged = merged.merge(disc_df, on=["date", "ts_code"], how="left")

    # 构建事件 dummy
    merged["disclosure_near"] = (merged["days_to_disclosure"].fillna(999) <= 5).astype(int)
    merged["disclosure_very_near"] = (merged["days_to_disclosure"].fillna(999) <= 2).astype(int)
    merged["since_disclosure_lt10"] = (merged["days_since_disclosure"].fillna(999) <= 10).astype(int)

    for col in ["top_list_count_20d", "top_list_net_amount_20d", "top_list_amount_rate_20d",
                "top_list_turnover_rate_20d", "top_list_pct_change_20d"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # 对新增特征做截尾/中性化/rank
    new_cols = [
        "days_to_disclosure", "days_since_disclosure",
        "disclosure_near", "disclosure_very_near", "since_disclosure_lt10",
        "top_list_count_20d", "top_list_net_amount_20d", "top_list_amount_rate_20d",
        "top_list_turnover_rate_20d", "top_list_pct_change_20d",
    ]
    merged = build_numeric_features(merged, feature_cols=[c for c in new_cols if c in merged.columns],
                                    neutralize=True, rank=True, winsorize=True)

    print(f"[append_phase2_v2] final shape {merged.shape}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    print(f"[append_phase2_v2] saved to {args.output}")


if __name__ == "__main__":
    main()
