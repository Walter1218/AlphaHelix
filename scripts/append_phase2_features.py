"""
把 Phase 2 另类数据特征增量合并到已有 dataset。

避免重新跑 build_dataset（耗时），直接按日期拉取市场/个股特征，
并基于预约披露时间表计算个股披露日特征，然后按 date+ts_code 合并。
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
    fetch_margin_daily, fetch_northbound_daily, fetch_top_list,
    fetch_disclosure_schedule, get_trade_date_before,
)


def build_market_features(date: str) -> dict:
    """返回某日期对应的市场层面特征（margin / northbound 用 T-1）。"""
    lag = get_trade_date_before(date, days=1)
    feats = {}

    try:
        mdf = fetch_margin_daily(lag)
        if not mdf.empty:
            mdf["rzrqye"] = pd.to_numeric(mdf.get("rzrqye"), errors="coerce")
            mdf["rzye"] = pd.to_numeric(mdf.get("rzye"), errors="coerce")
            mdf["rqye"] = pd.to_numeric(mdf.get("rqye"), errors="coerce")
            feats["margin_total_balance"] = mdf["rzrqye"].sum()
            feats["margin_financing_ratio"] = mdf["rzye"].sum() / (mdf["rqye"].sum() + 1e-9)
    except Exception:
        pass

    try:
        ndf = fetch_northbound_daily(lag)
        if not ndf.empty:
            feats["northbound_net"] = pd.to_numeric(ndf.get("north_money"), errors="coerce").sum()
            nb_5d = 0.0
            cnt = 0
            for d in pd.date_range(end=pd.to_datetime(lag), periods=5, freq="B"):
                try:
                    tmp = fetch_northbound_daily(d.strftime("%Y%m%d"))
                    if not tmp.empty:
                        nb_5d += pd.to_numeric(tmp.get("north_money"), errors="coerce").sum()
                        cnt += 1
                except Exception:
                    continue
            feats["northbound_net_5d"] = nb_5d if cnt > 0 else np.nan
    except Exception:
        pass

    return feats


def build_top_list_features(date: str) -> pd.DataFrame:
    """返回某日期（T-1）龙虎榜个股特征。"""
    lag = get_trade_date_before(date, days=1)
    try:
        df = fetch_top_list(lag)
        if df.empty:
            return pd.DataFrame()
        df["ts_code"] = df["ts_code"].astype(str)
        for col in ["net_amount", "amount_rate", "turnover_rate", "pct_change"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.rename(columns={
            "net_amount": "top_list_net_amount",
            "amount_rate": "top_list_amount_rate",
            "turnover_rate": "top_list_turnover_rate",
            "pct_change": "top_list_pct_change",
        })
        df["top_list_flag"] = 1
        keep = ["ts_code", "top_list_flag", "top_list_net_amount",
                "top_list_amount_rate", "top_list_turnover_rate", "top_list_pct_change"]
        return df[[c for c in keep if c in df.columns]]
    except Exception:
        return pd.DataFrame()


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

    # 目标笛卡尔积
    target = pd.DataFrame({"date": pd.to_datetime(dates)})
    target["key"] = 1
    stocks = pd.DataFrame({"ts_code": ts_codes})
    stocks["key"] = 1
    grid = target.merge(stocks, on="key").drop(columns=["key"])

    # 下一次预约披露（pre_date >= date，取最早）
    pre = disc.dropna(subset=["pre_date"]).sort_values("pre_date")
    grid_pre = pd.merge_asof(
        grid.sort_values("date"),
        pre[["ts_code", "pre_date"]].sort_values("pre_date"),
        left_on="date",
        right_on="pre_date",
        by="ts_code",
        direction="forward",
    )
    grid_pre["days_to_disclosure"] = (grid_pre["pre_date"] - grid_pre["date"]).dt.days

    # 上一次实际披露（ann_date <= date，取最晚）
    ann = disc.dropna(subset=["ann_date"]).sort_values("ann_date")
    grid_ann = pd.merge_asof(
        grid.sort_values("date"),
        ann[["ts_code", "ann_date"]].sort_values("ann_date"),
        left_on="date",
        right_on="ann_date",
        by="ts_code",
        direction="backward",
    )
    grid_ann["days_since_disclosure"] = (grid_ann["date"] - grid_ann["ann_date"]).dt.days

    out = grid_pre[["date", "ts_code", "days_to_disclosure"]].merge(
        grid_ann[["date", "ts_code", "days_since_disclosure"]], on=["date", "ts_code"], how="outer")
    out["date"] = out["date"].dt.strftime("%Y%m%d")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="memory/dataset/features_h10.parquet")
    parser.add_argument("--output", default="memory/dataset/features_h10_phase2.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    dates = sorted(df["date"].unique())
    ts_codes = df["ts_code"].astype(str).unique().tolist()
    print(f"[append_phase2] {len(df)} rows, {len(dates)} dates, {len(ts_codes)} stocks")

    # 市场特征：每日期一条
    market_rows = []
    for i, d in enumerate(dates):
        if i % 10 == 0:
            print(f"[append_phase2] market {i}/{len(dates)} {d}")
        feats = build_market_features(d)
        feats["date"] = d
        market_rows.append(feats)
    market_df = pd.DataFrame(market_rows)

    # 龙虎榜特征：每日期一个 DataFrame，合并到市场特征
    top_list_parts = []
    for i, d in enumerate(dates):
        if i % 10 == 0:
            print(f"[append_phase2] top_list {i}/{len(dates)} {d}")
        tdf = build_top_list_features(d)
        if not tdf.empty:
            tdf = tdf.drop_duplicates(subset=["ts_code"], keep="first")
            tdf["date"] = d
            top_list_parts.append(tdf)
    top_list_df = pd.concat(top_list_parts, ignore_index=True) if top_list_parts else pd.DataFrame()

    # 披露日特征
    print("[append_phase2] building disclosure features...")
    disc_df = build_disclosure_features(dates, ts_codes)

    # 合并
    merged = df.merge(market_df, on="date", how="left")
    if not top_list_df.empty:
        merged = merged.merge(top_list_df, on=["date", "ts_code"], how="left")
    if not disc_df.empty:
        merged = merged.merge(disc_df, on=["date", "ts_code"], how="left")

    # 填充缺失
    for col in ["top_list_flag"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # 对 Phase 2 特征做与 Phase 1 一致的截尾/中性化/rank
    from feature_engineering import build_numeric_features
    phase2_cols = [
        "margin_total_balance", "margin_financing_ratio",
        "northbound_net", "northbound_net_5d",
        "top_list_flag", "top_list_net_amount", "top_list_amount_rate",
        "top_list_turnover_rate", "top_list_pct_change",
        "days_to_disclosure", "days_since_disclosure",
    ]
    merged = build_numeric_features(merged, feature_cols=[c for c in phase2_cols if c in merged.columns],
                                    neutralize=True, rank=True, winsorize=True)

    print(f"[append_phase2] final shape {merged.shape}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    print(f"[append_phase2] saved to {args.output}")


if __name__ == "__main__":
    main()
