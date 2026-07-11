"""
分红/拆股特征计算模块

从 Tushare 获取分红送股数据，计算以下特征：
- days_to_ex_date: 距离除权除息日天数（正=即将到来，负=已过）
- dividend_yield: 股息率（每股分红 / 股价）
- dividend_frequency: 分红频率（过去3年分红次数）
- dividend_growth: 分红增长率（本次 vs 上次）
- is_ex_date_5d: 5天内有除权除息日标志
- historical_ex_drop: 历史除息日平均跌幅
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import tushare_call


def get_dividend_data(ts_code: str, years: int = 3) -> pd.DataFrame:
    """获取股票分红送股数据。"""
    df = tushare_call("dividend", {"ts_code": ts_code, "limit": 50})
    if df.empty:
        return df
    
    # 只保留已实施或进度明确的记录
    df = df[df["div_proc"].isin(["实施", "股东大会通过", "预案"])].copy()
    
    # 转换日期
    for col in ["ann_date", "record_date", "ex_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
    
    return df


def calculate_dividend_features(ts_code: str, date: str, price: float) -> dict:
    """
    计算分红相关特征。
    
    Args:
        ts_code: 股票代码
        date: 基准日 YYYYMMDD
        price: 当前股价
    
    Returns:
        分红特征字典
    """
    features = {
        "days_to_ex_date": 999,  # 默认无分红
        "dividend_yield": 0.0,
        "dividend_frequency": 0,
        "dividend_growth": 0.0,
        "is_ex_date_5d": 0,
        "historical_ex_drop": 0.0,
    }
    
    try:
        df = get_dividend_data(ts_code)
        if df.empty:
            return features
        
        current_date = pd.to_datetime(date, format="%Y%m%d")
        
        # 1. 计算距离最近除权除息日的天数
        ex_dates = df["ex_date"].dropna().sort_values()
        if not ex_dates.empty:
            # 找到最近的除权除息日
            future_ex = ex_dates[ex_dates >= current_date]
            past_ex = ex_dates[ex_dates < current_date]
            
            if not future_ex.empty:
                # 有即将到来的除权除息日
                next_ex = future_ex.iloc[0]
                features["days_to_ex_date"] = (next_ex - current_date).days
            elif not past_ex.empty:
                # 最近的除权除息日已过
                last_ex = past_ex.iloc[-1]
                features["days_to_ex_date"] = (last_ex - current_date).days  # 负数
        
        # 2. 计算股息率
        # 找到最近已实施的现金分红
        cash_divs = df[(df["cash_div_tax"] > 0) & (df["div_proc"] == "实施")]
        if not cash_divs.empty:
            latest_div = cash_divs.iloc[0]
            if price > 0:
                features["dividend_yield"] = latest_div["cash_div_tax"] / price
        
        # 3. 计算分红频率（过去3年分红次数）
        three_years_ago = current_date - timedelta(days=365 * 3)
        recent_divs = df[(df["ann_date"] >= three_years_ago) & (df["div_proc"] == "实施")]
        features["dividend_frequency"] = len(recent_divs)
        
        # 4. 计算分红增长率
        if len(cash_divs) >= 2:
            latest = cash_divs.iloc[0]["cash_div_tax"]
            previous = cash_divs.iloc[1]["cash_div_tax"]
            if previous > 0:
                features["dividend_growth"] = (latest - previous) / previous
        
        # 5. 5天内是否有除权除息日
        if features["days_to_ex_date"] >= 0 and features["days_to_ex_date"] <= 5:
            features["is_ex_date_5d"] = 1
        
        # 6. 历史除息日平均跌幅（需要日线数据）
        if not past_ex.empty:
            drops = []
            for ex_date in past_ex.tail(3):  # 最近3次
                try:
                    # 获取除息日前后日线
                    ex_str = ex_date.strftime("%Y%m%d")
                    before_date = (ex_date - timedelta(days=5)).strftime("%Y%m%d")
                    after_date = (ex_date + timedelta(days=5)).strftime("%Y%m%d")
                    
                    daily = tushare_call("daily", {
                        "ts_code": ts_code,
                        "start_date": before_date,
                        "end_date": after_date,
                    })
                    
                    if not daily.empty and len(daily) >= 2:
                        daily = daily.sort_values("trade_date")
                        # 找到除息日当天或之后第一天的收盘价
                        ex_day = daily[daily["trade_date"] >= ex_str].iloc[0]
                        prev_day = daily[daily["trade_date"] < ex_str].iloc[-1]
                        drop = (ex_day["close"] - prev_day["close"]) / prev_day["close"]
                        drops.append(drop)
                except Exception:
                    continue
            
            if drops:
                features["historical_ex_drop"] = np.mean(drops)
        
    except Exception as e:
        print(f"计算分红特征失败 {ts_code}: {e}")
    
    return features


def main():
    """测试分红特征计算。"""
    import time
    
    stocks = ["600036.SH", "300750.SZ"]
    date = "20260709"
    
    for code in stocks:
        print(f"\n{'='*50}")
        print(f"计算 {code} 分红特征")
        print(f"{'='*50}")
        
        # 获取当前价格
        daily = tushare_call("daily", {"ts_code": code, "trade_date": date})
        if daily.empty:
            print(f"无法获取 {code} 价格数据")
            continue
        
        price = daily.iloc[0]["close"]
        print(f"当前价格: {price}")
        
        features = calculate_dividend_features(code, date, price)
        for k, v in features.items():
            print(f"  {k}: {v}")
        
        time.sleep(1)


if __name__ == "__main__":
    main()
