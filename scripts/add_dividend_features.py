"""
为数据集添加分红特征

用法：
    python add_dividend_features.py --input memory/dataset/features_h10_enhanced_fixed_v2.parquet --output memory/dataset/features_h10_with_dividend.parquet
"""
import sys
import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import tushare_call
from dividend_features import calculate_dividend_features, get_dividend_data

warnings.filterwarnings("ignore")


def add_dividend_features_to_dataset(input_path: str, output_path: str):
    """为数据集添加分红特征。"""
    print(f"加载数据集: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"原始数据集: {len(df)} 行, {len(df.columns)} 列")
    
    # 获取所有股票代码
    stocks = df["ts_code"].unique()
    print(f"股票数量: {len(stocks)}")
    
    # 预加载所有股票的分红数据
    print("预加载分红数据...")
    dividend_cache = {}
    for i, code in enumerate(stocks):
        if (i + 1) % 50 == 0:
            print(f"  已加载 {i + 1}/{len(stocks)}")
        try:
            dividend_cache[code] = get_dividend_data(code)
        except Exception:
            dividend_cache[code] = pd.DataFrame()
    
    # 计算分红特征
    print("计算分红特征...")
    dividend_features = []
    for idx, row in df.iterrows():
        if (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx + 1}/{len(df)}")
        
        code = row["ts_code"]
        date = row["date"]
        price = row.get("close", 0)
        
        # 从缓存获取分红数据
        div_df = dividend_cache.get(code, pd.DataFrame())
        
        # 计算特征
        features = {
            "days_to_ex_date": 999,
            "dividend_yield": 0.0,
            "dividend_frequency": 0,
            "dividend_growth": 0.0,
            "is_ex_date_5d": 0,
            "historical_ex_drop": 0.0,
        }
        
        if not div_df.empty:
            try:
                current_date = pd.to_datetime(date, format="%Y%m%d")
                
                # 计算距离最近除权除息日的天数
                ex_dates = div_df["ex_date"].dropna().sort_values()
                if not ex_dates.empty:
                    future_ex = ex_dates[ex_dates >= current_date]
                    past_ex = ex_dates[ex_dates < current_date]
                    
                    if not future_ex.empty:
                        next_ex = future_ex.iloc[0]
                        features["days_to_ex_date"] = (next_ex - current_date).days
                    elif not past_ex.empty:
                        last_ex = past_ex.iloc[-1]
                        features["days_to_ex_date"] = (last_ex - current_date).days
                
                # 计算股息率
                cash_divs = div_df[(div_df["cash_div_tax"] > 0) & (div_df["div_proc"] == "实施")]
                if not cash_divs.empty:
                    latest_div = cash_divs.iloc[0]
                    if price > 0:
                        features["dividend_yield"] = latest_div["cash_div_tax"] / price
                
                # 计算分红频率
                three_years_ago = current_date - pd.Timedelta(days=365 * 3)
                recent_divs = div_df[(div_df["ann_date"] >= three_years_ago) & (div_df["div_proc"] == "实施")]
                features["dividend_frequency"] = len(recent_divs)
                
                # 计算分红增长率
                if len(cash_divs) >= 2:
                    latest = cash_divs.iloc[0]["cash_div_tax"]
                    previous = cash_divs.iloc[1]["cash_div_tax"]
                    if previous > 0:
                        features["dividend_growth"] = (latest - previous) / previous
                
                # 5天内是否有除权除息日
                if features["days_to_ex_date"] >= 0 and features["days_to_ex_date"] <= 5:
                    features["is_ex_date_5d"] = 1
                
            except Exception:
                pass
        
        dividend_features.append(features)
    
    # 合并特征
    dividend_df = pd.DataFrame(dividend_features)
    df = pd.concat([df, dividend_df], axis=1)
    
    print(f"添加分红特征后: {len(df)} 行, {len(df.columns)} 列")
    print(f"新增列: {list(dividend_df.columns)}")
    
    # 保存
    df.to_parquet(output_path, index=False)
    print(f"保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="为数据集添加分红特征")
    parser.add_argument("--input", required=True, help="输入数据集路径")
    parser.add_argument("--output", required=True, help="输出数据集路径")
    args = parser.parse_args()
    
    add_dividend_features_to_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
