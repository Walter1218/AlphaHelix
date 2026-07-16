"""
每日数据更新脚本

从 Tushare 获取最新行情，计算特征，生成预测。

用法：
    python update_and_predict.py
"""
import sys
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tushare as ts

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")


def fetch_latest_data(days=30):
    """获取最新行情数据"""
    token = os.getenv('TUSHARE_TOKEN', '')
    pro = ts.pro_api(token)
    
    # 获取股票列表
    stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    
    # 获取最新交易日
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    
    all_data = []
    for ts_code in stocks['ts_code'].tolist():
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                df['name'] = stocks[stocks['ts_code'] == ts_code]['name'].values[0]
                df['industry'] = stocks[stocks['ts_code'] == ts_code]['industry'].values[0]
                all_data.append(df)
        except:
            continue
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None


def calculate_features(df):
    """计算特征"""
    # 按股票分组计算
    df = df.sort_values(['ts_code', 'trade_date'])
    
    # 动量
    df['mom_5'] = df.groupby('ts_code')['close'].pct_change(5)
    df['mom_20'] = df.groupby('ts_code')['close'].pct_change(20)
    
    # 波动率
    df['volatility_20'] = df.groupby('ts_code')['close'].pct_change().rolling(20).std().reset_index(level=0, drop=True)
    
    # 均线
    df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
    
    # 成交量
    df['vol_ratio'] = df['vol'] / df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(20).mean())
    
    return df


def generate_predictions(df, model_path='memory/models/double_ensemble.pkl'):
    """生成预测"""
    import pickle
    
    # 加载模型
    if not os.path.exists(model_path):
        # 如果没有保存的模型，用简单规则
        return generate_rule_based_predictions(df)
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # TODO: 使用保存的模型预测
    return generate_rule_based_predictions(df)


def generate_rule_based_predictions(df):
    """基于规则的预测"""
    # 获取最新数据
    latest_date = df['trade_date'].max()
    latest = df[df['trade_date'] == latest_date].copy()
    
    # 过滤：只保留主板和创业板
    latest = latest[latest['ts_code'].str.contains('\\.(SZ|SH)$', regex=True)]
    
    # 计算综合得分
    latest['score'] = 0.0
    
    # 动量得分（正动量加分）
    if 'mom_20' in latest.columns:
        mom_rank = latest['mom_20'].rank(pct=True)
        mom_rank = mom_rank.fillna(0.5)
        latest['score'] += mom_rank * 0.3
    
    # 波动率得分（低波动率加分）
    if 'volatility_20' in latest.columns:
        vol_rank = 1 - latest['volatility_20'].rank(pct=True)
        vol_rank = vol_rank.fillna(0.5)
        latest['score'] += vol_rank * 0.3
    
    # 成交量得分（高成交量加分）
    if 'vol_ratio' in latest.columns:
        vol_ratio_rank = latest['vol_ratio'].rank(pct=True)
        vol_ratio_rank = vol_ratio_rank.fillna(0.5)
        latest['score'] += vol_ratio_rank * 0.2
    
    # 均线得分（价格在均线上方加分）
    if 'ma5' in latest.columns and 'ma20' in latest.columns:
        ma_score = ((latest['close'] > latest['ma5']) & (latest['close'] > latest['ma20'])).astype(float)
        latest['score'] += ma_score * 0.2
    
    # 过滤掉得分异常的
    latest = latest[latest['score'].notna() & (latest['score'] > 0)]
    
    return latest


def main():
    print("=== 每日数据更新 ===")
    print()
    
    # 1. 获取最新数据
    print("1. 获取最新数据...")
    df = fetch_latest_data(days=30)
    if df is None:
        print("获取数据失败")
        return
    
    print(f"   数据行数: {len(df)}")
    print(f"   日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print()
    
    # 2. 计算特征
    print("2. 计算特征...")
    df = calculate_features(df)
    print("   特征计算完成")
    print()
    
    # 3. 生成预测
    print("3. 生成预测...")
    predictions = generate_predictions(df)
    
    # 获取 Top-10
    top10 = predictions.nlargest(10, 'score')
    
    print()
    print("=== Top-10 推荐 ===")
    print()
    
    for idx, row in enumerate(top10.iterrows(), 1):
        _, row = row
        print(f"{idx}. {row['name']} ({row['ts_code']})")
        print(f"   行业: {row['industry']}")
        print(f"   收盘价: {row['close']:.2f}")
        print(f"   5日动量: {row.get('mom_5', 0):.2%}")
        print(f"   20日动量: {row.get('mom_20', 0):.2%}")
        print(f"   综合得分: {row['score']:.4f}")
        print()
    
    # 保存预测结果
    os.makedirs('memory/predictions', exist_ok=True)
    today = datetime.now().strftime('%Y-%m-%d')
    top10[['ts_code', 'name', 'industry', 'close', 'mom_5', 'mom_20', 'score']].to_parquet(
        f'memory/predictions/predictions_{today}.parquet', index=False
    )
    print(f"预测结果已保存到 memory/predictions/predictions_{today}.parquet")


if __name__ == "__main__":
    main()
