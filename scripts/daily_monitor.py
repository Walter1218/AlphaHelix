"""
每日监控脚本

跟踪模型表现：
1. 预测准确率
2. 累计收益
3. 最大回撤
4. 夏普比率

用法：
    python daily_monitor.py
"""
import sys
import os
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

MONITOR_FILE = "memory/monitor/performance.json"


def load_history() -> list:
    """加载历史记录"""
    if os.path.exists(MONITOR_FILE):
        with open(MONITOR_FILE, 'r') as f:
            return json.load(f)
    return []


def save_history(history: list):
    """保存历史记录"""
    os.makedirs(os.path.dirname(MONITOR_FILE), exist_ok=True)
    with open(MONITOR_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def verify_predictions(date: str) -> dict:
    """验证某天的预测"""
    import tushare as ts
    
    # 加载预测
    pred_path = f"memory/predictions/predictions_{date}.parquet"
    if not os.path.exists(pred_path):
        return None
    
    predictions = pd.read_parquet(pred_path)
    
    # 获取实际收益
    token = os.getenv('TUSHARE_TOKEN', '')
    pro = ts.pro_api(token)
    
    # 计算次日收益
    next_date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y%m%d')
    
    results = []
    for _, row in predictions.iterrows():
        ts_code = row['ts_code']
        try:
            df = pro.daily(ts_code=ts_code, start_date=date.replace('-', ''), end_date=next_date)
            if df is not None and len(df) >= 2:
                actual_return = (df.iloc[0]['close'] - df.iloc[1]['close']) / df.iloc[1]['close']
                results.append({
                    'ts_code': ts_code,
                    'predicted': row.get('predicted', 0),
                    'actual': actual_return,
                    'correct': actual_return > 0,
                })
        except:
            continue
    
    if not results:
        return None
    
    df_results = pd.DataFrame(results)
    
    # 计算 Top-10 表现
    top10 = df_results.nlargest(10, 'predicted')
    top10_return = top10['actual'].mean()
    top10_correct = (top10['actual'] > 0).sum()
    
    return {
        'date': date,
        'total_stocks': len(df_results),
        'top10_return': top10_return,
        'top10_correct': top10_correct,
        'top10_total': len(top10),
        'top10_win_rate': top10_correct / len(top10),
        'overall_ic': df_results['predicted'].corr(df_results['actual']),
    }


def update_monitor():
    """更新监控数据"""
    history = load_history()
    
    # 获取昨天日期
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # 检查是否已记录
    if any(h['date'] == yesterday for h in history):
        print(f"已记录 {yesterday} 的数据")
        return
    
    # 验证预测
    result = verify_predictions(yesterday)
    if result:
        history.append(result)
        save_history(history)
        print(f"已记录 {yesterday}: 胜率={result['top10_win_rate']:.1%}, 收益={result['top10_return']:.2%}")
    else:
        print(f"无法验证 {yesterday} 的预测")


def show_performance():
    """显示历史表现"""
    history = load_history()
    if not history:
        print("暂无历史数据")
        return
    
    df = pd.DataFrame(history)
    
    print("=== 模型表现监控 ===")
    print()
    print(f"记录天数: {len(df)}")
    print(f"平均胜率: {df['top10_win_rate'].mean():.1%}")
    print(f"平均收益: {df['top10_return'].mean():.2%}")
    print(f"累计收益: {(1 + df['top10_return']).prod() - 1:.2%}")
    print(f"夏普比率: {df['top10_return'].mean() / (df['top10_return'].std() + 1e-6) * np.sqrt(252):.2f}")
    
    # 最大回撤
    cumulative = (1 + df['top10_return']).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    print(f"最大回撤: {drawdown.min():.2%}")
    
    print()
    print("=== 最近10天 ===")
    recent = df.tail(10)
    for _, row in recent.iterrows():
        emoji = "✅" if row['top10_win_rate'] > 0.5 else "❌"
        print(f"{emoji} {row['date']}: 胜率={row['top10_win_rate']:.1%}, 收益={row['top10_return']:.2%}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="每日监控")
    parser.add_argument("--update", action="store_true", help="更新监控数据")
    parser.add_argument("--show", action="store_true", help="显示历史表现")
    args = parser.parse_args()
    
    if args.update:
        update_monitor()
    elif args.show:
        show_performance()
    else:
        # 默认：更新并显示
        update_monitor()
        show_performance()


if __name__ == "__main__":
    main()
