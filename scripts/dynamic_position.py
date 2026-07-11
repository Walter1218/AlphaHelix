"""
动态持仓策略脚本

根据模型得分动态调整持仓数量，提升胜率或累计超额。

策略：
1. 胜率优先：Top-5% 中选 Top-1（胜率 54.0%）
2. 累计超额优先：得分>0 中选 Top-10（累计 +218.1%）
3. 平衡：Top-10% 中选 Top-3（胜率 51.3%）

用法：
    python dynamic_position.py --strategy win_rate --date 20260601
    python dynamic_position.py --strategy balanced --date 20260601
    python dynamic_position.py --strategy cumulative --date 20260601
"""
import sys
import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

# 策略配置
STRATEGIES = {
    "win_rate": {
        "name": "胜率优先",
        "description": "Top-5% 中选 Top-1",
        "score_pct_threshold": 0.95,
        "max_positions": 1,
    },
    "cumulative": {
        "name": "累计超额优先",
        "description": "得分>0 中选 Top-10",
        "score_threshold": 0,
        "max_positions": 10,
    },
    "balanced": {
        "name": "平衡策略",
        "description": "Top-10% 中选 Top-3",
        "score_pct_threshold": 0.9,
        "max_positions": 3,
    },
}


def load_predictions(pred_path: str = None) -> pd.DataFrame:
    """加载预测文件。"""
    if pred_path is None:
        pred_path = "memory/predictions/predictions_h10_stacking_5model.parquet"
    
    pred = pd.read_parquet(pred_path)
    pred["date"] = pd.to_datetime(pred["date"])
    return pred


def apply_strategy(pred: pd.DataFrame, strategy: str, date: str = None) -> pd.DataFrame:
    """
    应用动态持仓策略。
    
    Args:
        pred: 预测数据
        strategy: 策略名称
        date: 指定日期（可选）
    
    Returns:
        筛选后的持仓
    """
    config = STRATEGIES.get(strategy)
    if config is None:
        raise ValueError(f"未知策略: {strategy}")
    
    if date:
        pred = pred[pred["date"] == pd.to_datetime(date)]
    
    if pred.empty:
        return pd.DataFrame()
    
    # 计算得分百分位
    pred = pred.copy()
    pred["score_pct"] = pred.groupby("date")["predicted"].rank(pct=True)
    
    # 应用策略
    if "score_pct_threshold" in config:
        # 按百分位筛选
        threshold = config["score_pct_threshold"]
        filtered = pred[pred["score_pct"] >= threshold]
    elif "score_threshold" in config:
        # 按绝对值筛选
        threshold = config["score_threshold"]
        filtered = pred[pred["predicted"] > threshold]
    else:
        filtered = pred
    
    # 按得分排名，取 Top-N
    max_positions = config["max_positions"]
    filtered = filtered.copy()
    filtered["rank"] = filtered.groupby("date")["predicted"].rank(ascending=False)
    selected = filtered[filtered["rank"] <= max_positions]
    
    return selected


def analyze_strategy(pred: pd.DataFrame, strategy: str) -> dict:
    """分析策略效果。"""
    config = STRATEGIES.get(strategy)
    if config is None:
        return {}
    
    # 应用策略
    selected = apply_strategy(pred, strategy)
    
    if selected.empty:
        return {"strategy": strategy, "win_rate": 0, "cum_excess": 0, "avg_positions": 0}
    
    # 计算指标
    win_rate = (selected["excess_return"] > 0).mean()
    avg_excess = selected["excess_return"].mean()
    cum_excess = selected.groupby("date")["excess_return"].mean().sum()
    avg_positions = selected.groupby("date").size().mean()
    
    return {
        "strategy": strategy,
        "name": config["name"],
        "description": config["description"],
        "win_rate": win_rate,
        "avg_excess": avg_excess,
        "cum_excess": cum_excess,
        "avg_positions": avg_positions,
        "total_trades": len(selected),
    }


def main():
    parser = argparse.ArgumentParser(description="动态持仓策略")
    parser.add_argument("--strategy", choices=["win_rate", "cumulative", "balanced"], 
                       default="balanced", help="策略名称")
    parser.add_argument("--date", "-d", default="", help="指定日期 YYYYMMDD")
    parser.add_argument("--pred-path", default=None, help="预测文件路径")
    parser.add_argument("--analyze", action="store_true", help="分析所有策略")
    args = parser.parse_args()
    
    pred = load_predictions(args.pred_path)
    
    if args.analyze:
        print("=== 动态持仓策略分析 ===\n")
        print(f"{'策略':<15} {'描述':<20} {'胜率':<10} {'平均超额':<12} {'累计超额':<12} {'平均持仓':<10}")
        print("-" * 80)
        
        for strategy in ["win_rate", "cumulative", "balanced"]:
            result = analyze_strategy(pred, strategy)
            print(f"{result['strategy']:<15} {result['description']:<20} "
                  f"{result['win_rate']:<10.1%} {result['avg_excess']:<12.4f} "
                  f"{result['cum_excess']:<12.2%} {result['avg_positions']:<10.1f}")
        
        # 固定 Top-10 作为基线
        pred_sorted = pred.copy()
        pred_sorted["rank"] = pred_sorted.groupby("date")["predicted"].rank(ascending=False)
        top10 = pred_sorted[pred_sorted["rank"] <= 10]
        wr = (top10["excess_return"] > 0).mean()
        ce = top10.groupby("date")["excess_return"].mean().sum()
        print(f"\n{'固定Top-10':<15} {'基线':<20} {wr:<10.1%} "
              f"{top10['excess_return'].mean():<12.4f} {ce:<12.2%} {10:<10}")
    else:
        # 应用指定策略
        selected = apply_strategy(pred, args.strategy, args.date)
        
        if selected.empty:
            print(f"策略 {args.strategy} 在指定日期无持仓")
            return
        
        config = STRATEGIES[args.strategy]
        print(f"=== {config['name']} ===")
        print(f"描述: {config['description']}")
        print(f"日期: {args.date if args.date else '全部'}")
        print(f"\n{'排名':<6} {'股票':<12} {'行业':<10} {'得分':<12} {'得分百分位':<12}")
        print("-" * 55)
        
        for _, row in selected.iterrows():
            print(f"{int(row.get('rank', 0)):<6} {row['ts_code']:<12} "
                  f"{row.get('industry', ''):<10} {row['predicted']:<12.6f} "
                  f"{row.get('score_pct', 0):<12.2%}")


if __name__ == "__main__":
    main()
