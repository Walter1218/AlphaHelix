"""
组合策略

实现 TopkDropout 策略，限制每日换仓数量。

用法：
    python portfolio_strategy.py --date 20260601
"""
import sys
import os
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")


class TopkDropoutStrategy:
    """
    TopkDropout 策略
    
    每日持有预测分数最高的 topk 只股票，每日最多换仓 n_drop 只。
    """
    
    def __init__(self, topk: int = 50, n_drop: int = 5):
        """
        Args:
            topk: 持仓股票数量
            n_drop: 每日最多换仓数量
        """
        self.topk = topk
        self.n_drop = n_drop
        self.current_positions: List[str] = []
    
    def generate_signal(
        self, predictions: pd.DataFrame, current_positions: List[str] = None
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        生成换仓信号
        
        Args:
            predictions: 预测数据，包含 ts_code 和 predicted 列
            current_positions: 当前持仓股票列表
        
        Returns:
            (保持持仓, 卖出股票, 买入股票)
        """
        if current_positions is None:
            current_positions = self.current_positions
        
        # 按预测分数排序
        ranked = predictions.sort_values("predicted", ascending=False)
        
        # 选择 topk
        target_stocks = ranked.head(self.topk)["ts_code"].tolist()
        
        # 计算换仓
        to_sell = [s for s in current_positions if s not in target_stocks]
        to_buy = [s for s in target_stocks if s not in current_positions]
        
        # 限制换仓数量
        to_sell = to_sell[: self.n_drop]
        to_buy = to_buy[: self.n_drop]
        
        # 计算保持持仓
        to_hold = [s for s in current_positions if s not in to_sell]
        
        # 更新当前持仓
        self.current_positions = to_hold + to_buy
        
        return to_hold, to_sell, to_buy
    
    def get_position_weights(
        self, predictions: pd.DataFrame, to_hold: List[str], to_buy: List[str]
    ) -> Dict[str, float]:
        """
        计算持仓权重（等权重）
        
        Args:
            predictions: 预测数据
            to_hold: 保持持仓
            to_buy: 买入股票
        
        Returns:
            股票权重字典
        """
        all_positions = to_hold + to_buy
        weight = 1.0 / len(all_positions) if all_positions else 0
        return {stock: weight for stock in all_positions}


class PortfolioManager:
    """组合管理器"""
    
    def __init__(self, strategy: TopkDropoutStrategy, initial_capital: float = 100000000):
        self.strategy = strategy
        self.capital = initial_capital
        self.positions: Dict[str, float] = {}  # {股票代码: 持仓金额}
        self.history: List[Dict] = []
    
    def rebalance(self, predictions: pd.DataFrame, date: str) -> Dict:
        """
        调仓
        
        Args:
            predictions: 预测数据
            date: 日期
        
        Returns:
            调仓结果
        """
        # 获取当前持仓股票列表
        current_stocks = list(self.positions.keys())
        
        # 生成换仓信号
        to_hold, to_sell, to_buy = self.strategy.generate_signal(
            predictions, current_stocks
        )
        
        # 计算新持仓权重
        new_weights = self.strategy.get_position_weights(predictions, to_hold, to_buy)
        
        # 计算调仓金额
        sell_amount = sum(self.positions.get(s, 0) for s in to_sell)
        buy_amount = self.capital * sum(new_weights.get(s, 0) for s in to_buy)
        
        # 更新持仓
        new_positions = {}
        for stock, weight in new_weights.items():
            new_positions[stock] = self.capital * weight
        
        self.positions = new_positions
        
        # 记录历史
        record = {
            "date": date,
            "to_hold": to_hold,
            "to_sell": to_sell,
            "to_buy": to_buy,
            "sell_amount": sell_amount,
            "buy_amount": buy_amount,
            "num_positions": len(new_positions),
        }
        self.history.append(record)
        
        return record
    
    def get_current_positions(self) -> Dict[str, float]:
        """获取当前持仓"""
        return self.positions.copy()
    
    def get_position_count(self) -> int:
        """获取持仓数量"""
        return len(self.positions)


def main():
    parser = argparse.ArgumentParser(description="组合策略")
    parser.add_argument("--date", type=str, default="20260601", help="日期")
    parser.add_argument("--topk", type=int, default=50, help="持仓数量")
    parser.add_argument("--n-drop", type=int, default=5, help="每日最多换仓数量")
    args = parser.parse_args()
    
    # 创建策略
    strategy = TopkDropoutStrategy(topk=args.topk, n_drop=args.n_drop)
    manager = PortfolioManager(strategy)
    
    print(f"=== TopkDropout 策略 ===")
    print(f"持仓数量: {args.topk}")
    print(f"每日最多换仓: {args.n_drop}")
    print(f"日期: {args.date}")


if __name__ == "__main__":
    main()
