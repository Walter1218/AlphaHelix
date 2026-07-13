"""
交易成本模型

实现 A 股交易成本计算，包括佣金、印花税、滑点等。

用法：
    python transaction_cost.py --amount 100000 --is-open
"""
import sys
import os
import argparse
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")


# A 股交易成本参数
DEFAULT_COST_CONFIG = {
    # 佣金
    "commission_rate": 0.0003,  # 佣金费率 0.03%（万三）
    "min_commission": 5,  # 最低佣金 5 元
    
    # 印花税（卖出时收取）
    "stamp_tax_rate": 0.0005,  # 印花税 0.05%（万五）
    
    # 过户费
    "transfer_fee_rate": 0.00001,  # 过户费 0.001%（十万分之一）
    
    # 滑点
    "slippage_rate": 0.001,  # 滑点 0.1%
}


class TransactionCost:
    """交易成本计算器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_COST_CONFIG
    
    def calculate(self, amount: float, is_open: bool = True) -> Dict:
        """
        计算交易成本
        
        Args:
            amount: 交易金额
            is_open: 是否是买入（True=买入，False=卖出）
        
        Returns:
            成本明细字典
        """
        # 佣金
        commission = amount * self.config["commission_rate"]
        commission = max(commission, self.config["min_commission"])
        
        # 印花税（只有卖出时收取）
        stamp_tax = amount * self.config["stamp_tax_rate"] if not is_open else 0
        
        # 过户费
        transfer_fee = amount * self.config["transfer_fee_rate"]
        
        # 滑点
        slippage = amount * self.config["slippage_rate"]
        
        # 总成本
        total_cost = commission + stamp_tax + transfer_fee + slippage
        
        return {
            "commission": commission,
            "stamp_tax": stamp_tax,
            "transfer_fee": transfer_fee,
            "slippage": slippage,
            "total_cost": total_cost,
            "cost_rate": total_cost / amount if amount > 0 else 0,
        }
    
    def calculate_round_trip(self, amount: float) -> Dict:
        """
        计算往返交易成本（买入+卖出）
        
        Args:
            amount: 交易金额
        
        Returns:
            成本明细字典
        """
        open_cost = self.calculate(amount, is_open=True)
        close_cost = self.calculate(amount, is_open=False)
        
        return {
            "open_cost": open_cost["total_cost"],
            "close_cost": close_cost["total_cost"],
            "total_cost": open_cost["total_cost"] + close_cost["total_cost"],
            "cost_rate": (open_cost["total_cost"] + close_cost["total_cost"]) / amount if amount > 0 else 0,
        }


class PortfolioTransactionCost:
    """组合交易成本计算器"""
    
    def __init__(self, cost_calculator: TransactionCost = None):
        self.cost_calculator = cost_calculator or TransactionCost()
    
    def calculate_rebalance_cost(
        self,
        old_positions: Dict[str, float],
        new_positions: Dict[str, float],
        total_capital: float,
    ) -> Dict:
        """
        计算调仓成本
        
        Args:
            old_positions: 旧持仓 {股票代码: 权重}
            new_positions: 新持仓 {股票代码: 权重}
            total_capital: 总资金
        
        Returns:
            调仓成本明细
        """
        # 计算需要卖出的股票
        to_sell = {}
        for stock, weight in old_positions.items():
            if stock not in new_positions:
                to_sell[stock] = weight
            else:
                weight_diff = weight - new_positions[stock]
                if weight_diff > 0:
                    to_sell[stock] = weight_diff
        
        # 计算需要买入的股票
        to_buy = {}
        for stock, weight in new_positions.items():
            if stock not in old_positions:
                to_buy[stock] = weight
            else:
                weight_diff = weight - old_positions[stock]
                if weight_diff > 0:
                    to_buy[stock] = weight_diff
        
        # 计算卖出成本
        sell_cost = 0
        for stock, weight in to_sell.items():
            amount = weight * total_capital
            cost = self.cost_calculator.calculate(amount, is_open=False)
            sell_cost += cost["total_cost"]
        
        # 计算买入成本
        buy_cost = 0
        for stock, weight in to_buy.items():
            amount = weight * total_capital
            cost = self.cost_calculator.calculate(amount, is_open=True)
            buy_cost += cost["total_cost"]
        
        total_cost = sell_cost + buy_cost
        
        return {
            "sell_cost": sell_cost,
            "buy_cost": buy_cost,
            "total_cost": total_cost,
            "cost_rate": total_cost / total_capital if total_capital > 0 else 0,
            "num_sell": len(to_sell),
            "num_buy": len(to_buy),
            "to_sell": to_sell,
            "to_buy": to_buy,
        }


def main():
    parser = argparse.ArgumentParser(description="交易成本计算")
    parser.add_argument("--amount", type=float, default=100000, help="交易金额")
    parser.add_argument("--is-open", action="store_true", help="是否是买入")
    parser.add_argument("--round-trip", action="store_true", help="计算往返成本")
    args = parser.parse_args()
    
    calc = TransactionCost()
    
    if args.round_trip:
        result = calc.calculate_round_trip(args.amount)
        print(f"=== 往返交易成本 ===")
        print(f"交易金额: {args.amount:,.2f}")
        print(f"买入成本: {result['open_cost']:,.2f}")
        print(f"卖出成本: {result['close_cost']:,.2f}")
        print(f"总成本: {result['total_cost']:,.2f}")
        print(f"成本率: {result['cost_rate']:.4%}")
    else:
        result = calc.calculate(args.amount, args.is_open)
        action = "买入" if args.is_open else "卖出"
        print(f"=== {action}交易成本 ===")
        print(f"交易金额: {args.amount:,.2f}")
        print(f"佣金: {result['commission']:,.2f}")
        print(f"印花税: {result['stamp_tax']:,.2f}")
        print(f"过户费: {result['transfer_fee']:,.2f}")
        print(f"滑点: {result['slippage']:,.2f}")
        print(f"总成本: {result['total_cost']:,.2f}")
        print(f"成本率: {result['cost_rate']:.4%}")


if __name__ == "__main__":
    main()
