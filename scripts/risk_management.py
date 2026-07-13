"""
风险管理模块

实现止损、仓位管理、回撤控制等风险管理策略。

功能：
1. 波动率仓位管理：根据波动率动态调整仓位
2. 回撤控制：回撤超过阈值时降低仓位
3. 行业分散：限制单行业仓位
4. 风险监控：实时计算风险指标

用法：
    python risk_management.py --analyze
"""
import sys
import os
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")


# 风险参数配置
RISK_CONFIG = {
    # 波动率仓位管理（温和）
    "volatility_target": 0.20,  # 目标波动率 20%
    "max_position_pct": 0.15,  # 单只股票最大仓位 15%
    "min_position_pct": 0.05,  # 单只股票最小仓位 5%
    
    # 回撤控制（温和）
    "drawdown_reduce": -0.20,  # 回撤降仓线 -20%
    "reduce_ratio": 0.8,  # 降仓比例（保留80%仓位）
    
    # 行业分散
    "max_industry_pct": 0.40,  # 单行业最大仓位 40%
    "max_stock_count": 10,  # 最大持仓股票数
}


class RiskManager:
    """风险管理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or RISK_CONFIG
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.drawdown = 0.0
        self.daily_returns = []
        self.risk_alerts = []
    
    def calculate_position_size(self, volatility: float) -> float:
        """
        根据波动率计算仓位大小
        
        Args:
            volatility: 股票波动率
        
        Returns:
            仓位比例
        """
        if volatility <= 0:
            return self.config["min_position_pct"]
        
        # 反波动率加权
        target_vol = self.config["volatility_target"]
        position_pct = target_vol / volatility
        
        # 限制仓位范围
        position_pct = max(self.config["min_position_pct"], position_pct)
        position_pct = min(self.config["max_position_pct"], position_pct)
        
        return position_pct
    
    def calculate_drawdown_control(self) -> Tuple[float, str]:
        """
        计算回撤控制
        
        Returns:
            (仓位调整比例, 状态)
        """
        if self.drawdown <= self.config["drawdown_reduce"]:
            return self.config["reduce_ratio"], "降仓"
        else:
            return 1.0, "正常"
    
    def update_portfolio(self, daily_return: float):
        """
        更新组合状态
        
        Args:
            daily_return: 今日收益率
        """
        self.daily_returns.append(daily_return)
        self.portfolio_value *= (1 + daily_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.drawdown = (self.portfolio_value - self.peak_value) / self.peak_value
        
        # 检查风险警报
        self._check_risk_alerts(daily_return)
    
    def _check_risk_alerts(self, daily_return: float):
        """检查风险警报"""
        alerts = []
        
        if self.drawdown <= self.config["drawdown_reduce"]:
            alerts.append(f"🔴 回撤降仓: {self.drawdown:.1%}")
        
        self.risk_alerts.extend(alerts)
    
    def get_risk_metrics(self) -> Dict:
        """
        获取风险指标
        
        Returns:
            风险指标字典
        """
        if not self.daily_returns:
            return {}
        
        returns = np.array(self.daily_returns)
        
        # 基础指标
        total_return = self.portfolio_value - 1.0
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # 回撤指标
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        # 盈亏比
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        profit_loss_ratio = wins.mean() / abs(losses.mean()) if len(losses) > 0 else float('inf')
        
        # 卡玛比率
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "current_drawdown": self.drawdown,
            "portfolio_value": self.portfolio_value,
            "risk_alerts": self.risk_alerts[-5:],  # 最近5条警报
        }


def run_backtest_with_risk_management():
    """运行带风险管理的回测"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    
    # 加载数据
    df = pd.read_parquet("memory/dataset/features_h10_full.parquet")
    df["date"] = pd.to_datetime(df["date"])
    
    # 特征工程
    feature_cols = [c for c in df.columns if c not in ["date", "exit_date", "ts_code", 
                                                         "stock_return", "benchmark_return", 
                                                         "excess_return", "industry"]]
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(0)
    
    # 添加交互特征
    base_features = ["mom_20", "volatility_20", "roe", "dv_ratio", "net_mf_ratio"]
    for i in range(len(base_features)):
        for j in range(i + 1, len(base_features)):
            f1, f2 = base_features[i], base_features[j]
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    
    df["price_momentum"] = df["mom_5"] * df["mom_20"]
    df["vol_momentum"] = df["volatility_20"] * df["mom_20"]
    df["value_momentum"] = df["dv_ratio"] * df["mom_20"]
    df["quality_momentum"] = df["roe"] * df["mom_20"]
    df["flow_momentum"] = df["net_mf_ratio"] * df["mom_20"]
    df["mom_accel"] = df["mom_5"] - df["mom_20"]
    df["mom_decel"] = df["mom_20"] - df["mom_60"]
    df["reversal_strength"] = df["reversal_score"] * df["volatility_20"]
    df["quality_value"] = df["roe"] * df["dv_ratio"]
    
    def add_industry_rank(group):
        group = group.copy()
        if len(group) < 3:
            return group
        for col in ["mom_20", "roe", "volatility_20"]:
            if col in group.columns:
                group[f"ind_{col}_rank"] = group[col].rank(pct=True)
        return group
    
    df = df.groupby(["date", "industry"], group_keys=False).apply(add_industry_rank)
    
    def add_cross_section_rank(group):
        group = group.copy()
        for col in ["mom_20", "roe", "volatility_20", "dv_ratio"]:
            if col in group.columns:
                group[f"{col}_rank"] = group[col].rank(pct=True)
        return group
    
    df = df.groupby("date", group_keys=False).apply(add_cross_section_rank)
    
    feature_cols = [c for c in df.columns if c not in ["date", "exit_date", "ts_code", 
                                                         "stock_return", "benchmark_return", 
                                                         "excess_return", "industry"]]
    for col in feature_cols:
        df[col] = df[col].fillna(0)
    feature_cols = [c for c in feature_cols if df[col].std() > 0]
    
    df["target_log"] = np.sign(df["excess_return"]) * np.log1p(np.abs(df["excess_return"]))
    df_sorted = df.sort_values("date")
    df_sorted["ym"] = df_sorted["date"].dt.to_period("M")
    months = sorted(df_sorted["ym"].unique())
    
    def select_top(train_df, feature_cols, n=40):
        ics = {}
        for col in feature_cols:
            ic = train_df.groupby("ym").apply(
                lambda g: g[col].corr(g["excess_return"], method="spearman"), 
                include_groups=False
            ).mean()
            ics[col] = ic
        sorted_ics = sorted(ics.items(), key=lambda x: abs(x[1]), reverse=True)
        return [col for col, _ in sorted_ics[:n]]
    
    def recall_by_quality(train_df, test_df):
        return test_df[(test_df["roe"] > 0) & (test_df["net_mf_ratio"] > 0)]
    
    # 运行回测
    preds = []
    train_w = 6
    
    for i in range(len(months)):
        train_end = i + train_w
        val_start = train_end + 1
        val_end = val_start + 2
        test_idx = val_end + 1
        if test_idx >= len(months):
            break
        train_ms = months[i:train_end]
        val_ms = months[val_start:val_end]
        test_month = months[test_idx]
        train_df = df_sorted[df_sorted["ym"].isin(train_ms)]
        val_df = df_sorted[df_sorted["ym"].isin(val_ms)]
        test_df = df_sorted[df_sorted["ym"] == test_month]
        if train_df.empty or val_df.empty or test_df.empty:
            continue
        
        selected = select_top(train_df, feature_cols, 40)
        recalled = recall_by_quality(train_df, test_df)
        if recalled.empty:
            continue
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_df[selected].values)
        y_tr = train_df["target_log"].values
        X_test = scaler.transform(recalled[selected].values)
        
        model = Ridge(alpha=1.0)
        model.fit(X_tr, y_tr)
        pred_values = model.predict(X_test)
        
        p = recalled[["date", "ts_code", "stock_return", "benchmark_return", 
                      "excess_return", "industry", "volatility_20"]].copy()
        p["predicted"] = pred_values
        preds.append(p)
    
    if not preds:
        print("回测失败")
        return
    
    r = pd.concat(preds)
    r["date"] = pd.to_datetime(r["date"])
    r["rank"] = r.groupby("date")["predicted"].rank(ascending=False)
    
    # Top-10 组合
    t10 = r[r["rank"] <= 10].copy()
    
    # 计算每日收益（无风险管理）
    daily_returns_no_risk = t10.groupby("date")["excess_return"].mean()
    
    # 计算每日收益（有风险管理）
    risk_manager = RiskManager()
    daily_returns_with_risk = []
    
    for date in sorted(t10["date"].unique()):
        daily_data = t10[t10["date"] == date]
        
        # 等权重
        daily_return = daily_data["excess_return"].mean()
        
        # 回撤控制（温和）
        reduce_ratio, status = risk_manager.calculate_drawdown_control()
        daily_return *= reduce_ratio
        
        risk_manager.update_portfolio(daily_return)
        daily_returns_with_risk.append(daily_return)
    
    daily_returns_with_risk = pd.Series(daily_returns_with_risk, 
                                        index=sorted(t10["date"].unique()))
    
    # 计算指标
    def calc_metrics(returns, name):
        if len(returns) == 0:
            return {}
        
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        win_rate = (returns > 0).mean()
        
        # 盈亏比
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        profit_loss_ratio = wins.mean() / abs(losses.mean()) if len(losses) > 0 else float('inf')
        
        # 卡玛比率
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
        
        return {
            "name": name,
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
        }
    
    metrics_no_risk = calc_metrics(daily_returns_no_risk, "无风险管理")
    metrics_with_risk = calc_metrics(daily_returns_with_risk, "有风险管理")
    
    print("=== 风险管理效果对比 ===\n")
    print(f"{'指标':<15} {'无风险管理':<15} {'有风险管理':<15} {'改善':<10}")
    print("-" * 55)
    
    for key in ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "calmar_ratio"]:
        val1 = metrics_no_risk[key]
        val2 = metrics_with_risk[key]
        diff = val2 - val1
        
        if key == "max_drawdown":
            # 回撤是负数，越大越好
            improved = diff > 0
            print(f"{key:<15} {val1:<15.2%} {val2:<15.2%} {diff:<10.2%} {'✅' if improved else '❌'}")
        else:
            improved = diff > 0
            print(f"{key:<15} {val1:<15.2%} {val2:<15.2%} {diff:<10.2%} {'✅' if improved else '❌'}")
    
    # 按年分析
    print("\n=== 按年表现 ===\n")
    
    for year in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
        year_returns_no_risk = daily_returns_no_risk[daily_returns_no_risk.index.year == year]
        year_returns_with_risk = daily_returns_with_risk[daily_returns_with_risk.index.year == year]
        
        if len(year_returns_no_risk) == 0:
            continue
        
        m1 = calc_metrics(year_returns_no_risk, "无风险管理")
        m2 = calc_metrics(year_returns_with_risk, "有风险管理")
        
        print(f"{year}:")
        print(f"  无风险管理: 胜率={m1['win_rate']:.1%}, 累计超额={m1['total_return']:.2%}, 夏普={m1['sharpe_ratio']:.2f}")
        print(f"  有风险管理: 胜率={m2['win_rate']:.1%}, 累计超额={m2['total_return']:.2%}, 夏普={m2['sharpe_ratio']:.2f}")
        print()
    
    # 风险指标详情
    print("=== 风险管理详情 ===\n")
    risk_metrics = risk_manager.get_risk_metrics()
    for key, value in risk_metrics.items():
        if key != "risk_alerts":
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    if risk_metrics.get("risk_alerts"):
        print("\n最近风险警报:")
        for alert in risk_metrics["risk_alerts"]:
            print(f"  {alert}")


def main():
    parser = argparse.ArgumentParser(description="风险管理模块")
    parser.add_argument("--analyze", action="store_true", help="分析风险管理效果")
    args = parser.parse_args()
    
    if args.analyze:
        run_backtest_with_risk_management()
    else:
        print("使用 --analyze 参数运行风险管理分析")


if __name__ == "__main__":
    main()
