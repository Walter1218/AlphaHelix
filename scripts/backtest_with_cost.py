"""
带交易成本的回测

集成交易成本模型和换仓限制，进行更真实的回测。

用法：
    python backtest_with_cost.py
"""
import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from transaction_cost import TransactionCost, PortfolioTransactionCost


def calc_ic(pred, actual):
    """计算 IC"""
    if len(pred) < 10:
        return 0
    ic, _ = spearmanr(pred, actual)
    return ic if not np.isnan(ic) else 0


class BacktestEngine:
    """回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = 100000000,
        topk: int = 10,
        n_drop: int = 5,
        cost_config: dict = None,
    ):
        self.initial_capital = initial_capital
        self.topk = topk
        self.n_drop = n_drop
        self.cost_calculator = TransactionCost(cost_config)
        self.portfolio_cost = PortfolioTransactionCost(self.cost_calculator)
        
        # 状态
        self.capital = initial_capital
        self.positions: dict = {}  # {ts_code: weight}
        self.history: list = []
    
    def run(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        运行回测
        
        Args:
            predictions: 预测数据，包含 date, ts_code, predicted, excess_return
        
        Returns:
            回测结果
        """
        dates = sorted(predictions['date'].unique())
        
        for date in dates:
            daily_pred = predictions[predictions['date'] == date].copy()
            if daily_pred.empty:
                continue
            
            # 生成目标持仓
            daily_pred = daily_pred.sort_values('predicted', ascending=False)
            target_stocks = daily_pred.head(self.topk)['ts_code'].tolist()
            target_weights = {s: 1.0 / self.topk for s in target_stocks}
            
            # 计算换仓
            old_stocks = set(self.positions.keys())
            new_stocks = set(target_stocks)
            
            to_sell = old_stocks - new_stocks
            to_buy = new_stocks - old_stocks
            
            # 限制换仓数量
            to_sell = list(to_sell)[:self.n_drop]
            to_buy = list(to_buy)[:self.n_drop]
            
            # 计算实际持仓
            actual_stocks = [s for s in self.positions.keys() if s not in to_sell] + to_buy
            actual_weights = {s: 1.0 / len(actual_stocks) for s in actual_stocks}
            
            # 计算交易成本
            sell_amount = sum(self.positions.get(s, 0) for s in to_sell) * self.capital
            buy_amount = sum(target_weights.get(s, 0) for s in to_buy) * self.capital
            
            sell_cost = self.cost_calculator.calculate(sell_amount, is_open=False)['total_cost']
            buy_cost = self.cost_calculator.calculate(buy_amount, is_open=True)['total_cost']
            total_cost = sell_cost + buy_cost
            
            # 更新持仓
            self.positions = actual_weights
            
            # 计算当日收益
            daily_return = 0
            for stock in actual_stocks:
                stock_data = daily_pred[daily_pred['ts_code'] == stock]
                if not stock_data.empty:
                    weight = actual_weights[stock]
                    excess_return = stock_data['excess_return'].values[0]
                    daily_return += weight * excess_return
            
            # 扣除交易成本
            daily_return -= total_cost / self.capital
            
            # 更新资金
            self.capital *= (1 + daily_return)
            
            # 记录
            self.history.append({
                'date': date,
                'capital': self.capital,
                'daily_return': daily_return,
                'total_cost': total_cost,
                'cost_rate': total_cost / self.capital,
                'num_positions': len(actual_stocks),
                'num_sell': len(to_sell),
                'num_buy': len(to_buy),
            })
        
        return pd.DataFrame(self.history)


def run_backtest_with_cost():
    """运行带交易成本的回测"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    
    # 加载数据
    df = pd.read_parquet('memory/dataset/features_h10_full.parquet')
    df['date'] = pd.to_datetime(df['date'])
    
    # 特征工程
    feature_cols = [c for c in df.columns if c not in ['date','exit_date','ts_code',
        'stock_return','benchmark_return','excess_return','industry']]
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(0)
    
    # 添加交互特征
    base_features = ['mom_20', 'volatility_20', 'roe', 'dv_ratio', 'net_mf_ratio']
    for i in range(len(base_features)):
        for j in range(i+1, len(base_features)):
            f1, f2 = base_features[i], base_features[j]
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
    
    df['price_momentum'] = df['mom_5'] * df['mom_20']
    df['vol_momentum'] = df['volatility_20'] * df['mom_20']
    df['value_momentum'] = df['dv_ratio'] * df['mom_20']
    df['quality_momentum'] = df['roe'] * df['mom_20']
    df['flow_momentum'] = df['net_mf_ratio'] * df['mom_20']
    df['mom_accel'] = df['mom_5'] - df['mom_20']
    df['mom_decel'] = df['mom_20'] - df['mom_60']
    df['reversal_strength'] = df['reversal_score'] * df['volatility_20']
    df['quality_value'] = df['roe'] * df['dv_ratio']
    
    # 截面排名
    key = ['mom_20','volatility_20','roe','dv_ratio','net_mf_ratio','bp','ep','mom_5','mom_60']
    for c in key:
        if c in df.columns:
            df[f'{c}_r'] = df.groupby('date')[c].rank(pct=True)
    
    # 更新特征列表
    feature_cols = [c for c in df.columns if c not in ['date','exit_date','ts_code',
        'stock_return','benchmark_return','excess_return','industry']]
    for col in feature_cols:
        df[col] = df[col].fillna(0)
    feature_cols = [c for c in feature_cols if df[c].std() > 0]
    
    # 目标
    df['target_return'] = df['excess_return']
    df['ym'] = df['date'].dt.to_period('M')
    df = df.sort_values('date')
    months = sorted(df['ym'].unique())
    
    # 从2024开始测试
    start_idx = 0
    for idx, ym in enumerate(months):
        if str(ym) >= '2024-01':
            start_idx = idx
            break
    
    # 运行回测
    preds = []
    train_w = 12
    
    for i in range(start_idx, len(months)):
        te = i + train_w + 1
        if te >= len(months): break
        tr_df = df[df['ym'].isin(months[i:i+train_w])]
        te_df = df[df['ym'] == months[te]]
        if tr_df.empty or te_df.empty: continue
        
        # 特征选择
        ics = {}
        for c in feature_cols:
            ic = calc_ic(tr_df[c].values, tr_df['target_return'].values)
            if not np.isnan(ic):
                ics[c] = abs(ic)
        sel = sorted(ics, key=ics.get, reverse=True)[:30]
        
        # 召回
        rec = te_df[(te_df['roe']>0) & (te_df['net_mf_ratio']>0)]
        if rec.empty: continue
        
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr_df[sel].values)
        ytr = tr_df['target_return'].values
        Xte = sc.transform(rec[sel].values)
        
        m = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.05, num_leaves=15,
            max_depth=4, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
            verbose=-1, random_state=500)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        
        p = rec[['date','ts_code','excess_return']].copy()
        p['predicted'] = pred
        preds.append(p)
    
    predictions = pd.concat(preds)
    predictions['date'] = pd.to_datetime(predictions['date'])
    
    # 测试不同配置
    print('=== 带交易成本的回测 ===')
    print()
    
    configs = [
        {'topk': 10, 'n_drop': 5, 'name': 'Top10, 每日最多换5只'},
        {'topk': 10, 'n_drop': 3, 'name': 'Top10, 每日最多换3只'},
        {'topk': 10, 'n_drop': 10, 'name': 'Top10, 无换仓限制'},
        {'topk': 20, 'n_drop': 5, 'name': 'Top20, 每日最多换5只'},
        {'topk': 50, 'n_drop': 5, 'name': 'Top50, 每日最多换5只'},
    ]
    
    print(f'{"配置":<30} {"总收益":<12} {"年化收益":<12} {"最大回撤":<12} {"夏普":<8} {"换仓次数":<10} {"交易成本":<12}')
    print('-' * 100)
    
    for config in configs:
        engine = BacktestEngine(
            initial_capital=100000000,
            topk=config['topk'],
            n_drop=config['n_drop'],
        )
        
        result = engine.run(predictions)
        
        if result.empty:
            continue
        
        # 计算指标
        total_return = (result['capital'].iloc[-1] / result['capital'].iloc[0]) - 1
        days = (result['date'].iloc[-1] - result['date'].iloc[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # 最大回撤
        cumulative = result['capital'] / result['capital'].iloc[0]
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 夏普比率
        daily_returns = result['daily_return']
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # 换仓次数
        total_trades = result['num_sell'].sum() + result['num_buy'].sum()
        
        # 交易成本
        total_cost = result['total_cost'].sum()
        
        print(f'{config["name"]:<30} {total_return:<12.2%} {annual_return:<12.2%} {max_drawdown:<12.2%} {sharpe:<8.2f} {total_trades:<10} {total_cost:<12,.0f}')
    
    # 无交易成本的基准
    print()
    print('=== 无交易成本基准 ===')
    
    # 简单计算无成本收益
    daily_returns = []
    for date in sorted(predictions['date'].unique()):
        daily_pred = predictions[predictions['date'] == date]
        top10 = daily_pred.nlargest(10, 'predicted')
        daily_return = top10['excess_return'].mean()
        daily_returns.append(daily_return)
    
    no_cost_return = (1 + pd.Series(daily_returns)).prod() - 1
    print(f'无成本总收益: {no_cost_return:.2%}')


if __name__ == "__main__":
    run_backtest_with_cost()
