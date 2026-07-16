"""
Walk-Forward 验证框架

基于 ML4T 方法论，实现严格的 walk-forward 验证：
1. 证据边界：严格分离调优期和评估期
2. Purge gap：防止训练/测试数据泄漏
3. 特征选择：在每期内用训练数据完成
4. 模型训练：只用训练数据
5. 评估：只用测试数据

用法：
    python walkforward.py --start 2023 --train-w 6 --purge 1
"""
import sys
import os
import json
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")


def calc_ic(pred, actual):
    """计算 IC (Spearman 相关系数)"""
    if len(pred) < 10:
        return 0
    ic, _ = spearmanr(pred, actual)
    return ic if not np.isnan(ic) else 0


def calc_metrics(predictions: pd.DataFrame) -> Dict:
    """
    计算评估指标
    
    Args:
        predictions: 包含 date, ts_code, excess_return, pred 列
    
    Returns:
        指数字典
    """
    # IC
    ic = calc_ic(predictions['pred'].values, predictions['excess_return'].values)
    
    # 每日 IC
    daily_ics = []
    for date in predictions['date'].unique():
        daily = predictions[predictions['date'] == date]
        if len(daily) >= 10:
            daily_ic = calc_ic(daily['pred'].values, daily['excess_return'].values)
            daily_ics.append(daily_ic)
    
    mean_ic = np.mean(daily_ics) if daily_ics else 0
    icir = mean_ic / (np.std(daily_ics) + 1e-6) if daily_ics else 0
    
    # Top-10 胜率和收益
    daily_returns = []
    for date in predictions['date'].unique():
        daily = predictions[predictions['date'] == date]
        top10 = daily.nlargest(10, 'pred')
        daily_returns.append(top10['excess_return'].mean())
    
    win_rate = np.mean([1 if x > 0 else 0 for x in daily_returns])
    total_return = (1 + pd.Series(daily_returns)).prod() - 1
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)
    
    return {
        'ic': ic,
        'mean_ic': mean_ic,
        'icir': icir,
        'win_rate': win_rate,
        'total_return': total_return,
        'sharpe': sharpe,
        'days': len(daily_returns),
    }


class WalkForwardValidator:
    """
    Walk-Forward 验证器
    
    实现严格的 walk-forward 验证，防止数据泄漏。
    """
    
    def __init__(
        self,
        train_window: int = 6,  # 训练窗口（月）
        purge_gap: int = 1,  # Purge gap（月）
        test_window: int = 1,  # 测试窗口（月）
        n_features: int = 20,  # 特征数量
        model_type: str = 'ridge',  # 模型类型
        alpha: float = 1.0,  # 正则化参数
        use_5d_target: bool = True,  # 使用5天均值目标
    ):
        self.train_window = train_window
        self.purge_gap = purge_gap
        self.test_window = test_window
        self.n_features = n_features
        self.model_type = model_type
        self.alpha = alpha
        self.use_5d_target = use_5d_target
    
    def select_features(self, train_df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
        """
        在训练集上选择特征（用 IC）
        
        Args:
            train_df: 训练数据
            feature_cols: 所有特征列表
        
        Returns:
            选中的特征列表
        """
        # 计算目标
        if self.use_5d_target:
            target = train_df.groupby('ts_code')['excess_return'].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
        else:
            target = train_df['excess_return']
        
        # 计算 IC
        ics = {}
        for col in feature_cols:
            ic = calc_ic(train_df[col].values, target.values)
            if not np.isnan(ic) and ic > 0:
                ics[col] = ic
        
        # 选择正 IC 特征
        selected = sorted(ics, key=ics.get, reverse=True)[:self.n_features]
        return selected
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
        
        Returns:
            训练好的模型
        """
        if self.model_type == 'ridge':
            model = Ridge(alpha=self.alpha)
            model.fit(X_train, y_train)
            return model
        elif self.model_type == 'lgb':
            model = lgb.LGBMRegressor(
                n_estimators=50,
                num_leaves=15,
                max_depth=4,
                min_child_samples=50,
                reg_alpha=self.alpha,
                reg_lambda=self.alpha,
                verbose=-1,
                random_state=500,
            )
            model.fit(X_train, y_train)
            return model
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def run(self, df: pd.DataFrame, feature_cols: List[str], 
            start_year: int = 2023, use_recall: bool = True) -> Dict:
        """
        运行 walk-forward 验证
        
        Args:
            df: 数据
            feature_cols: 特征列表
            start_year: 评估起始年
            use_recall: 是否使用质量因子召回
        
        Returns:
            评估结果
        """
        df['ym'] = df['date'].dt.to_period('M')
        months = sorted(df['ym'].unique())
        
        # 找到起始月份
        start_idx = 0
        for idx, ym in enumerate(months):
            if str(ym) >= f'{start_year}-01':
                start_idx = idx
                break
        
        all_predictions = []
        fold_metrics = []
        
        for i in range(start_idx, len(months)):
            # 训练期
            train_end = i
            if train_end < self.train_window:
                continue
            train_start = train_end - self.train_window
            train_months = months[train_start:train_end]
            
            # Purge gap
            purge_start = train_end
            purge_end = purge_start + self.purge_gap
            if purge_end >= len(months):
                continue
            
            # 测试期
            test_start = purge_end
            test_end = test_start + self.test_window
            if test_end > len(months):
                continue
            test_month = months[test_start]
            
            # 获取数据
            train_df = df[df['ym'].isin(train_months)]
            test_df = df[df['ym'] == test_month]
            
            if train_df.empty or test_df.empty:
                continue
            
            # 特征选择（只用训练数据）
            selected = self.select_features(train_df, feature_cols)
            if len(selected) < 5:
                continue
            
            # 召回
            if use_recall:
                test_recalled = test_df[(test_df['roe'] > 0) & (test_df['net_mf_ratio'] > 0)]
            else:
                test_recalled = test_df
            
            if test_recalled.empty:
                continue
            
            # 训练模型
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_df[selected].values)
            y_train = train_df['excess_return'].values
            X_test = scaler.transform(test_recalled[selected].values)
            
            model = self.train_model(X_train, y_train)
            predictions = model.predict(X_test)
            
            # 记录预测
            pred_df = test_recalled[['date', 'ts_code', 'excess_return']].copy()
            pred_df['pred'] = predictions
            all_predictions.append(pred_df)
            
            # 计算 fold 指标
            fold_metrics.append({
                'fold': i,
                'train_months': [str(m) for m in train_months],
                'test_month': str(test_month),
                'n_features': len(selected),
                'n_test': len(test_recalled),
            })
        
        if not all_predictions:
            return None
        
        # 合并所有预测
        all_pred = pd.concat(all_predictions)
        
        # 计算整体指标
        metrics = calc_metrics(all_pred)
        metrics['folds'] = fold_metrics
        metrics['n_folds'] = len(fold_metrics)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward 验证")
    parser.add_argument("--start", type=int, default=2023, help="评估起始年")
    parser.add_argument("--train-w", type=int, default=6, help="训练窗口（月）")
    parser.add_argument("--purge", type=int, default=1, help="Purge gap（月）")
    parser.add_argument("--n-features", type=int, default=20, help="特征数量")
    parser.add_argument("--model", type=str, default='ridge', choices=['ridge', 'lgb'], help="模型类型")
    parser.add_argument("--alpha", type=float, default=1.0, help="正则化参数")
    parser.add_argument("--no-5d", action="store_true", help="不使用5天均值目标")
    parser.add_argument("--no-recall", action="store_true", help="不使用质量因子召回")
    args = parser.parse_args()
    
    # 加载数据
    print("加载数据...")
    df = pd.read_parquet('memory/dataset/features_h10_full.parquet')
    df['date'] = pd.to_datetime(df['date'])
    
    # 特征工程
    print("特征工程...")
    base = ['mom_20', 'volatility_20', 'roe', 'dv_ratio', 'net_mf_ratio']
    for i in range(len(base)):
        for j in range(i+1, len(base)):
            df[f'{base[i]}x{base[j]}'] = df[base[i]] * df[base[j]]
    
    feature_cols = [c for c in df.columns if c not in ['date', 'exit_date', 'ts_code',
        'stock_return', 'benchmark_return', 'excess_return', 'industry']]
    for col in feature_cols:
        df[col] = df[col].fillna(0)
    feature_cols = [c for c in feature_cols if df[c].std() > 0]
    
    print(f"特征数: {len(feature_cols)}")
    
    # 运行验证
    print("\n运行 Walk-Forward 验证...")
    validator = WalkForwardValidator(
        train_window=args.train_w,
        purge_gap=args.purge,
        n_features=args.n_features,
        model_type=args.model,
        alpha=args.alpha,
        use_5d_target=not args.no_5d,
    )
    
    results = validator.run(
        df, 
        feature_cols, 
        start_year=args.start,
        use_recall=not args.no_recall,
    )
    
    if results:
        print("\n=== 结果 ===")
        print(f"IC: {results['ic']:.4f}")
        print(f"MeanIC: {results['mean_ic']:.4f}")
        print(f"ICIR: {results['icir']:.2f}")
        print(f"胜率: {results['win_rate']:.1%}")
        print(f"收益: {results['total_return']:.2%}")
        print(f"夏普: {results['sharpe']:.2f}")
        print(f"天数: {results['days']}")
        print(f"Folds: {results['n_folds']}")
        
        # 保存结果
        os.makedirs('memory/eval', exist_ok=True)
        result_path = f'memory/eval/walkforward_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n结果已保存: {result_path}")
    else:
        print("验证失败")


if __name__ == "__main__":
    main()
