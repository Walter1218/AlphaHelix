"""
DoubleEnsemble 模型

参考 Microsoft Qlib 的 DoubleEnsemble 模型。
使用两阶段集成学习提升预测精度。

用法：
    python double_ensemble.py
"""
import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")


def calc_ic(pred, actual):
    """计算 IC"""
    if len(pred) < 10:
        return 0
    ic, _ = spearmanr(pred, actual)
    return ic if not np.isnan(ic) else 0


class DoubleEnsemble:
    """
    DoubleEnsemble 模型
    
    两阶段集成学习：
    1. 训练基础模型（LightGBM）
    2. 计算残差
    3. 训练残差模型（LightGBM）
    4. 组合预测
    """
    
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        max_depth: int = 4,
        min_child_samples: int = 50,
        reg_alpha: float = 1.0,
        reg_lambda: float = 1.0,
        random_state: int = 500,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        
        self.model1 = None
        self.model2 = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练 DoubleEnsemble 模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
        """
        # 第一阶段：训练基础模型
        self.model1 = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbose=-1,
        )
        self.model1.fit(X, y)
        
        # 计算残差
        pred1 = self.model1.predict(X)
        residuals = y - pred1
        
        # 第二阶段：训练残差模型
        self.model2 = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state + 1,
            verbose=-1,
        )
        self.model2.fit(X, residuals)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征矩阵
        
        Returns:
            预测值
        """
        pred1 = self.model1.predict(X)
        pred2 = self.model2.predict(X)
        return pred1 + pred2
    
    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        imp1 = self.model1.feature_importances_
        imp2 = self.model2.feature_importances_
        return (imp1 + imp2) / 2


def run_double_ensemble_experiment():
    """运行 DoubleEnsemble 实验"""
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
    
    # 运行实验
    print('=== DoubleEnsemble 实验 ===')
    print()
    
    # 测试不同配置
    configs = [
        {'n_estimators': 50, 'name': 'DoubleEnsemble (ne=50)'},
        {'n_estimators': 100, 'name': 'DoubleEnsemble (ne=100)'},
        {'n_estimators': 200, 'name': 'DoubleEnsemble (ne=200)'},
    ]
    
    print(f'{"模型":<30} {"IC":<10} {"MeanIC":<10} {"ICIR":<10} {"胜率":<10} {"总收益":<12}')
    print('-' * 85)
    
    for config in configs:
        preds = []
        all_ics = []
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
            
            # 训练 DoubleEnsemble
            model = DoubleEnsemble(
                n_estimators=config['n_estimators'],
                learning_rate=0.05,
                num_leaves=15,
                max_depth=4,
                min_child_samples=50,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=500,
            )
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            
            # 计算 IC
            month_ic = calc_ic(pred, rec['excess_return'].values)
            all_ics.append(month_ic)
            
            p = rec[['date','ts_code','excess_return']].copy()
            p['predicted'] = pred
            preds.append(p)
        
        if not preds:
            continue
        
        r = pd.concat(preds)
        
        # 计算指标
        overall_ic = calc_ic(r['predicted'].values, r['excess_return'].values)
        mean_ic = np.mean(all_ics)
        icir = mean_ic / (np.std(all_ics) + 1e-6)
        
        # 计算胜率和收益
        daily_returns = []
        daily_wins = []
        
        for date in sorted(r['date'].unique()):
            daily_pred = r[r['date'] == date]
            top10 = daily_pred.nlargest(10, 'predicted')
            daily_return = top10['excess_return'].mean()
            daily_returns.append(daily_return)
            daily_wins.append(1 if daily_return > 0 else 0)
        
        total_return = (1 + pd.Series(daily_returns)).prod() - 1
        win_rate = np.mean(daily_wins)
        
        print(f'{config["name"]:<30} {overall_ic:<10.4f} {mean_ic:<10.4f} {icir:<10.2f} {win_rate:<10.1%} {total_return:<12.2%}')
    
    # 对比 LightGBM
    print()
    print('=== 对比 LightGBM ===')
    print()
    
    preds = []
    all_ics = []
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
        
        # 训练 LightGBM
        model = lgb.LGBMRegressor(
            n_estimators=50,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=50,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=500,
            verbose=-1,
        )
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        
        # 计算 IC
        month_ic = calc_ic(pred, rec['excess_return'].values)
        all_ics.append(month_ic)
        
        p = rec[['date','ts_code','excess_return']].copy()
        p['predicted'] = pred
        preds.append(p)
    
    r = pd.concat(preds)
    
    # 计算指标
    overall_ic = calc_ic(r['predicted'].values, r['excess_return'].values)
    mean_ic = np.mean(all_ics)
    icir = mean_ic / (np.std(all_ics) + 1e-6)
    
    # 计算胜率和收益
    daily_returns = []
    daily_wins = []
    
    for date in sorted(r['date'].unique()):
        daily_pred = r[r['date'] == date]
        top10 = daily_pred.nlargest(10, 'predicted')
        daily_return = top10['excess_return'].mean()
        daily_returns.append(daily_return)
        daily_wins.append(1 if daily_return > 0 else 0)
    
    total_return = (1 + pd.Series(daily_returns)).prod() - 1
    win_rate = np.mean(daily_wins)
    
    print(f'{"LightGBM":<30} {overall_ic:<10.4f} {mean_ic:<10.4f} {icir:<10.2f} {win_rate:<10.1%} {total_return:<12.2%}')


if __name__ == "__main__":
    run_double_ensemble_experiment()
