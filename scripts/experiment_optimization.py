"""
综合优化实验

在实验链路验证以下优化方向：
1. Alpha158 特征
2. 超参数调优
3. 动态召回
4. 多预测周期

不集成主链路，仅验证效果。
"""
import sys
import os
import warnings
from datetime import datetime

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


def load_data():
    """加载数据并进行特征工程"""
    df = pd.read_parquet('memory/dataset/features_h10_full.parquet')
    df['date'] = pd.to_datetime(df['date'])
    
    # 基础特征
    feature_cols = [c for c in df.columns if c not in ['date','exit_date','ts_code',
        'stock_return','benchmark_return','excess_return','industry']]
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(0)
    
    # 交互特征
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
    
    # Alpha158 风格特征
    key_features = ['mom_5', 'mom_20', 'mom_60', 'mom_120', 'volatility_20']
    for feat in key_features:
        if feat in df.columns:
            for d in [5, 10, 20]:
                df[f'{feat}_MA{d}'] = df[feat].rolling(d, min_periods=1).mean()
                df[f'{feat}_STD{d}'] = df[feat].rolling(d, min_periods=1).std()
                df[f'{feat}_RSV{d}'] = (df[feat] - df[feat].rolling(d, min_periods=1).min()) / \
                    (df[feat].rolling(d, min_periods=1).max() - df[feat].rolling(d, min_periods=1).min() + 1e-6)
    
    # 截面排名
    key = ['mom_20','volatility_20','roe','dv_ratio','net_mf_ratio','bp','ep','mom_5','mom_60']
    for c in key:
        if c in df.columns:
            df[f'{c}_rank'] = df.groupby('date')[c].rank(pct=True)
    
    # 更新特征列表
    all_feature_cols = [c for c in df.columns if c not in ['date','exit_date','ts_code',
        'stock_return','benchmark_return','excess_return','industry']]
    for col in all_feature_cols:
        df[col] = df[col].fillna(0)
    all_feature_cols = [c for c in all_feature_cols if df[c].std() > 0]
    
    return df, all_feature_cols


def select_features(df, feature_cols, n=30, method='ic'):
    """特征选择"""
    if method == 'ic':
        ics = {}
        for c in feature_cols:
            ic = calc_ic(df[c].values, df['excess_return'].values)
            if not np.isnan(ic):
                ics[c] = abs(ic)
        return sorted(ics, key=ics.get, reverse=True)[:n]
    else:
        return feature_cols[:n]


def run_experiment(df, feature_cols, config):
    """运行单个实验"""
    start_idx = 0
    for idx, ym in enumerate(sorted(df['ym'].unique())):
        if str(ym) >= '2024-01':
            start_idx = idx
            break
    
    months = sorted(df['ym'].unique())
    preds = []
    all_ics = []
    
    for i in range(start_idx, len(months)):
        train_end = i + config['train_w']
        test_idx = train_end + 1
        if test_idx >= len(months):
            break
        
        train_ms = months[i:train_end]
        test_month = months[test_idx]
        train_df = df[df['ym'].isin(train_ms)]
        test_df = df[df['ym'] == test_month]
        
        if train_df.empty or test_df.empty:
            continue
        
        # 特征选择
        selected = select_features(train_df, feature_cols, config['n_features'])
        
        # 召回
        if config['recall'] == 'quality':
            rec = test_df[(test_df['roe'] > 0) & (test_df['net_mf_ratio'] > 0)]
        elif config['recall'] == 'momentum':
            rec = test_df[test_df['mom_20'] > 0]
        elif config['recall'] == 'value':
            rec = test_df[(test_df['dv_ratio'] > 0) & (test_df['bp'] > 0)]
        elif config['recall'] == 'all':
            rec = test_df
        else:
            rec = test_df[(test_df['roe'] > 0) & (test_df['net_mf_ratio'] > 0)]
        
        if rec.empty:
            continue
        
        # 训练
        sc = StandardScaler()
        X_tr = sc.fit_transform(train_df[selected].values)
        y_tr = train_df[config['target']].values
        X_test = sc.transform(rec[selected].values)
        
        # DoubleEnsemble
        model1 = lgb.LGBMRegressor(
            n_estimators=config.get('n_estimators', 100),
            learning_rate=config.get('learning_rate', 0.05),
            num_leaves=config.get('num_leaves', 15),
            max_depth=config.get('max_depth', 4),
            min_child_samples=50,
            reg_alpha=config.get('reg_alpha', 1.0),
            reg_lambda=config.get('reg_lambda', 1.0),
            random_state=500,
            verbose=-1
        )
        model1.fit(X_tr, y_tr)
        
        residuals = y_tr - model1.predict(X_tr)
        model2 = lgb.LGBMRegressor(
            n_estimators=config.get('n_estimators', 100),
            learning_rate=config.get('learning_rate', 0.05),
            num_leaves=config.get('num_leaves', 15),
            max_depth=config.get('max_depth', 4),
            min_child_samples=50,
            reg_alpha=config.get('reg_alpha', 1.0),
            reg_lambda=config.get('reg_lambda', 1.0),
            random_state=501,
            verbose=-1
        )
        model2.fit(X_tr, residuals)
        
        pred = model1.predict(X_test) + model2.predict(X_test)
        
        month_ic = calc_ic(pred, rec['excess_return'].values)
        all_ics.append(month_ic)
        
        p = rec[['date','ts_code','excess_return']].copy()
        p['predicted'] = pred
        preds.append(p)
    
    if not preds:
        return None
    
    r = pd.concat(preds)
    overall_ic = calc_ic(r['predicted'].values, r['excess_return'].values)
    mean_ic = np.mean(all_ics)
    icir = mean_ic / (np.std(all_ics) + 1e-6)
    
    # 胜率
    daily_returns = []
    for date in sorted(r['date'].unique()):
        daily_pred = r[r['date'] == date]
        top10 = daily_pred.nlargest(10, 'predicted')
        daily_returns.append(top10['excess_return'].mean())
    
    win_rate = np.mean([1 if x > 0 else 0 for x in daily_returns])
    total_return = (1 + pd.Series(daily_returns)).prod() - 1
    
    return {
        'ic': overall_ic,
        'mean_ic': mean_ic,
        'icir': icir,
        'win_rate': win_rate,
        'total_return': total_return,
    }


def main():
    print("=" * 70)
    print("综合优化实验")
    print("=" * 70)
    print()
    
    # 加载数据
    df, all_feature_cols = load_data()
    df['ym'] = df['date'].dt.to_period('M')
    
    # 基准配置
    baseline = {
        'train_w': 6,
        'n_features': 30,
        'target': 'excess_return',
        'recall': 'quality',
        'n_estimators': 100,
        'learning_rate': 0.05,
        'num_leaves': 15,
        'max_depth': 4,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
    }
    
    # 实验1: Alpha158 特征
    print("=" * 70)
    print("实验1: Alpha158 特征")
    print("=" * 70)
    
    alpha158_features = [c for c in all_feature_cols if any(x in c for x in ['_MA', '_STD', '_RSV'])]
    original_features = [c for c in all_feature_cols if c not in alpha158_features]
    
    for feat_set, name in [
        (original_features, '原始特征'),
        (alpha158_features, 'Alpha158特征'),
        (all_feature_cols, '全部特征'),
    ]:
        config = baseline.copy()
        result = run_experiment(df, feat_set, config)
        if result:
            print(f"{name:15s}: IC={result['ic']:.4f}, MeanIC={result['mean_ic']:.4f}, ICIR={result['icir']:.2f}, 胜率={result['win_rate']:.1%}, 收益={result['total_return']:.2%}")
    
    print()
    
    # 实验2: 超参数调优
    print("=" * 70)
    print("实验2: 超参数调优")
    print("=" * 70)
    
    for n_est in [50, 100, 200]:
        for lr in [0.01, 0.05, 0.1]:
            config = baseline.copy()
            config['n_estimators'] = n_est
            config['learning_rate'] = lr
            result = run_experiment(df, all_feature_cols, config)
            if result:
                print(f"ne={n_est:3d} lr={lr:.2f}: IC={result['ic']:.4f}, 胜率={result['win_rate']:.1%}, 收益={result['total_return']:.2%}")
    
    print()
    
    # 实验3: 动态召回
    print("=" * 70)
    print("实验3: 动态召回")
    print("=" * 70)
    
    for recall, name in [
        ('quality', '质量因子'),
        ('momentum', '动量因子'),
        ('value', '价值因子'),
        ('all', '无召回'),
    ]:
        config = baseline.copy()
        config['recall'] = recall
        result = run_experiment(df, all_feature_cols, config)
        if result:
            print(f"{name:10s}: IC={result['ic']:.4f}, 胜率={result['win_rate']:.1%}, 收益={result['total_return']:.2%}")
    
    print()
    
    # 实验4: 多预测周期
    print("=" * 70)
    print("实验4: 多预测周期")
    print("=" * 70)
    
    for target, name in [
        ('excess_return', '原始收益'),
    ]:
        # 测试不同训练窗口
        for train_w in [3, 6, 9, 12]:
            config = baseline.copy()
            config['train_w'] = train_w
            config['target'] = target
            result = run_experiment(df, all_feature_cols, config)
            if result:
                print(f"train_w={train_w:2d}: IC={result['ic']:.4f}, 胜率={result['win_rate']:.1%}, 收益={result['total_return']:.2%}")
    
    print()
    print("=" * 70)
    print("实验完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
