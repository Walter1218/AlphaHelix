"""
超参数调优脚本

测试不同的 LightGBM 超参数组合，找到最优配置。
"""
import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")


def train_and_evaluate(X_train, y_train, X_val, y_val, params):
    """训练模型并评估性能。"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    
    # 预测
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    # 计算 IC (Information Coefficient)
    ic = np.corrcoef(y_pred, y_val)[0, 1]
    
    # 计算胜率（top-10）
    val_df = pd.DataFrame({'predicted': y_pred, 'actual': y_val})
    val_df['rank'] = val_df['predicted'].rank(ascending=False)
    top10 = val_df[val_df['rank'] <= 10]
    win_rate = (top10['actual'] > 0).mean() if len(top10) > 0 else 0
    
    return {
        'ic': ic,
        'win_rate': win_rate,
        'best_iteration': model.best_iteration,
        'model': model,
    }


def main():
    # 加载数据集
    df = pd.read_parquet('memory/dataset/features_h10_optimized.parquet')
    
    # 获取特征列
    feature_cols = [col for col in df.columns if col not in ['date', 'exit_date', 'ts_code', 'stock_return', 'benchmark_return', 'excess_return', 'industry']]
    
    # 按时间划分训练集和验证集
    df['date'] = pd.to_datetime(df['date'])
    split_date = df['date'].quantile(0.8)
    
    train_df = df[df['date'] < split_date]
    val_df = df[df['date'] >= split_date]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['excess_return'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['excess_return'].values
    
    print(f'训练集: {len(train_df)} 样本')
    print(f'验证集: {len(val_df)} 样本')
    print(f'特征数: {len(feature_cols)}')
    
    # 定义超参数搜索空间
    param_grid = [
        # 基线
        {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'lambda_l1': 0,
            'lambda_l2': 0,
            'verbose': -1,
            'seed': 42,
        },
        # 更多叶子
        {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'lambda_l1': 0,
            'lambda_l2': 0,
            'verbose': -1,
            'seed': 42,
        },
        # 更低学习率
        {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'lambda_l1': 0,
            'lambda_l2': 0,
            'verbose': -1,
            'seed': 42,
        },
        # L1 正则化
        {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0,
            'verbose': -1,
            'seed': 42,
        },
        # L2 正则化
        {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'lambda_l1': 0,
            'lambda_l2': 0.1,
            'verbose': -1,
            'seed': 42,
        },
        # DART boosting
        {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting': 'dart',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'lambda_l1': 0,
            'lambda_l2': 0,
            'verbose': -1,
            'seed': 42,
        },
    ]
    
    print('\\n=== 超参数调优 ===')
    results = []
    for i, params in enumerate(param_grid):
        print(f'\\n测试配置 {i+1}/{len(param_grid)}...')
        result = train_and_evaluate(X_train, y_train, X_val, y_val, params)
        results.append(result)
        print(f'  IC: {result["ic"]:.4f}')
        print(f'  Top-10 胜率: {result["win_rate"]:.1%}')
        print(f'  最佳迭代: {result["best_iteration"]}')
    
    # 找到最优配置
    best_idx = max(range(len(results)), key=lambda i: results[i]['ic'])
    best_result = results[best_idx]
    best_params = param_grid[best_idx]
    
    print(f'\\n=== 最优配置 ===')
    print(f'配置 {best_idx + 1}:')
    print(f'  IC: {best_result["ic"]:.4f}')
    print(f'  Top-10 胜率: {best_result["win_rate"]:.1%}')
    print(f'  最佳迭代: {best_result["best_iteration"]}')
    print(f'  参数: {best_params}')
    
    # 保存最优模型
    best_model = best_result['model']
    best_model.save_model('memory/models/gbdt_h10_optimized.lightgbm.txt')
    print(f'\\n保存最优模型到: memory/models/gbdt_h10_optimized.lightgbm.txt')


if __name__ == '__main__':
    main()
