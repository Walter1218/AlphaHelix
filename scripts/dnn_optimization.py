"""
DNN 模型优化脚本

系统性测试不同配置，目标：胜率 55%+

测试维度：
1. 模型架构（隐藏层大小、残差连接、注意力机制）
2. 特征选择（不同特征数量、特征组合）
3. 损失函数（MSE、Huber、Ranking Loss）
4. 训练策略（学习率、批大小、正则化）
"""
import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")


class SimpleDNN(nn.Module):
    """简单 DNN"""
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        return self.model(x).squeeze()


class ResidualDNN(nn.Module):
    """带残差连接的 DNN"""
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.input_proj(x))
        residual = x
        x = self.res_block(x)
        x = x + residual  # 残差连接
        return self.output(x).squeeze()


class AttentionDNN(nn.Module):
    """带注意力机制的 DNN"""
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.output = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.input_proj(x))
        # 添加序列维度用于注意力
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = attn_out.squeeze(1)
        x = self.ffn(x)
        return self.output(x).squeeze()


class WideDeepDNN(nn.Module):
    """Wide & Deep 模型"""
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        # Wide 部分（线性）
        self.wide = nn.Linear(input_dim, 1)
        # Deep 部分
        self.deep = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        return (wide_out + deep_out).squeeze()


def load_data():
    """加载数据"""
    df = pd.read_parquet('memory/dataset/features_h10_full.parquet')
    df['date'] = pd.to_datetime(df['date'])
    
    feature_cols = [c for c in df.columns if c not in ['date', 'exit_date', 'ts_code', 
                                                          'stock_return', 'benchmark_return', 
                                                          'excess_return', 'industry']]
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # 添加交互特征
    base_features = ['mom_20', 'volatility_20', 'roe', 'dv_ratio', 'net_mf_ratio']
    for i in range(len(base_features)):
        for j in range(i + 1, len(base_features)):
            f1, f2 = base_features[i], base_features[j]
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
    
    feature_cols = [c for c in df.columns if c not in ['date', 'exit_date', 'ts_code', 
                                                          'stock_return', 'benchmark_return', 
                                                          'excess_return', 'industry']]
    
    df_sorted = df.sort_values('date')
    df_sorted['ym'] = df_sorted['date'].dt.to_period('M')
    months = sorted(df_sorted['ym'].unique())
    
    # 计算特征 IC
    ics = {}
    for col in feature_cols:
        ic = df_sorted.groupby('ym').apply(
            lambda g: g[col].corr(g['excess_return'], method='spearman'), 
            include_groups=False
        ).mean()
        ics[col] = ic
    
    sorted_ics = sorted(ics.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return df_sorted, feature_cols, sorted_ics, months


def run_dnn(df_sorted, selected, months, model_class, hidden_dim=32, 
            lr=0.001, epochs=15, batch_size=1024, train_w=9):
    """运行 DNN 模型"""
    preds = []
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
        
        train_df = df_sorted[df_sorted['ym'].isin(train_ms)]
        val_df = df_sorted[df_sorted['ym'].isin(val_ms)]
        test_df = df_sorted[df_sorted['ym'] == test_month]
        
        if train_df.empty or val_df.empty or test_df.empty:
            continue
        
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        X_tr = scaler.fit_transform(train_df[selected].values)
        y_tr = train_df['excess_return'].values
        X_val = scaler.transform(val_df[selected].values)
        y_val = val_df['excess_return'].values
        X_test = scaler.transform(test_df[selected].values)
        
        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.FloatTensor(y_tr)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        X_test_t = torch.FloatTensor(X_test)
        
        train_dataset = TensorDataset(X_tr_t, y_tr_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        model = model_class(len(selected), hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.HuberLoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            pred_values = model(X_test_t).numpy()
        
        p = test_df[['date', 'ts_code', 'stock_return', 'benchmark_return', 
                      'excess_return', 'industry']].copy()
        p['predicted'] = pred_values
        preds.append(p)
    
    if not preds:
        return None
    return pd.concat(preds)


def analyze(r, name):
    """分析结果"""
    r = r.copy()
    r['date'] = pd.to_datetime(r['date'])
    r['rank'] = r.groupby('date')['predicted'].rank(ascending=False)
    
    win_rates = []
    for rank in range(1, 11):
        g = r[r['rank'] == rank]
        wr = (g['excess_return'] > 0).mean() if len(g) > 0 else 0
        win_rates.append(wr)
    
    corr, _ = stats.spearmanr(range(1, 11), win_rates)
    t10 = r[r['rank'] <= 10]
    wr_all = (t10['excess_return'] > 0).mean() if len(t10) > 0 else 0
    ce_all = t10.groupby('date')['excess_return'].mean().sum() if len(t10) > 0 else 0
    
    return {
        'name': name,
        'top1': win_rates[0],
        'top10': wr_all,
        'mono': corr,
        'ce': ce_all,
    }


def main():
    print("=== DNN 模型优化 ===\n")
    
    df_sorted, feature_cols, sorted_ics, months = load_data()
    print(f"数据量: {len(df_sorted)} 行, {len(months)} 个月")
    print(f"总特征数: {len(feature_cols)}\n")
    
    results = []
    
    # 测试 1: 不同模型架构
    print("=== 测试 1: 模型架构 ===")
    selected = [col for col, _ in sorted_ics[:40]]
    
    configs = [
        ('SimpleDNN h=32', SimpleDNN, 32),
        ('SimpleDNN h=64', SimpleDNN, 64),
        ('ResidualDNN h=32', ResidualDNN, 32),
        ('ResidualDNN h=64', ResidualDNN, 64),
        ('AttentionDNN h=32', AttentionDNN, 32),
        ('WideDeepDNN h=32', WideDeepDNN, 32),
        ('WideDeepDNN h=64', WideDeepDNN, 64),
    ]
    
    for name, model_class, hidden_dim in configs:
        r = run_dnn(df_sorted, selected, months, model_class, hidden_dim=hidden_dim)
        if r is not None:
            result = analyze(r, name)
            results.append(result)
            print(f"  {name}: Top-1={result['top1']:.1%}, Top-10={result['top10']:.1%}, mono={result['mono']:.4f}")
    
    # 测试 2: 不同特征数量
    print("\n=== 测试 2: 特征数量 ===")
    best_model = SimpleDNN
    best_hidden = 32
    
    for n_feat in [20, 30, 40, 50, 60]:
        selected = [col for col, _ in sorted_ics[:n_feat]]
        r = run_dnn(df_sorted, selected, months, best_model, hidden_dim=best_hidden)
        if r is not None:
            result = analyze(r, f'Features={n_feat}')
            results.append(result)
            print(f"  Top-{n_feat}: Top-1={result['top1']:.1%}, Top-10={result['top10']:.1%}, mono={result['mono']:.4f}")
    
    # 测试 3: 不同学习率
    print("\n=== 测试 3: 学习率 ===")
    selected = [col for col, _ in sorted_ics[:40]]
    
    for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
        r = run_dnn(df_sorted, selected, months, SimpleDNN, hidden_dim=32, lr=lr)
        if r is not None:
            result = analyze(r, f'LR={lr}')
            results.append(result)
            print(f"  LR={lr}: Top-1={result['top1']:.1%}, Top-10={result['top10']:.1%}, mono={result['mono']:.4f}")
    
    # 测试 4: 不同训练窗口
    print("\n=== 测试 4: 训练窗口 ===")
    
    for train_w in [6, 9, 12, 15]:
        r = run_dnn(df_sorted, selected, months, SimpleDNN, hidden_dim=32, train_w=train_w)
        if r is not None:
            result = analyze(r, f'Window={train_w}')
            results.append(result)
            print(f"  Window={train_w}: Top-1={result['top1']:.1%}, Top-10={result['top10']:.1%}, mono={result['mono']:.4f}")
    
    # 输出最优结果
    print("\n=== 最优结果 ===")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('top1', ascending=False)
    
    print(f"{'配置':<25} {'Top-1':<10} {'Top-10':<10} {'单调性':<10} {'累计超额':<12}")
    print("-" * 67)
    for _, row in results_df.head(10).iterrows():
        print(f"{row['name']:<25} {row['top1']:<10.1%} {row['top10']:<10.1%} {row['mono']:<10.4f} {row['ce']:<12.2%}")
    
    # 保存最优结果
    best = results_df.iloc[0]
    print(f"\n最优配置: {best['name']}")
    print(f"Top-1 胜率: {best['top1']:.1%}")
    print(f"Top-10 胜率: {best['top10']:.1%}")
    print(f"单调性: {best['mono']:.4f}")
    print(f"累计超额: {best['ce']:.2%}")


if __name__ == "__main__":
    main()
