"""
AlphaHelix Online Learning

实现在线学习，让模型持续适应市场变化。

方法：
1. 用历史数据训练基础模型
2. 每月用新数据增量更新模型
3. 保留模型快照，支持回测

用法：
    python scripts/online_learning.py --dataset memory/dataset/features_h10_selected_v2.parquet --horizon 10
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_trainer import load_dataset, get_feature_cols

PRED_DIR = Path("memory/predictions")
MODEL_DIR = Path("memory/models")


class OnlineGBDT:
    """在线学习的 GBDT 模型"""
    
    def __init__(self, feature_cols, learning_rate=0.01, num_leaves=31):
        self.feature_cols = feature_cols
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.model = None
        self.train_count = 0
    
    def _get_params(self):
        return {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "seed": 42,
        }
    
    def train(self, X, y, num_boost_round=100):
        """训练或增量更新模型"""
        dtrain = lgb.Dataset(X, label=y, feature_name=self.feature_cols)
        
        if self.model is None:
            # 首次训练
            self.model = lgb.train(
                self._get_params(), 
                dtrain, 
                num_boost_round=num_boost_round
            )
        else:
            # 增量更新
            self.model = lgb.train(
                self._get_params(),
                dtrain,
                num_boost_round=num_boost_round // 2,  # 更新时用更少的轮数
                init_model=self.model
            )
        
        self.train_count += 1
    
    def predict(self, X):
        if self.model is None:
            return np.zeros(X.shape[0])
        return self.model.predict(X)
    
    def save(self, path):
        if self.model is not None:
            self.model.save_model(path)
    
    def load(self, path):
        self.model = lgb.Booster(model_file=path)


def walk_forward_online(df, feature_cols, train_window_months=18, update_months=1):
    """Walk-forward 在线学习"""
    df = df.sort_values("date").copy()
    df["year_month"] = df["date"].dt.to_period("M")
    months = sorted(df["year_month"].unique())
    
    # 初始化在线模型
    online_model = OnlineGBDT(feature_cols)
    
    all_preds = []
    
    for i, test_month in enumerate(months):
        train_months = months[max(0, i - train_window_months):i]
        if len(train_months) < 3:
            continue
        
        # 训练数据
        train_df = df[df["year_month"].isin(train_months)]
        test_df = df[df["year_month"] == test_month]
        
        if train_df.empty or test_df.empty:
            continue
        
        X_tr = train_df[feature_cols].fillna(0).values
        y_tr = train_df["excess_return"].values
        X_test = test_df[feature_cols].fillna(0).values
        
        # 训练或更新模型
        if online_model.train_count == 0:
            # 首次训练：用完整训练集
            online_model.train(X_tr, y_tr, num_boost_round=200)
        else:
            # 增量更新：只用最近的数据
            recent_months = train_months[-update_months:]
            recent_df = df[df["year_month"].isin(recent_months)]
            X_recent = recent_df[feature_cols].fillna(0).values
            y_recent = recent_df["excess_return"].values
            online_model.train(X_recent, y_recent, num_boost_round=50)
        
        # 预测
        preds = online_model.predict(X_test)
        
        pred = test_df[["date", "ts_code", "stock_return", "benchmark_return", "excess_return", "industry"]].copy()
        pred["predicted"] = preds
        all_preds.append(pred)
    
    if not all_preds:
        return pd.DataFrame()
    
    return pd.concat(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--train-window-months", type=int, default=18)
    parser.add_argument("--update-months", type=int, default=1)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    
    print("[online_learning] Loading dataset...")
    df = load_dataset(args.horizon, args.dataset)
    feature_cols = get_feature_cols(df)
    print(f"  Dataset: {len(df)} rows, {len(feature_cols)} features")
    
    print("[online_learning] Running walk-forward online learning...")
    result = walk_forward_online(df, feature_cols, args.train_window_months, args.update_months)
    
    if result.empty:
        print("[online_learning] No predictions generated")
        return
    
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"predictions_h{args.horizon}_online.parquet"
    output_path = PRED_DIR / output_name
    result.to_parquet(output_path, index=False)
    print(f"[online_learning] Saved {len(result)} predictions to {output_path}")


if __name__ == "__main__":
    main()
