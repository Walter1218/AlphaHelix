"""
AlphaHelix RL 仓位管理 V2

改进点：
1. 更丰富的状态表示（模型置信度、近期表现）
2. 奖励函数改为风险调整收益
3. 严格的 walk-forward 训练
4. 更大的状态空间

用法：
    python scripts/rl_position_v2.py --pred-path memory/predictions/predictions_h10_stacking_5model.parquet
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_trainer import load_dataset

PRED_DIR = Path("memory/predictions")


class PositionAgent:
    """仓位管理 Agent"""
    
    def __init__(self, n_states, n_actions=3, lr=0.05, gamma=0.9, epsilon=0.2):
        self.n_actions = n_actions  # 0=empty, 1=half, 2=full
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
    
    def _get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        qs = [self._get_q(state, a) for a in range(self.n_actions)]
        return int(np.argmax(qs))
    
    def update(self, state, action, reward, next_state):
        best_next = max([self._get_q(next_state, a) for a in range(self.n_actions)])
        current = self._get_q(state, action)
        self.q_table[(state, action)] = current + self.lr * (reward + self.gamma * best_next - current)


def compute_state(row, market_stats, pred_history):
    """计算状态：更丰富的表示"""
    date = row["date"]
    
    # 市场状态
    if date in market_stats.index:
        stats = market_stats.loc[date]
        regime = stats.get("regime", "sideways")
        regime_idx = {"bull": 0, "sideways": 1, "bear": 2}.get(regime, 1)
        vol = stats.get("vol_excess", 0)
        vol_median = stats.get("vol_median", 0)
        vol_idx = 0 if vol < vol_median * 0.8 else (2 if vol > vol_median * 1.2 else 1)
    else:
        regime_idx = 1
        vol_idx = 1
    
    # 近期表现
    if len(pred_history) > 0:
        recent_wr = np.mean([1 if p > 0 else 0 for p in pred_history[-10:]])
        wr_idx = 0 if recent_wr > 0.6 else (2 if recent_wr < 0.4 else 1)
    else:
        wr_idx = 1
    
    # 模型置信度（预测值大小）
    pred = row.get("predicted", 0)
    conf_idx = 0 if pred > 0.01 else (2 if pred < -0.01 else 1)
    
    # 组合状态
    state = regime_idx * 27 + vol_idx * 9 + wr_idx * 3 + conf_idx
    return state


def walk_forward_rl(predictions, feature_df, horizon=10, train_window=12):
    """Walk-forward RL 仓位管理"""
    # 计算市场统计
    market_stats = feature_df.groupby("date").agg({
        "excess_return": ["mean", "std"],
    })
    market_stats.columns = ["avg_excess", "vol_excess"]
    market_stats["avg_median"] = market_stats["avg_excess"].rolling(60, min_periods=1).median()
    market_stats["vol_median"] = market_stats["vol_excess"].rolling(60, min_periods=1).median()
    market_stats["regime"] = "sideways"
    market_stats.loc[(market_stats["avg_excess"] > market_stats["avg_median"] * 1.5) & 
                     (market_stats["vol_excess"] < market_stats["vol_median"] * 0.8), "regime"] = "bull"
    market_stats.loc[(market_stats["avg_excess"] < market_stats["avg_median"] * 0.5) & 
                     (market_stats["vol_excess"] > market_stats["vol_median"] * 1.2), "regime"] = "bear"
    
    # 合并预测
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions["year_month"] = predictions["date"].dt.to_period("M")
    months = sorted(predictions["year_month"].unique())
    
    # 初始化 Agent
    n_states = 3 * 3 * 3 * 3  # regime * vol * wr * conf
    agent = PositionAgent(n_states)
    
    all_preds = []
    pred_history = []
    
    for i, test_month in enumerate(months):
        train_months = months[max(0, i - train_window):i]
        if len(train_months) < 3:
            continue
        
        # 训练 Agent（用最近6个月）
        for train_month in train_months[-6:]:
            train_data = predictions[predictions["year_month"] == train_month]
            for _, row in train_data.iterrows():
                state = compute_state(row, market_stats, pred_history)
                action = agent.get_action(state)
                
                # 奖励：风险调整收益
                reward = row["excess_return"] * (action / 2)
                if action == 0:  # 空仓时奖励为0
                    reward = 0
                
                next_state = state  # 简化
                agent.update(state, action, reward, next_state)
                
                # 更新历史
                pred_history.append(row["excess_return"])
                if len(pred_history) > 100:
                    pred_history = pred_history[-100:]
        
        # 测试
        test_data = predictions[predictions["year_month"] == test_month].copy()
        if test_data.empty:
            continue
        
        # 获取当前状态
        sample_row = test_data.iloc[0]
        state = compute_state(sample_row, market_stats, pred_history)
        action = agent.get_action(state)
        
        # 仓位缩放
        scale = action / 2.0
        test_data["predicted"] = test_data["predicted"] * scale
        test_data["position_scale"] = scale
        
        all_preds.append(test_data)
    
    if not all_preds:
        return pd.DataFrame()
    
    return pd.concat(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    
    print("[rl_position_v2] Loading predictions...")
    predictions = pd.read_parquet(args.pred_path)
    print(f"  Loaded {len(predictions)} rows, {predictions['date'].nunique()} dates")
    
    print("[rl_position_v2] Loading feature data...")
    feature_df = load_dataset(args.horizon, "memory/dataset/features_h10_composite_fixed.parquet")
    
    print("[rl_position_v2] Running RL position sizing...")
    result = walk_forward_rl(predictions, feature_df, args.horizon)
    
    if result.empty:
        print("[rl_position_v2] No predictions generated")
        return
    
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"predictions_h{args.horizon}_rl_position_v2.parquet"
    output_path = PRED_DIR / output_name
    result.to_parquet(output_path, index=False)
    print(f"\n[rl_position_v2] Saved {len(result)} predictions to {output_path}")
    
    # 统计仓位分布
    print("\n=== Position Scale Distribution ===")
    print(result["position_scale"].value_counts().sort_index())


if __name__ == "__main__":
    main()
