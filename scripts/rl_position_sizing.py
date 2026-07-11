"""
AlphaHelix RL 仓位管理

用 Q-learning 学习何时满仓/半仓/空仓。

状态空间：
- 市场 regime (bull/bear/sideways)
- 波动率状态 (low/medium/high)
- 近期收益 (positive/negative)

动作空间：
- full (1.0): 满仓
- half (0.5): 半仓
- empty (0.0): 空仓

奖励：
- 组合收益

用法：
    python scripts/rl_position_sizing.py --pred-path memory/predictions/predictions_h10_stacking_5model.parquet
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


class QLearningAgent:
    """简单的 Q-learning Agent"""
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (reward + self.gamma * best_next - self.q_table[state, action])


def detect_state(row, market_stats):
    """检测市场状态"""
    date = row["date"]
    if date not in market_stats.index:
        return 0  # default state
    
    stats = market_stats.loc[date]
    
    # Regime
    regime = stats.get("regime", "sideways")
    regime_idx = {"bull": 0, "sideways": 1, "bear": 2}.get(regime, 1)
    
    # 波动率状态
    vol = stats.get("vol_excess", 0)
    vol_median = stats.get("vol_median", 0)
    if vol < vol_median * 0.8:
        vol_idx = 0  # low
    elif vol > vol_median * 1.2:
        vol_idx = 2  # high
    else:
        vol_idx = 1  # medium
    
    # 近期收益
    avg_excess = stats.get("avg_excess", 0)
    if avg_excess > 0:
        ret_idx = 0  # positive
    else:
        ret_idx = 1  # negative
    
    # 组合状态
    state = regime_idx * 4 + vol_idx * 2 + ret_idx
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
    
    # 初始化 RL Agent
    n_states = 3 * 3 * 2  # regime * vol * ret
    n_actions = 3  # full, half, empty
    agent = QLearningAgent(n_states, n_actions)
    
    all_preds = []
    
    for i, test_month in enumerate(months):
        train_months = months[max(0, i - train_window):i]
        if len(train_months) < 3:
            continue
        
        # 训练 RL Agent
        for train_month in train_months[-6:]:  # 只用最近6个月训练
            train_data = predictions[predictions["year_month"] == train_month]
            for _, row in train_data.iterrows():
                state = detect_state(row, market_stats)
                action = agent.get_action(state)
                reward = row["excess_return"] * (action / 2)  # 0=empty, 1=half, 2=full
                next_state = state  # 简化：假设状态不变
                agent.update(state, action, reward, next_state)
        
        # 测试：用 RL Agent 决定仓位
        test_data = predictions[predictions["year_month"] == test_month].copy()
        if test_data.empty:
            continue
        
        # 获取当前状态
        sample_row = test_data.iloc[0]
        state = detect_state(sample_row, market_stats)
        action = agent.get_action(state)
        
        # 仓位缩放
        scale = action / 2.0  # 0=empty, 0.5=half, 1.0=full
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
    
    print("[rl_position_sizing] Loading predictions...")
    predictions = pd.read_parquet(args.pred_path)
    print(f"  Loaded {len(predictions)} rows, {predictions['date'].nunique()} dates")
    
    print("[rl_position_sizing] Loading feature data...")
    feature_df = load_dataset(args.horizon, "memory/dataset/features_h10_composite_fixed.parquet")
    
    print("[rl_position_sizing] Running RL position sizing...")
    result = walk_forward_rl(predictions, feature_df, args.horizon)
    
    if result.empty:
        print("[rl_position_sizing] No predictions generated")
        return
    
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"predictions_h{args.horizon}_rl_positioning.parquet"
    output_path = PRED_DIR / output_name
    result.to_parquet(output_path, index=False)
    print(f"\n[rl_position_sizing] Saved {len(result)} predictions to {output_path}")
    
    # 统计仓位分布
    print("\n=== Position Scale Distribution ===")
    print(result["position_scale"].value_counts().sort_index())


if __name__ == "__main__":
    main()
