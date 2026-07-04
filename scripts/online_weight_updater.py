"""
AlphaHelix 在线权重更新器

为 walk-forward 在线学习提供 regime 条件的滚动权重更新。
核心约束：T 期权重只能使用 < T 期的数据，严防未来函数。

数据流：
    walkforward.py --online-update
        ↓
    每期结束后：将该期 (date, strategy, regime, snapshot, eval) 加入滚动窗口
        ↓
    重新计算该 regime 最近 N 期的因子 IC
        ↓
    用 IC 更新该 regime 的 pass1/pass2 权重
        ↓
    保存到 memory/weights/{strategy}_{regime}_rolling.json
        ↓
    下一期该 regime 触发时使用新权重
"""
import sys
import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_ic import compute_ic
from weight_optimizer import adjust_factor_weights, DEFAULT_BASE_WEIGHTS

SNAPSHOT_DIR = Path("memory/stock")
EVAL_DIR = Path("memory/eval")
WEIGHTS_DIR = Path("memory/weights")


def get_rolling_weights_path(strategy: str, regime: str) -> Path:
    return WEIGHTS_DIR / f"{strategy}_{regime}_rolling.json"


def load_rolling_weights(strategy: str, regime: str) -> dict:
    """加载 regime 滚动权重；不存在则返回策略默认权重。"""
    path = get_rolling_weights_path(strategy, regime)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("weights", {})
        except Exception:
            pass
    base = DEFAULT_BASE_WEIGHTS.get(strategy, {})
    return {
        "pass1": dict(base.get("pass1", {})),
        "pass2": dict(base.get("pass2", {})),
    }


def save_rolling_weights(strategy: str, regime: str, weights: dict, metadata: dict = None):
    """保存 regime 滚动权重到 JSON。"""
    path = get_rolling_weights_path(strategy, regime)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "strategy": strategy,
        "regime": regime,
        "weights": weights,
    }
    if metadata:
        output.update(metadata)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2))


def load_snapshot_and_eval(date: str, strategy: str, horizon: int):
    """加载某期的快照和评估结果。"""
    snap_path = SNAPSHOT_DIR / f"{date}_{strategy}.json"
    if not snap_path.exists():
        snap_path = SNAPSHOT_DIR / f"{date}.json"
    eval_path = EVAL_DIR / f"{date}_{strategy}_h{horizon}.json"

    if not snap_path.exists() or not eval_path.exists():
        return None, None

    with open(snap_path, "r", encoding="utf-8") as f:
        snapshot = json.load(f)
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_result = json.load(f)

    return snapshot, eval_result


def compute_ic_for_window(dates: list, strategy: str, horizon: int) -> dict:
    """计算一个日期窗口内的平均因子 IC。"""
    ic_values = defaultdict(list)
    valid_count = 0
    for d in dates:
        snapshot, eval_result = load_snapshot_and_eval(d, strategy, horizon)
        if snapshot is None or eval_result is None:
            continue
        try:
            ic = compute_ic(snapshot, eval_result)
            if "error" in ic:
                continue
            valid_count += 1
            for k, v in ic.items():
                if k in ("n", "date", "error"):
                    continue
                if isinstance(v, (int, float)) and not np.isnan(v):
                    ic_values[k].append(v)
        except Exception:
            continue

    if valid_count == 0:
        return {}

    return {k: float(np.mean(v)) for k, v in ic_values.items() if v}


def update_regime_weights(strategy: str, regime: str, horizon: int,
                          new_date: str, max_lookback: int = 6,
                          learning_rate: float = 0.5) -> dict:
    """
    将新日期加入该 regime 的滚动窗口，并更新权重。

    注意：new_date 本身的收益已经确定，但其权重更新结果只应用于 new_date 之后的日期，
    绝不用于 new_date 本身（由 walkforward.py 保证调用顺序）。
    """
    state_path = WEIGHTS_DIR / f"{strategy}_{regime}_rolling_state.json"

    # 加载或初始化状态
    state = {"dates": [], "last_update": None}
    if state_path.exists():
        try:
            state = json.load(open(state_path, "r", encoding="utf-8"))
        except Exception:
            pass

    # 追加新日期并截断窗口
    dates = state.get("dates", [])
    if new_date not in dates:
        dates.append(new_date)
    dates = dates[-max_lookback:]
    state["dates"] = dates
    state["last_update"] = new_date

    # 加载当前权重（作为 base）
    current_weights = load_rolling_weights(strategy, regime)

    # 计算窗口 IC
    ic_mean = compute_ic_for_window(dates, strategy, horizon)

    # 更新权重
    updated = {}
    for phase in ("pass1", "pass2"):
        base = current_weights.get(phase, {})
        if not base:
            base = DEFAULT_BASE_WEIGHTS.get(strategy, {}).get(phase, {})
        if not base:
            updated[phase] = {}
            continue

        # 将 ic_mean 转为 DataFrame 行，以便复用 adjust_factor_weights
        if ic_mean:
            ic_df = pd.DataFrame([ic_mean])
        else:
            ic_df = pd.DataFrame()
        updated[phase] = adjust_factor_weights(base, ic_df, learning_rate)

    # 保存权重和状态
    metadata = {
        "last_update": new_date,
        "based_on_dates": dates,
        "ic_mean": ic_mean,
        "learning_rate": learning_rate,
        "max_lookback": max_lookback,
    }
    save_rolling_weights(strategy, regime, updated, metadata)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))

    return updated


def initialize_rolling_weights(strategy: str, regime: str):
    """用默认权重初始化某个 regime 的滚动权重文件。"""
    weights = load_rolling_weights(strategy, regime)
    save_rolling_weights(strategy, regime, weights, {"last_update": None, "based_on_dates": []})


if __name__ == "__main__":
    # 简单测试
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--lookback", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.5)
    args = parser.parse_args()

    weights = update_regime_weights(args.strategy, args.regime, args.horizon,
                                    args.date, args.lookback, args.lr)
    print(json.dumps(weights, ensure_ascii=False, indent=2))
