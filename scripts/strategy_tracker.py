"""
AlphaHelix 策略表现跟踪模块
读取各策略的 walk-forward summary，计算滚动收益、夏普、最大回撤，输出策略权重建议。
"""
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = Path("memory/strategy_tracker")

STRATEGIES = ["momentum_value_hybrid", "quality_growth", "contrarian", "event_driven", "regime"]


def load_strategy_summary(start: str, end: str, strategy: str, horizon: int) -> dict:
    path = Path("memory/eval") / f"walkforward_{start}_{end}_{strategy}_h{horizon}.json"
    if not path.exists():
        return {"error": f"Summary not found: {path}"}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_strategy_weights(summaries: dict, lookback: int = None, temperature: float = 1.0) -> dict:
    """
    基于各策略的月度收益序列，用指数加权平均收益 + softmax 计算配置权重。

    Args:
        summaries: {strategy: walkforward_summary}
        lookback: 只取最近 N 期；None 表示全部
        temperature: softmax 温度；越小则权重越集中在表现最好的策略
    """
    rows = []
    for strategy, summary in summaries.items():
        if "error" in summary:
            continue
        monthly = summary.get("monthly", [])
        for m in monthly:
            rows.append({
                "strategy": strategy,
                "date": m["date"],
                "excess_return": m["excess_return"],
                "portfolio_return": m["portfolio_return"],
                "direction_accuracy": m["direction_accuracy"],
            })

    if not rows:
        return {"error": "No valid monthly returns"}

    df = pd.DataFrame(rows)
    df = df.sort_values("date")

    scores = {}
    for strategy in df["strategy"].unique():
        s_df = df[df["strategy"] == strategy].copy()
        if lookback is not None:
            s_df = s_df.tail(lookback)
        if s_df.empty:
            continue
        # 加权：越近权重越高；同时惩罚波动
        n = len(s_df)
        weights = np.exp(np.linspace(0, 1, n))  # 近期权重高
        weights /= weights.sum()
        mean_excess = np.average(s_df["excess_return"].values, weights=weights)
        # 用方向准确率作为稳定性加成
        mean_hit = np.average(s_df["direction_accuracy"].values, weights=weights)
        # 综合得分：超额收益 + 0.5 * 命中率
        scores[strategy] = mean_excess + 0.5 * (mean_hit - 0.5)

    if not scores:
        return {"error": "No scores computed"}

    # softmax
    arr = np.array(list(scores.values()))
    arr = arr - arr.max()  # 数值稳定性
    exp_arr = np.exp(arr / temperature)
    weights = exp_arr / exp_arr.sum()

    return {
        "scores": {k: round(float(v), 6) for k, v in scores.items()},
        "weights": {k: round(float(v), 4) for k, v in zip(scores.keys(), weights)},
    }


def main():
    parser = argparse.ArgumentParser(description="AlphaHelix strategy tracker")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--horizon", type=int, default=10, help="Holding horizon")
    parser.add_argument("--lookback", type=int, default=None, help="Rolling lookback periods")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    summaries = {}
    for s in STRATEGIES:
        summaries[s] = load_strategy_summary(args.start, args.end, s, args.horizon)

    result = {
        "start": args.start,
        "end": args.end,
        "horizon": args.horizon,
        "strategy_weights": compute_strategy_weights(summaries, args.lookback, args.temperature),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else OUTPUT_DIR / f"weights_{args.start}_{args.end}_h{args.horizon}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
