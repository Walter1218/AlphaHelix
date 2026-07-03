"""
AlphaHelix 多目标离线权重优化器

在方向准确率硬约束下，搜索使超额收益最大的因子权重组合。

形式化：
    maximize  avg_excess_return
    subject to avg_direction_accuracy >= threshold

输入：已有 walk-forward 产物
    - memory/stock/{date}_{strategy}.json   选股快照（含因子值）
    - memory/eval/{date}_{strategy}_h{horizon}.json  评估结果（含实际收益）

输出：
    - memory/weights/{strategy}_mo_latest.json
"""
import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import get_trade_calendar
from screen import STRATEGIES, rank_fill

SNAPSHOT_DIR = Path("memory/stock")
EVAL_DIR = Path("memory/eval")
WEIGHTS_DIR = Path("memory/weights")


def get_monthly_trade_dates(start_date: str, end_date: str) -> list:
    cal = get_trade_calendar("SSE", start_date, end_date)
    cal = cal[cal["is_open"].astype(int) == 1].copy()
    cal["cal_date"] = pd.to_datetime(cal["cal_date"], format="%Y%m%d")
    cal = cal.sort_values("cal_date")
    cal["ym"] = cal["cal_date"].dt.to_period("M")
    monthly = cal.groupby("ym")["cal_date"].last().dt.strftime("%Y%m%d").tolist()
    return monthly


def load_period_data(date: str, strategy: str, horizon: int) -> dict:
    """加载单期的快照与评估数据，返回可重用的 DataFrame 与 benchmark。优先使用完整 pass2 池。"""
    snap_path = SNAPSHOT_DIR / f"{date}_{strategy}_full.json"
    if not snap_path.exists():
        snap_path = SNAPSHOT_DIR / f"{date}_{strategy}.json"
    eval_path = EVAL_DIR / f"{date}_{strategy}_h{horizon}.json"

    if not snap_path.exists():
        raise FileNotFoundError(f"Strategy snapshot not found: {snap_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    with open(snap_path, "r", encoding="utf-8") as f:
        snapshot = json.load(f)
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    picks = snapshot.get("picks", [])
    if not picks:
        return None

    df = pd.DataFrame(picks)
    df = df.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)

    # 合并实际收益
    details = eval_data.get("details", [])
    returns_map = {d["ts_code"]: d for d in details if "ts_code" in d}
    df["abs_return"] = df["ts_code"].map(lambda x: returns_map.get(x, {}).get("abs_return", np.nan))
    df["excess_return"] = df["ts_code"].map(lambda x: returns_map.get(x, {}).get("excess_return", np.nan))

    # 只保留有收益数据的行
    df = df.dropna(subset=["abs_return"]).reset_index(drop=True)
    if df.empty:
        return None

    benchmark_return = eval_data.get("benchmark_return", 0.0)
    return {"df": df, "benchmark_return": benchmark_return, "date": date}


def factor_series(df: pd.DataFrame, factor: str) -> pd.Series:
    """复现 screen.py 中的因子标准化逻辑。"""
    if factor == "ep":
        return rank_fill(1 / df["pe"].replace(0, np.nan))
    if factor == "bp":
        return rank_fill(1 / df["pb"].replace(0, np.nan))
    if factor == "sp":
        return rank_fill(1 / df["ps"].replace(0, np.nan))
    if factor == "dividend":
        return rank_fill(df["dv_ratio"])
    if factor == "size":
        return rank_fill(df["total_mv"])
    if factor == "liquidity":
        return rank_fill(df["avg_amount_20"])
    if factor in df.columns:
        return rank_fill(df[factor])
    return pd.Series([0.5] * len(df), index=df.index)


def evaluate_weights(periods_data: list, weights: dict, top_n: int) -> dict:
    """对一组权重，在历史截面上重新选股并计算绩效。"""
    portfolio_returns = []
    excess_returns = []
    directions = []

    for pdata in periods_data:
        df = pdata["df"].copy()
        benchmark_return = pdata["benchmark_return"]

        score = pd.Series(0.0, index=df.index)
        for factor, w in weights.items():
            score += w * factor_series(df, factor)

        df["score"] = score
        selected = df.sort_values("score", ascending=False).head(top_n)
        if selected.empty:
            continue

        port_ret = selected["abs_return"].mean()
        exc_ret = port_ret - benchmark_return
        direction = 1.0 if exc_ret > 0 else 0.0

        portfolio_returns.append(port_ret)
        excess_returns.append(exc_ret)
        directions.append(direction)

    if not portfolio_returns:
        return None

    return {
        "avg_portfolio_return": float(np.mean(portfolio_returns)),
        "avg_excess_return": float(np.mean(excess_returns)),
        "avg_direction_accuracy": float(np.mean(directions)),
        "win_rate_excess": float(np.mean([e > 0 for e in excess_returns])),
    }


def random_weights(factors: list, rng: np.random.Generator, n: int) -> list:
    """生成 n 组随机权重（带符号，允许负权重，最后归一化到 L1 范数为 1）。"""
    weights = []
    for _ in range(n):
        w = rng.normal(0, 1, size=len(factors))
        w = w / (np.sum(np.abs(w)) + 1e-12)
        weights.append(dict(zip(factors, w)))
    return weights


def grid_weights(factors: list, levels: list) -> list:
    """对少量因子做网格搜索（谨慎使用，因子多时会爆炸）。"""
    import itertools
    weights = []
    for combo in itertools.product(levels, repeat=len(factors)):
        # 跳过全零
        if all(v == 0 for v in combo):
            continue
        w = np.array(combo, dtype=float)
        w = w / (np.sum(np.abs(w)) + 1e-12)
        weights.append(dict(zip(factors, w)))
    return weights


def optimize(periods_data: list, factors: list, top_n: int, threshold: float,
             n_trials: int, seed: int = 42) -> dict:
    """多目标随机搜索。"""
    rng = np.random.default_rng(seed)
    baseline_weights = STRATEGIES.get(periods_data[0].get("strategy", ""), {}).get("pass2", {}).get("weights", {}) if False else {}

    best = None
    valid_results = []
    baseline_metrics = None

    # 先生成 baseline（当前硬编码权重）
    # 这里无法直接拿到 pass2 权重，由调用方传入

    for trial in range(n_trials):
        weights_vec = rng.normal(0, 1, size=len(factors))
        weights_vec = weights_vec / (np.sum(np.abs(weights_vec)) + 1e-12)
        weights = dict(zip(factors, weights_vec))

        metrics = evaluate_weights(periods_data, weights, top_n)
        if metrics is None:
            continue

        if metrics["avg_direction_accuracy"] >= threshold:
            valid_results.append({"weights": weights, "metrics": metrics})
            if best is None or metrics["avg_excess_return"] > best["metrics"]["avg_excess_return"]:
                best = {"weights": weights, "metrics": metrics}

    return {"best": best, "valid_count": len(valid_results), "total_trials": n_trials}


def main():
    parser = argparse.ArgumentParser(description="AlphaHelix multi-objective offline weight optimizer")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--strategy", required=True, help="Strategy name")
    parser.add_argument("--horizon", type=int, default=10, help="Holding horizon")
    parser.add_argument("--top-n", type=int, default=10, help="Number of picks per period")
    parser.add_argument("--phase", default="pass2", choices=["pass1", "pass2"], help="Which phase weights to optimize")
    parser.add_argument("--threshold", type=float, default=0.70, help="Minimum avg direction accuracy")
    parser.add_argument("--n-trials", type=int, default=3000, help="Random search trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.strategy not in STRATEGIES:
        print(f"Unknown strategy: {args.strategy}. Available: {list(STRATEGIES.keys())}")
        sys.exit(1)

    strategy_config = STRATEGIES[args.strategy]
    factors = list(strategy_config[args.phase]["weights"].keys())
    print(f"[mo_optimizer] Strategy={args.strategy}, phase={args.phase}, factors={factors}")
    print(f"[mo_optimizer] Threshold={args.threshold:.1%}, trials={args.n_trials}, top_n={args.top_n}, horizon={args.horizon}")

    dates = get_monthly_trade_dates(args.start, args.end)
    print(f"[mo_optimizer] Loading {len(dates)} periods from {args.start} to {args.end}")

    periods_data = []
    for d in dates:
        try:
            pdata = load_period_data(d, args.strategy, args.horizon)
            if pdata:
                periods_data.append(pdata)
        except Exception as e:
            print(f"[mo_optimizer] Skip {d}: {e}")

    if len(periods_data) < 2:
        print("[mo_optimizer] Not enough valid periods")
        sys.exit(1)

    print(f"[mo_optimizer] Valid periods: {len(periods_data)}")

    # Baseline: current hardcoded weights
    baseline_weights = strategy_config[args.phase]["weights"]
    baseline_metrics = evaluate_weights(periods_data, baseline_weights, args.top_n)
    print(f"[mo_optimizer] Baseline weights: {baseline_weights}")
    print(f"[mo_optimizer] Baseline metrics: avg_ret={baseline_metrics['avg_portfolio_return']:+.2%} "
          f"avg_excess={baseline_metrics['avg_excess_return']:+.2%} "
          f"dir_acc={baseline_metrics['avg_direction_accuracy']:.1%}")

    result = optimize(periods_data, factors, args.top_n, args.threshold, args.n_trials, args.seed)
    print(f"[mo_optimizer] Valid combos (dir_acc >= {args.threshold:.1%}): {result['valid_count']} / {result['total_trials']}")

    if result["best"] is None:
        print(f"[mo_optimizer] No combination meets direction accuracy threshold {args.threshold:.1%}")
        print("[mo_optimizer] Try lowering --threshold or increasing --n-trials")
        sys.exit(0)

    best = result["best"]
    print(f"[mo_optimizer] Best weights: {best['weights']}")
    print(f"[mo_optimizer] Best metrics: avg_ret={best['metrics']['avg_portfolio_return']:+.2%} "
          f"avg_excess={best['metrics']['avg_excess_return']:+.2%} "
          f"dir_acc={best['metrics']['avg_direction_accuracy']:.1%}")

    output = {
        "strategy": args.strategy,
        "phase": args.phase,
        "start_date": args.start,
        "end_date": args.end,
        "horizon": args.horizon,
        "top_n": args.top_n,
        "threshold": args.threshold,
        "n_trials": args.n_trials,
        "baseline": {
            "weights": baseline_weights,
            "metrics": baseline_metrics,
        },
        "optimized": best,
        "valid_count": result["valid_count"],
        "total_trials": result["total_trials"],
        "generated_at": datetime.now().isoformat(),
    }

    output_path = Path(args.output) if args.output else WEIGHTS_DIR / f"{args.strategy}_mo_latest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"[mo_optimizer] Saved to {output_path}")


if __name__ == "__main__":
    main()
