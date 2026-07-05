"""
宏观择时/仓位缩放

输入：GBDT walk-forward 预测结果（parquet）
输出：应用宏观仓位缩放后的组合绩效

逻辑：
- 对每一个再平衡日，读取/计算宏观指标（北向资金 20 日 zscore、融资融券 5 日变化率）。
- 计算 regime_score ∈ [-1, 1]，越负面仓位越低。
- position_scale = clip(1 + regime_score, 0, 1)。
  例如：
  - regime_score = +1  -> 满仓（scale=1）
  - regime_score = 0   -> 半仓（scale=0.5）
  - regime_score = -1  -> 空仓（scale=0）
- 组合当日超额 = top-k 股票平均超额 * position_scale；现金部分超额为 0。

注意：本脚本只做近似绩效测算，未重新精细计算交易成本。若 regime 有效，
应在 portfolio_backtest.py 中接入 position_multiplier 做完整回测。
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portfolio_backtest import run_backtest


def load_macro_features(pred_df: pd.DataFrame, macro_dataset: str = None) -> pd.DataFrame:
    """加载宏观特征并与预测表按 date 合并。"""
    if macro_dataset and Path(macro_dataset).exists():
        df = pd.read_parquet(macro_dataset)
    else:
        raise FileNotFoundError(f"Macro dataset not found: {macro_dataset}")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.strftime("%Y%m%d")

    macro_cols = ["date", "margin_total_balance", "margin_change_5d", "margin_change_20d",
                  "northbound_net_today", "northbound_net_5d_sum", "northbound_net_20d_sum",
                  "northbound_net_20d_zscore", "northbound_net_5d_vs_20d"]
    macro_cols = [c for c in macro_cols if c in df.columns]
    macro = df[macro_cols].drop_duplicates(subset=["date"])
    return pred_df.merge(macro, on="date", how="left")


def compute_regime_score(row: pd.Series) -> float:
    """基于北向和融资融券计算 -1~1 的 regime 分数。"""
    nb = row.get("northbound_net_20d_zscore", 0)
    margin = row.get("margin_change_5d", 0)

    # 标准化到 [-1, 1]
    nb_score = np.clip(nb / 2.0, -1, 1) if pd.notna(nb) else 0.0
    margin_score = np.clip(margin / 0.05, -1, 1) if pd.notna(margin) else 0.0

    return 0.6 * nb_score + 0.4 * margin_score


def apply_timing(pred_path: str, macro_dataset: str, max_positions: int = 20,
                 regime_threshold: float = -0.5) -> dict:
    pred = pd.read_parquet(pred_path)
    pred = load_macro_features(pred, macro_dataset)
    pred["regime_score"] = pred.apply(compute_regime_score, axis=1)
    pred["position_scale"] = (1 + pred["regime_score"]).clip(0, 1)

    # 默认 composite baseline 的 top20 等权组合
    rows = []
    for d, g in pred.groupby("date"):
        top = g.sort_values("predicted", ascending=False).head(max_positions)
        scale = top["position_scale"].iloc[0]
        if scale <= 0 or len(top) == 0:
            rows.append({"date": d, "excess": 0.0, "scale": 0.0})
        else:
            avg_excess = top["excess_return"].mean()
            rows.append({"date": d, "excess": avg_excess * scale, "scale": scale})

    timing = pd.DataFrame(rows).sort_values("date")
    timing["cum_excess"] = (1 + timing["excess"]).cumprod() - 1

    # 也计算 baseline（无择时）
    baseline_rows = []
    for d, g in pred.groupby("date"):
        top = g.sort_values("predicted", ascending=False).head(max_positions)
        baseline_rows.append({"date": d, "excess": top["excess_return"].mean()})
    base = pd.DataFrame(baseline_rows).sort_values("date")
    base["cum_excess"] = (1 + base["excess"]).cumprod() - 1

    return {
        "avg_excess_timing": float(timing["excess"].mean()),
        "cum_excess_timing": float(timing["cum_excess"].iloc[-1]),
        "win_rate_timing": float((timing["excess"] > 0).mean()),
        "avg_scale": float(timing["scale"].mean()),
        "avg_excess_baseline": float(base["excess"].mean()),
        "cum_excess_baseline": float(base["cum_excess"].iloc[-1]),
        "win_rate_baseline": float((base["excess"] > 0).mean()),
        "dates": timing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="memory/predictions/predictions_h10_walkforward_excess_return_regression.parquet")
    parser.add_argument("--macro-dataset", default="memory/dataset/features_h10_composite_phase3.parquet")
    parser.add_argument("--max-positions", type=int, default=20)
    args = parser.parse_args()

    res = apply_timing(args.pred, args.macro_dataset, args.max_positions)
    print("=== Market Timing vs Baseline ===")
    print(f"Baseline  avg excess={res['avg_excess_baseline']:.4f}  cum={res['cum_excess_baseline']:.4f}  win={res['win_rate_baseline']:.1%}")
    print(f"Timing    avg excess={res['avg_excess_timing']:.4f}  cum={res['cum_excess_timing']:.4f}  win={res['win_rate_timing']:.1%}  avg_scale={res['avg_scale']:.2f}")


if __name__ == "__main__":
    main()
