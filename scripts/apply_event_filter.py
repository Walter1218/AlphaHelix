"""
事件过滤/增强 overlay

输入：GBDT walk-forward 预测结果 + 带事件特征的数据集（如 features_h10_composite_phase2.parquet）
输出：应用事件过滤后的组合绩效

策略：
- `strict`：只从 disclosure_near=1 或 since_disclosure_lt10=1 的股票中选 top-k；
- `prefer`：优先选带事件标志的股票，若不足 top-k 再用普通候选补齐。

只计算近似等权绩效（未扣成本），用于判断事件特征是否有独立增量。
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_event_features(pred_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.strftime("%Y%m%d")
    cols = ["date", "ts_code", "disclosure_near", "disclosure_very_near", "since_disclosure_lt10"]
    cols = [c for c in cols if c in df.columns]
    ev = df[cols].drop_duplicates(subset=["date", "ts_code"])
    return pred_df.merge(ev, on=["date", "ts_code"], how="left")


def evaluate_overlay(pred_path: str, dataset: str, max_positions: int = 20, mode: str = "strict"):
    pred = pd.read_parquet(pred_path)
    pred = load_event_features(pred, dataset)
    for c in ["disclosure_near", "disclosure_very_near", "since_disclosure_lt10"]:
        if c in pred.columns:
            pred[c] = pred[c].fillna(0)

    rows = []
    for d, g in pred.groupby("date"):
        g = g.sort_values("predicted", ascending=False)
        event_mask = ((g.get("disclosure_near", 0) == 1) |
                      (g.get("disclosure_very_near", 0) == 1) |
                      (g.get("since_disclosure_lt10", 0) == 1))
        event_pool = g[event_mask]
        normal_pool = g[~event_mask]

        if mode == "strict":
            selected = event_pool.head(max_positions)
        elif mode == "prefer":
            selected = pd.concat([event_pool, normal_pool]).head(max_positions)
        else:
            selected = g.head(max_positions)

        if selected.empty:
            rows.append({"date": d, "excess": 0.0, "n": 0, "event_ratio": 0.0})
        else:
            rows.append({
                "date": d,
                "excess": selected["excess_return"].mean(),
                "n": len(selected),
                "event_ratio": event_mask.mean(),
            })

    out = pd.DataFrame(rows).sort_values("date")
    out["cum"] = (1 + out["excess"]).cumprod() - 1

    # baseline
    base_rows = []
    for d, g in pred.groupby("date"):
        top = g.sort_values("predicted", ascending=False).head(max_positions)
        base_rows.append({"date": d, "excess": top["excess_return"].mean()})
    base = pd.DataFrame(base_rows).sort_values("date")
    base["cum"] = (1 + base["excess"]).cumprod() - 1

    return {
        "mode": mode,
        "avg_excess": float(out["excess"].mean()),
        "cum_excess": float(out["cum"].iloc[-1]),
        "win_rate": float((out["excess"] > 0).mean()),
        "avg_n": float(out["n"].mean()),
        "baseline_avg": float(base["excess"].mean()),
        "baseline_cum": float(base["cum"].iloc[-1]),
        "baseline_win": float((base["excess"] > 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="memory/predictions/predictions_h10_walkforward_excess_return_regression.parquet")
    parser.add_argument("--dataset", default="memory/dataset/features_h10_composite_phase2.parquet")
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--mode", choices=["strict", "prefer", "none"], default="strict")
    args = parser.parse_args()

    res = evaluate_overlay(args.pred, args.dataset, args.max_positions, args.mode)
    print(f"=== Event Overlay ({res['mode']}) ===")
    print(f"Baseline  avg={res['baseline_avg']:.4f}  cum={res['baseline_cum']:.4f}  win={res['baseline_win']:.1%}")
    print(f"Overlay   avg={res['avg_excess']:.4f}  cum={res['cum_excess']:.4f}  win={res['win_rate']:.1%}  avg_n={res['avg_n']:.1f}")


if __name__ == "__main__":
    main()
