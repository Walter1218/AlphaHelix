"""
Walk-forward 阈值校准

对 GBDT walk-forward 预测结果，每期只用过去 N 期的数据来校准“选股置信度阈值”，
再用于当期选股。避免在全样本上优化阈值导致的过拟合。

逻辑：
1. 读取 predictions parquet；
2. 对每个再平衡日 T，取过去 train_periods 个日期作为训练窗口；
3. 在训练窗口内，对候选分位数阈值 q ∈ [q_min, q_max] 计算等权组合平均超额；
4. 选择训练窗口内 avg_excess 最高的 q；
5. T 日只选预测得分 >= T 日 q 分位数的股票（最多 max_positions 只）。

输出新的 predictions parquet，可用 portfolio_backtest.py 做完整成本回测。
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def evaluate_threshold_gross(df: pd.DataFrame, q: float, max_positions: int = 20,
                             metric: str = "avg_excess") -> float:
    """在 df 上测试分位数阈值 q 的等权组合绩效（gross）。"""
    rows = []
    for _, g in df.groupby("date"):
        th = g["predicted"].quantile(q)
        sel = g[g["predicted"] >= th].sort_values("predicted", ascending=False).head(max_positions)
        if sel.empty:
            continue
        rows.append(sel["excess_return"].mean())
    if not rows:
        return -np.inf
    arr = np.array(rows)
    if metric == "avg_excess":
        return float(np.mean(arr))
    elif metric == "win_rate":
        return float(np.mean(arr > 0))
    elif metric == "sharpe":
        std = np.std(arr)
        return float(np.mean(arr) / std * np.sqrt(len(arr))) if std > 0 else -np.inf
    else:
        return float(np.mean(arr))


def _best_q_for_df(df: pd.DataFrame, q_grid: list, max_positions: int, metric: str) -> tuple:
    """在整个 df 上挑选最优固定分位数 q（用于训练期整体校准）。"""
    best_q = q_grid[0]
    best_metric = -np.inf
    for q in q_grid:
        m = evaluate_threshold_gross(df, q, max_positions, metric)
        if m > best_metric:
            best_metric = m
            best_q = q
    return best_q, best_metric


def calibrate_threshold_config(pred_df: pd.DataFrame,
                               max_positions: int = 20,
                               q_grid: list = None,
                               metric: str = "avg_excess") -> dict:
    """
    在一份 predictions DataFrame 上校准一个固定分位数阈值 q。
    返回配置 dict，供生产环境对单日预测做阈值过滤。
    """
    if q_grid is None:
        q_grid = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    best_q, best_metric = _best_q_for_df(pred_df, q_grid, max_positions, metric)
    return {
        "q": best_q,
        "metric": metric,
        "max_positions": max_positions,
        "train_avg_metric": best_metric,
    }


def apply_quantile_threshold(pred_df: pd.DataFrame, q: float) -> pd.DataFrame:
    """对单日预测应用固定分位数阈值：低于 q 分位数的得分置为 -inf。"""
    df = pred_df.copy()
    for d, g in df.groupby("date"):
        th = g["predicted"].quantile(q)
        mask = df["date"] == d
        df.loc[mask, "predicted"] = np.where(
            df.loc[mask, "predicted"] >= th,
            df.loc[mask, "predicted"],
            -1e9,
        )
    return df


def calibrate_and_mask(pred_path: str, output_path: str,
                       train_periods: int = 12,
                       max_positions: int = 20,
                       q_grid: list = None,
                       metric: str = "avg_excess"):
    if q_grid is None:
        q_grid = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    pred = pd.read_parquet(pred_path)
    pred["date"] = pd.to_datetime(pred["date"])
    dates = sorted(pred["date"].unique())

    pred["predicted_wf"] = pred["predicted"]
    calibration_log = []

    for i, d in enumerate(dates):
        # 训练窗口：d 之前的 train_periods 个日期
        train_dates = dates[max(0, i - train_periods):i]
        test_df = pred[pred["date"] == d]

        if len(train_dates) < 3 or test_df.empty:
            calibration_log.append({"date": d, "q": None, "train_avg_excess": None})
            continue

        train_df = pred[pred["date"].isin(train_dates)]

        best_q = q_grid[0]
        best_metric = -np.inf
        for q in q_grid:
            m = evaluate_threshold_gross(train_df, q, max_positions, metric)
            if m > best_metric:
                best_metric = m
                best_q = q

        th = test_df["predicted"].quantile(best_q)
        mask = pred["date"] == d
        pred.loc[mask, "predicted_wf"] = np.where(
            pred.loc[mask, "predicted"] >= th,
            pred.loc[mask, "predicted"],
            -1e9,
        )
        calibration_log.append({
            "date": d,
            "q": best_q,
            "threshold": th,
            "train_avg_excess": best_metric,
            "n_selected": int((pred.loc[mask, "predicted"] >= th).sum()),
        })

    out_cols = [c for c in pred.columns if c != "predicted"]
    pred_out = pred[out_cols].rename(columns={"predicted_wf": "predicted"})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pred_out.to_parquet(output_path, index=False)

    log_df = pd.DataFrame(calibration_log)
    log_path = str(output_path).replace(".parquet", "_calibration.csv")
    log_df.to_csv(log_path, index=False)
    print(f"[walkforward_threshold] Saved masked predictions to {output_path}")
    print(f"[walkforward_threshold] Calibration log saved to {log_path}")
    print(f"[walkforward_threshold] Avg selected q: {log_df['q'].mean():.3f}")
    print(f"[walkforward_threshold] Avg selected count: {log_df['n_selected'].mean():.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="memory/predictions/predictions_h10_walkforward_excess_return_regression.parquet")
    parser.add_argument("--output", default="memory/predictions/predictions_h10_walkforward_threshold_wf.parquet")
    parser.add_argument("--train-periods", type=int, default=12)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--metric", choices=["avg_excess", "win_rate", "sharpe"], default="avg_excess")
    args = parser.parse_args()

    # 回测模式：禁止读取未来权重（C01/C38 纪律）
    os.environ["AH_BACKTEST_MODE"] = "1"

    calibrate_and_mask(args.pred, args.output, args.train_periods, args.max_positions, metric=args.metric)


if __name__ == "__main__":
    main()
