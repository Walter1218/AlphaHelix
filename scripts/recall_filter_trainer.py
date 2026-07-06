"""
召回层过滤实验

在已有 composite 数据集上应用召回规则，重新训练 GBDT 回归模型并输出预测。
目的是验证：通过质量/波动率/市值等规则过滤后，排序模型的 top-20 胜率是否提升。

用法：
python scripts/recall_filter_trainer.py \
  --dataset memory/dataset/features_h10_composite.parquet \
  --filters roe:0.2:1 profit_growth:0.2:1 volatility_20:0:0.8 total_mv:0.2:1
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_trainer import load_dataset, get_feature_cols, walk_forward_predict
from _tushare_utils import fetch_disclosure_schedule, get_trade_calendar

PRED_DIR = Path("memory/predictions")


def parse_filters(raw: str) -> dict:
    """
    解析命令行过滤规则。
    格式: col:min:max,col2:min:max
    用 'none' 表示不限制该侧。
    """
    filters = {}
    if not raw:
        return filters
    for part in raw.split(","):
        tokens = part.split(":")
        if len(tokens) != 3:
            raise ValueError(f"Invalid filter: {part}")
        col, mn, mx = tokens
        bounds = {}
        if mn.lower() != "none":
            bounds["min"] = float(mn)
        if mx.lower() != "none":
            bounds["max"] = float(mx)
        filters[col.strip()] = bounds
    return filters


def _compute_trade_day_diff(trade_dates: list, start: pd.Timestamp, end: pd.Timestamp) -> int:
    """计算 start 到 end 之间的交易日天数（不含 start，含 end）。"""
    try:
        start_idx = trade_dates.index(start.strftime("%Y%m%d"))
        end_idx = trade_dates.index(end.strftime("%Y%m%d"))
        return max(0, end_idx - start_idx)
    except ValueError:
        return np.nan


def apply_disclosure_filter(df: pd.DataFrame, exclude_days: int) -> pd.DataFrame:
    """
    剔除未来 exclude_days 个交易日内有财报/季报预约披露的股票。

    基于 Tushare disclosure_date 接口获取预约披露时间表（pre_date），
    对每只股票计算下一个披露日距离当前决策日有多少个交易日，
    若 <= exclude_days 则剔除。
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    years = sorted(df["date"].dt.year.unique().tolist())
    schedule = fetch_disclosure_schedule(years=years)
    if schedule.empty or "pre_date" not in schedule.columns:
        print("[recall_filter_trainer] No disclosure schedule available, skip filter")
        return df

    schedule = schedule.copy()
    schedule["pre_date"] = pd.to_datetime(schedule["pre_date"], format="%Y%m%d", errors="coerce")
    schedule = schedule[schedule["pre_date"].notna()].copy()

    # 获取交易日历
    start_str = df["date"].min().strftime("%Y%m%d")
    end_str = (df["date"].max() + pd.Timedelta(days=180)).strftime("%Y%m%d")
    cal = get_trade_calendar("SSE", start_str, end_str)
    trade_dates = cal[cal["is_open"].astype(int) == 1]["cal_date"].astype(str).tolist()

    # 为每个 (date, ts_code) 计算下一个披露日的交易日差
    def get_days(row):
        code = row["ts_code"]
        d = row["date"]
        sub = schedule[schedule["ts_code"] == code]
        future = sub[sub["pre_date"] > d]
        if future.empty:
            return np.nan
        next_date = future["pre_date"].min()
        return _compute_trade_day_diff(trade_dates, d, next_date)

    df["days_to_disclosure"] = df.apply(get_days, axis=1)

    # 保留：缺失（无未来披露）或大于 exclude_days 的样本
    mask = df["days_to_disclosure"].isna() | (df["days_to_disclosure"] > exclude_days)
    kept = df[mask].copy()
    n_dropped = len(df) - len(kept)
    if n_dropped:
        print(f"[recall_filter_trainer] Disclosure filter dropped {n_dropped} rows "
              f"({n_dropped/len(df):.1%}) with disclosure within {exclude_days} trade days")
    kept = kept.drop(columns=["days_to_disclosure"])
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--model-type", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--train-window-months", type=int, default=12)
    parser.add_argument("--target", default="excess_return")
    parser.add_argument("--filters", default="",
                        help="召回过滤规则，例如 roe:0.2:1,volatility_20:0:0.8")
    parser.add_argument("--output-name", default=None,
                        help="输出文件名，默认自动生成")
    parser.add_argument("--disclosure-exclude-days", type=int, default=None,
                        help="剔除未来 N 个交易日内有财报/季报预约披露的股票")
    args = parser.parse_args()

    df = load_dataset(args.horizon, args.dataset)
    if args.disclosure_exclude_days is not None:
        df = apply_disclosure_filter(df, args.disclosure_exclude_days)
    feature_cols = get_feature_cols(df)
    filters = parse_filters(args.filters)
    print(f"[recall_filter_trainer] Filters: {filters}")

    pred_df = walk_forward_predict(
        df, feature_cols,
        train_window_months=args.train_window_months,
        model_type=args.model_type,
        target=args.target,
        objective="regression",
        recall_filters=filters,
    )

    if pred_df.empty:
        print("[recall_filter_trainer] No predictions generated")
        return

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        output_path = PRED_DIR / args.output_name
    else:
        tag = args.filters.replace(":", "_").replace(",", "-") or "nofilter"
        output_path = PRED_DIR / f"predictions_h{args.horizon}_recall_{tag}.parquet"
    pred_df.to_parquet(output_path, index=False)
    print(f"[recall_filter_trainer] Saved {len(pred_df)} predictions to {output_path}")

    # 简单诊断
    top20 = pred_df.groupby("date").apply(lambda g: g.sort_values("predicted", ascending=False).head(20), include_groups=False)
    if not top20.empty:
        avg_excess = top20["excess_return"].mean()
        pos_ratio = (top20["excess_return"] > 0).mean()
        print(f"\nDiagnostic: top20 avg excess={avg_excess:+.4f}, positive ratio={pos_ratio:.2%}")


if __name__ == "__main__":
    main()
