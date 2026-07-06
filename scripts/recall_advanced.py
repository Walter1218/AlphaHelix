"""
高级召回层实验：动态阈值 / 负样本过滤 / 业绩预告事件过滤

保持排序模型不变（GBDT 回归 h=10），只在召回层做优化，输出预测供 walkforward_gbdt.py 回测。
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_trainer import load_dataset, get_feature_cols, _train_model_for_fold
from _tushare_utils import tushare_call, get_trade_calendar

PRED_DIR = Path("memory/predictions")


def _compute_trade_day_diff(trade_dates: list, start: pd.Timestamp, end: pd.Timestamp) -> float:
    try:
        start_idx = trade_dates.index(start.strftime("%Y%m%d"))
        end_idx = trade_dates.index(end.strftime("%Y%m%d"))
        return max(0, end_idx - start_idx)
    except ValueError:
        return np.nan


def _get_trade_dates(start: pd.Timestamp, end: pd.Timestamp) -> list:
    cal = get_trade_calendar("SSE", start.strftime("%Y%m%d"),
                             (end + pd.Timedelta(days=30)).strftime("%Y%m%d"))
    return cal[cal["is_open"].astype(int) == 1]["cal_date"].astype(str).tolist()


def _make_bottom20_label(df: pd.DataFrame) -> np.ndarray:
    return (df.groupby("date")["excess_return"].rank(pct=True) <= 0.2).astype(int).values


def _apply_static_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for col, bounds in filters.items():
        if col not in df.columns:
            continue
        if "min" in bounds:
            mask &= df[col] >= bounds["min"]
        if "max" in bounds:
            mask &= df[col] <= bounds["max"]
    return df[mask].copy()


def _score_threshold_combo(df: pd.DataFrame, combo: tuple, metric: str = "win_rate") -> float:
    """用历史数据评估一组 (vol_max, mv_min) 阈值。"""
    vol_max, mv_min = combo
    sub = df[(df["volatility_20"] <= vol_max) & (df["total_mv"] >= mv_min)].copy()
    if sub.empty:
        return -999.0
    top20 = sub.groupby("date").apply(lambda g: g.sort_values("predicted", ascending=False).head(20),
                                      include_groups=False)
    if top20.empty or len(top20) < 20:
        return -999.0
    rets = top20["excess_return"].values
    if metric == "win_rate":
        return float(np.mean(rets > 0))
    elif metric == "sharpe":
        if rets.std(ddof=1) == 0:
            return -999.0
        ann_factor = 252 / 10
        return float((rets.mean() * ann_factor) / (rets.std(ddof=1) * np.sqrt(ann_factor)))
    else:  # avg_excess
        return float(rets.mean())


def fetch_earnings_events(dates: list, event_types: list) -> pd.DataFrame:
    """按 ann_date 逐个日期获取 forecast / express 的公告日期。"""
    all_events = []
    for et in event_types:
        api = et  # "forecast" or "express"
        for d in dates:
            try:
                df = tushare_call(api, {"ann_date": d})
                if df.empty or "ts_code" not in df.columns or "ann_date" not in df.columns:
                    continue
                df = df[["ts_code", "ann_date"]].copy()
                df["event_type"] = et
                all_events.append(df)
            except Exception as e:
                print(f"[recall_advanced] Failed to fetch {et} {d}: {e}")
    if not all_events:
        return pd.DataFrame(columns=["ts_code", "ann_date", "event_type"])
    return pd.concat(all_events, ignore_index=True)


def apply_earnings_filter(df: pd.DataFrame, event_types: list, exclude_days: int) -> pd.DataFrame:
    """剔除未来 exclude_days 个交易日内有 forecast/express 公告的股票。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    dates = sorted(df["date"].dt.strftime("%Y%m%d").unique().tolist())
    events = fetch_earnings_events(dates, event_types)
    if events.empty:
        print("[recall_advanced] No earnings events fetched, skip filter")
        return df

    events["ann_date"] = pd.to_datetime(events["ann_date"], format="%Y%m%d", errors="coerce")
    events = events[events["ann_date"].notna()].copy()

    trade_dates = _get_trade_dates(df["date"].min(), df["date"].max() + pd.Timedelta(days=60))

    def get_days(row):
        code = row["ts_code"]
        d = row["date"]
        sub = events[events["ts_code"] == code]
        future = sub[sub["ann_date"] > d]
        if future.empty:
            return np.nan
        next_date = future["ann_date"].min()
        return _compute_trade_day_diff(trade_dates, d, next_date)

    df["days_to_event"] = df.apply(get_days, axis=1)
    mask = df["days_to_event"].isna() | (df["days_to_event"] > exclude_days)
    kept = df[mask].copy()
    n_dropped = len(df) - len(kept)
    if n_dropped:
        print(f"[recall_advanced] Earnings filter dropped {n_dropped} rows "
              f"({n_dropped/len(df):.1%}) with {event_types} event within {exclude_days} trade days")
    return kept.drop(columns=["days_to_event"])


def walk_forward_advanced(df: pd.DataFrame, feature_cols: list,
                          train_window_months: int = 12,
                          model_type: str = "lightgbm",
                          target: str = "excess_return",
                          static_filters: dict = None,
                          dynamic_threshold: bool = False,
                          dynamic_metric: str = "win_rate",
                          negative_model: bool = False,
                          negative_filter_ratio: float = 0.0,
                          earnings_event_types: list = None,
                          earnings_exclude_days: int = None) -> pd.DataFrame:
    """
    滚动训练并应用高级召回层过滤。
    """
    df = df.sort_values("date").copy()
    df["year_month"] = df["date"].dt.to_period("M")
    months = sorted(df["year_month"].unique())

    all_preds = []
    for i, test_month in enumerate(months):
        train_months = months[max(0, i - train_window_months):i]
        if len(train_months) < 3:
            continue

        train_df = df[df["year_month"].isin(train_months)].copy()
        test_df = df[df["year_month"] == test_month].copy()
        if train_df.empty or test_df.empty:
            continue

        val_month = train_months[-1]
        tr = train_df[train_df["year_month"] != val_month].sort_values("date")
        val = train_df[train_df["year_month"] == val_month].sort_values("date")

        # ---- 静态过滤（用于训练与最终预测）----
        if static_filters:
            tr = _apply_static_filters(tr, static_filters)
            val = _apply_static_filters(val, static_filters)
            test_df = _apply_static_filters(test_df, static_filters)

        # ---- 业绩预告事件过滤 ----
        if earnings_event_types and earnings_exclude_days is not None:
            tr = apply_earnings_filter(tr, earnings_event_types, earnings_exclude_days)
            val = apply_earnings_filter(val, earnings_event_types, earnings_exclude_days)
            test_df = apply_earnings_filter(test_df, earnings_event_types, earnings_exclude_days)

        if tr.empty or test_df.empty:
            continue

        # ---- 训练排序模型（回归）----
        try:
            rank_model = _train_model_for_fold(tr, val, feature_cols, model_type, target, "regression")
        except Exception as e:
            print(f"[recall_advanced] rank model failed for {test_month}: {e}")
            continue

        # ---- 负样本模型（可选）----
        neg_model = None
        if negative_model and negative_filter_ratio > 0:
            y_tr_neg = _make_bottom20_label(tr)
            y_val_neg = _make_bottom20_label(val)
            try:
                neg_model = _train_model_for_fold(
                    tr.assign(is_bottom20=y_tr_neg),
                    val.assign(is_bottom20=y_val_neg),
                    feature_cols, model_type, "is_bottom20", "binary"
                )
            except Exception as e:
                print(f"[recall_advanced] negative model failed for {test_month}: {e}")

        # ---- 动态阈值校准（在验证集上搜索）----
        chosen_vol_max, chosen_mv_min = 1.0, 0.0
        if dynamic_threshold:
            # 用排序模型预测验证集
            X_val = val[feature_cols].values
            if model_type == "xgboost":
                import xgboost as xgb
                val["predicted"] = rank_model.predict(xgb.DMatrix(X_val))
            else:
                val["predicted"] = rank_model.predict(X_val, num_iteration=rank_model.best_iteration)

            grid = [(vol, mv) for vol in [0.90, 0.95, 1.0] for mv in [0.0, 0.05, 0.1, 0.2]]
            best_score = -999.0
            for combo in grid:
                score = _score_threshold_combo(val, combo, dynamic_metric)
                if score > best_score:
                    best_score = score
                    chosen_vol_max, chosen_mv_min = combo
            print(f"[recall_advanced] {test_month} dynamic threshold: vol<={chosen_vol_max}, mv>={chosen_mv_min} "
                  f"({dynamic_metric}={best_score:.3f})")

        # ---- 预测测试集 ----
        X_test = test_df[feature_cols].values
        if model_type == "xgboost":
            import xgboost as xgb
            test_df["predicted"] = rank_model.predict(xgb.DMatrix(X_test))
        else:
            test_df["predicted"] = rank_model.predict(X_test, num_iteration=rank_model.best_iteration)

        # 应用动态阈值
        test_df = test_df[(test_df["volatility_20"] <= chosen_vol_max) &
                          (test_df["total_mv"] >= chosen_mv_min)].copy()

        # 应用负样本过滤
        if neg_model is not None:
            if model_type == "xgboost":
                test_df["neg_prob"] = neg_model.predict(xgb.DMatrix(X_test))
            else:
                test_df["neg_prob"] = neg_model.predict(X_test, num_iteration=neg_model.best_iteration)
            test_df["neg_rank"] = test_df.groupby("date")["neg_prob"].rank(pct=True, ascending=False)
            n_before = len(test_df)
            test_df = test_df[test_df["neg_rank"] > negative_filter_ratio].copy()
            if len(test_df) < n_before:
                print(f"[recall_advanced] {test_month} negative filter dropped {n_before - len(test_df)} rows")

        pred_df = test_df[["date", "ts_code", "excess_return", "stock_return",
                           "benchmark_return", "industry", "predicted"]].copy()
        pred_df["train_end_month"] = str(train_months[-1])
        all_preds.append(pred_df)

    if not all_preds:
        return pd.DataFrame()
    return pd.concat(all_preds, ignore_index=True)


def parse_filters(raw: str) -> dict:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--model-type", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--train-window-months", type=int, default=12)
    parser.add_argument("--target", default="excess_return")
    parser.add_argument("--filters", default="",
                        help="静态召回规则，例如 volatility_20:0:0.95,total_mv:0.05:1")
    parser.add_argument("--dynamic-threshold", action="store_true",
                        help="启用动态 vol/mv 阈值校准")
    parser.add_argument("--dynamic-metric", choices=["win_rate", "sharpe", "avg_excess"], default="win_rate")
    parser.add_argument("--negative-model", action="store_true",
                        help="启用负样本识别模型过滤")
    parser.add_argument("--negative-filter-ratio", type=float, default=0.1,
                        help="剔除预测为 bottom20 概率最高的前 X 比例")
    parser.add_argument("--earnings-event-types", default="",
                        help="业绩预告类型：forecast,express，多个用逗号分隔")
    parser.add_argument("--earnings-exclude-days", type=int, default=None,
                        help="剔除未来 N 个交易日内有业绩预告的股票")
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    df = load_dataset(args.horizon, args.dataset)
    feature_cols = get_feature_cols(df)
    static_filters = parse_filters(args.filters)

    earnings_types = [x.strip() for x in args.earnings_event_types.split(",") if x.strip()] or None

    pred_df = walk_forward_advanced(
        df, feature_cols,
        train_window_months=args.train_window_months,
        model_type=args.model_type,
        target=args.target,
        static_filters=static_filters,
        dynamic_threshold=args.dynamic_threshold,
        dynamic_metric=args.dynamic_metric,
        negative_model=args.negative_model,
        negative_filter_ratio=args.negative_filter_ratio,
        earnings_event_types=earnings_types,
        earnings_exclude_days=args.earnings_exclude_days,
    )

    if pred_df.empty:
        print("[recall_advanced] No predictions generated")
        return

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PRED_DIR / args.output_name
    pred_df.to_parquet(output_path, index=False)
    print(f"[recall_advanced] Saved {len(pred_df)} predictions to {output_path}")

    top20 = pred_df.groupby("date").apply(lambda g: g.sort_values("predicted", ascending=False).head(20),
                                          include_groups=False)
    if not top20.empty:
        print(f"\nDiagnostic: top20 avg excess={top20['excess_return'].mean():+.4f}, "
              f"positive ratio={(top20['excess_return'] > 0).mean():.2%}")


if __name__ == "__main__":
    main()
