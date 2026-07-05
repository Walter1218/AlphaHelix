"""
AKShare 事件风险过滤器

1. 加载 GBDT 预测结果，取出每期 top-k 候选股票；
2. 用 AKShare 抓取这些股票在回测区间内的公告（`stock_individual_notice_report`），本地缓存；
3. 基于公告标题关键词计算事件风险分；
4. 剔除/降权高风险股票，重新评估组合绩效。

注意防穿越：T 日决策只能使用 T 日及之前发布的公告。
"""
import sys
import os
import argparse
import re
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import akshare as ak

# 尝试关闭 akshare 内部的 tqdm 进度条
os.environ["TUNE_DISABLE_STDERR"] = "1"

CACHE_PATH = Path("memory/factors/announcements_akshare.parquet")

NEGATIVE_KEYWORDS = [
    "诉讼", "仲裁", "处罚", "罚款", "监管", "警示", "立案", "调查", "退市",
    "亏损", "预亏", "减持", "质押", "违约", "债务", "查封", "冻结", "清算",
    "破产", "重整", "收购失败", "撤销", "终止", "关注函", "问询函", "监管函",
    "警示函", "责令改正", "内幕交易", "操纵市场", "行政处罚", "市场禁入",
    "失信被执行人", "高风险", "重大风险", "业绩下滑", "下滑", "下降",
]

POSITIVE_KEYWORDS = [
    "增持", "回购", "预增", "中标", "签约", "重大合同", "收购", "重组",
    "股权激励", "分红", "派息", "转正", "扭亏", "批复", "核准", "通过",
    "合作协议", "战略合作", "重大项目", "新产品", "突破", "获批",
]


def ts_code_to_symbol(ts_code: str) -> str:
    """000001.SZ -> 000001"""
    return ts_code.split(".")[0]


def _fetch_one_symbol(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        df = ak.stock_individual_notice_report(
            security=code, symbol="全部",
            begin_date=start_date, end_date=end_date,
        )
        if df.empty:
            return pd.DataFrame()
        df = df[["代码", "公告标题", "公告类型", "公告日期"]].copy()
        df["symbol"] = code
        return df
    except Exception as e:
        print(f"[akshare_event_risk] Error fetching {code}: {e}")
        return pd.DataFrame()


def fetch_announcements(codes: list, start_date: str, end_date: str,
                        use_cache: bool = True, max_workers: int = 5) -> pd.DataFrame:
    """并发抓取或加载公告数据。codes 为 symbol（不含交易所后缀）。"""
    if use_cache and CACHE_PATH.exists():
        print(f"[akshare_event_risk] Loading cached announcements from {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH)

    records = []
    total = len(codes)
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one_symbol, code, start_date, end_date): code
                   for code in codes}
        for future in as_completed(futures):
            completed += 1
            if completed % 20 == 0 or completed == total:
                print(f"[akshare_event_risk] Fetched {completed}/{total} symbols")
            df = future.result()
            if not df.empty:
                records.append(df)

    if not records:
        return pd.DataFrame(columns=["symbol", "代码", "公告标题", "公告类型", "公告日期"])

    all_df = pd.concat(records, ignore_index=True)
    all_df["公告日期"] = pd.to_datetime(all_df["公告日期"], errors="coerce")
    all_df = all_df.dropna(subset=["公告日期"])
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(CACHE_PATH, index=False)
    print(f"[akshare_event_risk] Cached {len(all_df)} announcements to {CACHE_PATH}")
    return all_df


def score_title(title: str) -> tuple:
    """基于标题关键词计算 (negative_count, positive_count)。"""
    t = str(title)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in t)
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in t)
    return neg, pos


def build_event_risk_scores(pred_df: pd.DataFrame, announcements: pd.DataFrame,
                            lookback_days: int = 10) -> pd.DataFrame:
    """为每个预测样本计算事件风险分。"""
    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    pred_df["symbol"] = pred_df["ts_code"].apply(ts_code_to_symbol)

    ann = announcements.copy()
    ann["公告日期"] = pd.to_datetime(ann["公告日期"])
    ann["neg"], ann["pos"] = zip(*ann["公告标题"].apply(score_title))

    # 按 symbol + 日期聚合：找到每个 (symbol, 公告日期) 的累计正负关键词数
    daily = ann.groupby(["symbol", "公告日期"]).agg(
        neg_sum=("neg", "sum"),
        pos_sum=("pos", "sum"),
        n_ann=("公告标题", "size"),
    ).reset_index()

    # 为每个预测样本，累加 [date - lookback_days, date] 窗口内的公告
    scores = []
    grouped = daily.groupby("symbol")
    for _, row in pred_df.iterrows():
        sym = row["symbol"]
        d = row["date"]
        start = d - timedelta(days=lookback_days)
        sub = grouped.get_group(sym) if sym in grouped.groups else pd.DataFrame()
        if sub.empty:
            scores.append({"event_risk_score": 0.0, "event_neg": 0, "event_pos": 0, "event_n": 0})
            continue
        mask = (sub["公告日期"] >= start) & (sub["公告日期"] <= d)
        window = sub[mask]
        neg = int(window["neg_sum"].sum())
        pos = int(window["pos_sum"].sum())
        score = neg - pos
        scores.append({
            "event_risk_score": score,
            "event_neg": neg,
            "event_pos": pos,
            "event_n": int(window["n_ann"].sum()),
        })

    scores_df = pd.DataFrame(scores)
    return pd.concat([pred_df.reset_index(drop=True), scores_df], axis=1)


def evaluate_with_filter(pred_df: pd.DataFrame, max_positions: int = 20,
                         risk_threshold: int = 2):
    """评估 baseline 和事件过滤后的组合。"""
    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.strftime("%Y%m%d")

    base_rows = []
    filter_rows = []
    for d, g in pred_df.groupby("date"):
        g = g.sort_values("predicted", ascending=False)
        base_top = g.head(max_positions)
        base_rows.append({"date": d, "excess": base_top["excess_return"].mean()})

        # 过滤：排除 event_risk_score >= threshold 的股票，用后续低分股票补齐
        low_risk = g[g["event_risk_score"] < risk_threshold]
        selected = low_risk.head(max_positions)
        if selected.empty:
            filter_rows.append({"date": d, "excess": 0.0, "n": 0})
        else:
            filter_rows.append({
                "date": d,
                "excess": selected["excess_return"].mean(),
                "n": len(selected),
            })

    def metrics(rows):
        df = pd.DataFrame(rows).sort_values("date")
        df["cum"] = (1 + df["excess"]).cumprod() - 1
        return {
            "avg_excess": float(df["excess"].mean()),
            "cum_excess": float(df["cum"].iloc[-1]),
            "win_rate": float((df["excess"] > 0).mean()),
        }

    return metrics(base_rows), metrics(filter_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="memory/predictions/predictions_h10_walkforward_excess_return_regression.parquet")
    parser.add_argument("--top-k", type=int, default=50, help="用于抓取公告的候选池深度")
    parser.add_argument("--max-symbols", type=int, default=None, help="只抓取出现频率最高的前 N 个 symbol，控制总耗时")
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--risk-threshold", type=int, default=2)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    pred = pd.read_parquet(args.pred)
    pred["date"] = pd.to_datetime(pred["date"])

    # 取每日期预测得分 top-k 的候选股票
    from collections import Counter
    code_freq = Counter()
    for _, g in pred.groupby("date"):
        top = g.sort_values("predicted", ascending=False).head(args.top_k)
        code_freq.update(top["ts_code"].apply(ts_code_to_symbol).tolist())

    if args.max_symbols and len(code_freq) > args.max_symbols:
        top_codes = [code for code, _ in code_freq.most_common(args.max_symbols)]
        coverage = sum(code_freq[c] for c in top_codes) / sum(code_freq.values())
        print(f"[akshare_event_risk] Top-{args.max_symbols} frequent symbols cover {coverage:.1%} of top-{args.top_k} slots")
    else:
        top_codes = sorted(code_freq.keys())
    print(f"[akshare_event_risk] Symbols to fetch: {len(top_codes)}")

    start = (pred["date"].min() - timedelta(days=60)).strftime("%Y%m%d")
    end = (pred["date"].max() + timedelta(days=10)).strftime("%Y%m%d")

    announcements = fetch_announcements(top_codes, start, end, use_cache=not args.no_cache, max_workers=args.workers)
    print(f"[akshare_event_risk] Total announcements: {len(announcements)}")

    pred_scored = build_event_risk_scores(pred, announcements, lookback_days=args.lookback_days)

    # 风险分布
    print("\n=== Event Risk Distribution ===")
    print(pred_scored["event_risk_score"].describe())
    print("High risk ratio (>0):", (pred_scored["event_risk_score"] > 0).mean())

    base, filt = evaluate_with_filter(pred_scored, args.max_positions, args.risk_threshold)
    print("\n=== Portfolio Comparison (gross, equal-weight) ===")
    print(f"Baseline  avg={base['avg_excess']:.4f}  cum={base['cum_excess']:.4f}  win={base['win_rate']:.1%}")
    print(f"Filtered  avg={filt['avg_excess']:.4f}  cum={filt['cum_excess']:.4f}  win={filt['win_rate']:.1%}")

    out_path = Path("memory/factors/predictions_event_risk.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_scored.to_parquet(out_path, index=False)
    print(f"\n[akshare_event_risk] Saved scored predictions to {out_path}")


if __name__ == "__main__":
    main()
