"""
AlphaHelix Walk-forward 回测脚本
对指定区间内的多个交易日运行选股 + 评估，输出累计指标和月度报告。
"""
import sys
import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import get_trade_calendar, get_trade_date_after
from screen import screen, STRATEGIES
from evaluate import evaluate
from market_regime import classify_regime, regime_to_strategy
from _trace import trace_event, new_run
from online_weight_updater import load_rolling_weights, update_regime_weights

DEFAULT_STRATEGY = "momentum_value_hybrid"
DEFAULT_HORIZON = 10
OUTPUT_DIR = Path("memory/eval")
SNAPSHOT_DIR = Path("memory/stock")

# 抑制 numpy 等库的警告输出，避免污染日志
import warnings
warnings.filterwarnings("ignore")


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d")


def format_date(d: datetime) -> str:
    return d.strftime("%Y%m%d")


def get_rebalance_dates(start_date: str, end_date: str, freq: str = "monthly") -> list:
    """获取再平衡选股日列表。freq 支持 monthly（每月最后一个交易日）或 weekly（每周最后一个交易日）。"""
    cal = get_trade_calendar("SSE", start_date, end_date)
    cal = cal[cal["is_open"].astype(int) == 1].copy()
    cal["cal_date"] = pd.to_datetime(cal["cal_date"], format="%Y%m%d")
    cal = cal.sort_values("cal_date")

    if freq == "weekly":
        # 按自然周（周日结束）分组，取每周最后一个交易日
        cal["week"] = cal["cal_date"].dt.to_period("W-SUN")
        rebalance = cal.groupby("week")["cal_date"].last().dt.strftime("%Y%m%d").tolist()
    elif freq == "monthly":
        cal["ym"] = cal["cal_date"].dt.to_period("M")
        rebalance = cal.groupby("ym")["cal_date"].last().dt.strftime("%Y%m%d").tolist()
    else:
        raise ValueError(f"Unsupported freq: {freq}. Use 'monthly' or 'weekly'.")
    return rebalance


def _pick_from_candidate(c: dict, rank: int, factor_fields: list) -> dict:
    """将单个候选对象转换为 snapshot pick。"""
    pick = {
        "ts_code": c["ts_code"],
        "name": c["name"],
        "score": round(c["total_score"], 4),
        "rank": rank,
        "rationale": f"{c.get('industry', '')}, score {c['total_score']:.4f}",
        "confidence": "medium",
        "stop_loss": 0.0,
    }
    for f in factor_fields:
        if f in c and c[f] is not None:
            try:
                pick[f] = round(float(c[f]), 6)
            except (ValueError, TypeError):
                pass
    return pick


def build_snapshot(trade_date: str, candidates: list, horizon: int) -> dict:
    """根据 screen.py 输出构建 evaluate.py 可读取的 snapshot。保留因子值供 IC 计算。"""
    factor_fields = [
        "mom_5", "mom_20", "mom_60", "pe", "pb", "ps", "dv_ratio",
        "roe", "revenue_growth", "profit_growth", "ocf_growth",
        "net_mf_5d", "net_mf_20d", "net_mf_ratio",
        "avg_amount_20", "amount_ratio_5d", "volatility_20", "total_mv",
        "reversal_score", "sector_momentum", "relative_to_sector", "sector_mom5", "sector_amount_ratio",
        "forecast_type_score", "forecast_pchange_mid", "express_diluted_roe", "express_diluted_eps",
    ]
    picks = [_pick_from_candidate(c, i, factor_fields) for i, c in enumerate(candidates, 1)]
    return {
        "date": trade_date,
        "data_as_of": trade_date,
        "market_summary": f"Walk-forward snapshot for {trade_date}, horizon={horizon}.",
        "picks": picks,
        "risk_notes": ["Backtest only.", "AlphaHelix stands behind the rigor of its research methodology and data quality, but does not guarantee future returns due to market uncertainty."],
    }


def build_full_snapshot(trade_date: str, df_pass2: pd.DataFrame, horizon: int) -> dict:
    """构建包含 pass2 全部候选（约 80 只）的完整 snapshot，供离线权重优化使用。"""
    factor_fields = [
        "mom_5", "mom_20", "mom_60", "pe", "pb", "ps", "dv_ratio",
        "roe", "revenue_growth", "profit_growth", "ocf_growth",
        "net_mf_5d", "net_mf_20d", "net_mf_ratio",
        "avg_amount_20", "amount_ratio_5d", "volatility_20", "total_mv",
        "reversal_score", "sector_momentum", "relative_to_sector", "sector_mom5", "sector_amount_ratio",
        "forecast_type_score", "forecast_pchange_mid", "express_diluted_roe", "express_diluted_eps",
    ]
    picks = []
    for i, (_, row) in enumerate(df_pass2.iterrows(), 1):
        c = row.to_dict()
        picks.append(_pick_from_candidate(c, i, factor_fields))
    return {
        "date": trade_date,
        "data_as_of": trade_date,
        "market_summary": f"Full pass2 snapshot for {trade_date}, horizon={horizon}.",
        "picks": picks,
        "risk_notes": ["Backtest only.", "AlphaHelix stands behind the rigor of its research methodology and data quality, but does not guarantee future returns due to market uncertainty."],
    }


def resolve_strategy(trade_date: str, strategy: str) -> tuple:
    """解析策略参数，返回 (实际策略, regime信息)。"""
    if strategy == "regime":
        regime_info = classify_regime(trade_date)
        actual = regime_to_strategy(regime_info["regime"], available=list(STRATEGIES.keys()))
        return actual, regime_info
    if strategy not in STRATEGIES and strategy != "regime":
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(STRATEGIES.keys())} or 'regime'")
    return strategy, None


def eval_path_for(trade_date: str, strategy: str, horizon: int) -> Path:
    """生成区分策略的评估文件路径。"""
    if strategy == "regime":
        return OUTPUT_DIR / f"{trade_date}_regime_h{horizon}.json"
    return OUTPUT_DIR / f"{trade_date}_{strategy}_h{horizon}.json"


def run_single_period(trade_date: str, strategy: str, horizon: int, top_n: int,
                      resume: bool = True, online_update: bool = False,
                      pass1_weights: dict = None, pass2_weights: dict = None) -> dict:
    """运行单个选股日期的选股 + 评估。"""
    snapshot_path = SNAPSHOT_DIR / f"{trade_date}.json"
    eval_path = eval_path_for(trade_date, strategy, horizon)

    actual_strategy, regime_info = resolve_strategy(trade_date, strategy)

    # 恢复：若快照与评估均存在且有效，直接读取（online_update 模式下不恢复，因为要重新用动态权重选股）
    if resume and not online_update and snapshot_path.exists() and eval_path.exists():
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing["candidates"] = len(json.loads(snapshot_path.read_text(encoding="utf-8")).get("picks", []))
            existing["resumed"] = True
            existing.setdefault("strategy", actual_strategy)
            if regime_info:
                existing.setdefault("regime", regime_info["regime"])
            return existing
        except Exception:
            pass

    # 1. 选股（同时获取完整 pass2 池供离线优化）
    screen_result = screen(trade_date, actual_strategy, top_n, return_full=True,
                           pass1_weights_override=pass1_weights,
                           pass2_weights_override=pass2_weights)
    if isinstance(screen_result, tuple):
        candidates, df_pass2 = screen_result
    else:
        candidates = screen_result
        df_pass2 = pd.DataFrame()

    if not candidates:
        return {"date": trade_date, "error": "No candidates generated"}

    # 2. 写入 snapshot（兼容原路径 + 策略专属路径 + 完整 pass2 路径）
    snapshot = build_snapshot(trade_date, candidates, horizon)
    full_snapshot = build_full_snapshot(trade_date, df_pass2, horizon)
    snapshot_path = SNAPSHOT_DIR / f"{trade_date}.json"
    strategy_snapshot_path = SNAPSHOT_DIR / f"{trade_date}_{actual_strategy}.json"
    full_snapshot_path = SNAPSHOT_DIR / f"{trade_date}_{actual_strategy}_full.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_json = json.dumps(snapshot, ensure_ascii=False, indent=2)
    snapshot_path.write_text(snapshot_json)
    strategy_snapshot_path.write_text(snapshot_json)
    full_snapshot_path.write_text(json.dumps(full_snapshot, ensure_ascii=False, indent=2))

    # 3. 评估
    try:
        exit_date = get_trade_date_after(trade_date, days=horizon)
        # 简单检查 exit_date 是否超出今天（无法评估未来）
        if exit_date > datetime.now().strftime("%Y%m%d"):
            return {
                "date": trade_date,
                "exit_date": exit_date,
                "error": f"Exit date {exit_date} is in the future, cannot evaluate",
                "candidates": len(candidates),
            }
        result = evaluate(trade_date, horizon)
        result["candidates"] = len(candidates)
        result["strategy"] = actual_strategy
        if regime_info:
            result["regime"] = regime_info["regime"]
            result["regime_reason"] = regime_info["reason"]
        eval_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except Exception as e:
        return {"date": trade_date, "error": str(e), "candidates": len(candidates)}


def aggregate_results(results: list) -> dict:
    """汇总多期回测结果。"""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "No valid evaluation results"}

    portfolio_returns = [r["portfolio_return"] for r in valid]
    excess_returns = [r["excess_return"] for r in valid]
    direction_accuracies = [r["direction_accuracy"] for r in valid]
    top3_hit_rates = [r.get("top3_hit_rate", 0) for r in valid]
    mdds = [r.get("portfolio_max_drawdown", 0) for r in valid]

    # 累计组合收益（假设每月再平衡，收益连乘）
    cumulative_return = float(np.prod([1 + x for x in portfolio_returns]) - 1)
    benchmark_returns = [r["benchmark_return"] for r in valid]
    cumulative_benchmark = float(np.prod([1 + x for x in benchmark_returns]) - 1)
    cumulative_excess = cumulative_return - cumulative_benchmark

    return {
        "periods": len(valid),
        "skipped": len(results) - len(valid),
        "avg_portfolio_return": round(float(np.mean(portfolio_returns)), 6),
        "avg_excess_return": round(float(np.mean(excess_returns)), 6),
        "avg_direction_accuracy": round(float(np.mean(direction_accuracies)), 4),
        "avg_top3_hit_rate": round(float(np.mean(top3_hit_rates)), 4),
        "avg_max_drawdown": round(float(np.mean(mdds)), 6),
        "win_rate_excess": round(float(np.mean([e > 0 for e in excess_returns])), 4),
        "cumulative_portfolio_return": round(cumulative_return, 6),
        "cumulative_benchmark_return": round(cumulative_benchmark, 6),
        "cumulative_excess_return": round(cumulative_excess, 6),
        "monthly": valid,
    }


def main():
    parser = argparse.ArgumentParser(description="AlphaHelix walk-forward backtest")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help="Screening strategy; use 'regime' to switch by market regime")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Holding horizon in trading days")
    parser.add_argument("--top-n", type=int, default=10, help="Number of picks per period")
    parser.add_argument("--freq", default="monthly", choices=["monthly", "weekly"], help="Rebalance frequency")
    parser.add_argument("--universe-size", type=int, default=None, help="Override screen.py UNIVERSE_SAMPLE (smaller=faster)")
    parser.add_argument("--skip-st-check", action="store_true", help="Skip historical ST check for speed (not for production)")
    parser.add_argument("--no-resume", action="store_true", help="Re-run even if previous results exist")
    parser.add_argument("--progress-file", default=None, help="Write ongoing progress JSON to this file")
    parser.add_argument("--online-update", action="store_true", help="Enable walk-forward online learning with regime-conditional weights")
    parser.add_argument("--online-lookback", type=int, default=6, help="Rolling window size for online weight update")
    parser.add_argument("--online-lr", type=float, default=0.5, help="Learning rate for online weight update")
    args = parser.parse_args()

    if args.universe_size is not None:
        os.environ["AH_UNIVERSE_SAMPLE"] = str(args.universe_size)
    if args.skip_st_check:
        os.environ["AH_SKIP_ST_CHECK"] = "1"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    if args.online_update:
        Path("memory/weights").mkdir(parents=True, exist_ok=True)
    run_id = new_run()

    trade_dates = get_rebalance_dates(args.start, args.end, freq=args.freq)
    trace_event(
        "walkforward.start",
        {
            "inputs": {
                "start": args.start,
                "end": args.end,
                "strategy": args.strategy,
                "horizon": args.horizon,
                "top_n": args.top_n,
                "freq": args.freq,
                "universe_size": args.universe_size,
                "skip_st_check": args.skip_st_check,
                "online_update": args.online_update,
                "online_lookback": args.online_lookback,
                "online_lr": args.online_lr,
                "run_id": run_id,
                "trade_dates": trade_dates,
            }
        },
        date=trade_dates[0] if trade_dates else args.start,
        strategy=args.strategy,
    )
    print(f"[walkforward] Running {len(trade_dates)} periods from {args.start} to {args.end}")
    print(f"[walkforward] Strategy={args.strategy}, horizon={args.horizon}, top_n={args.top_n}")

    results = []
    start_time = datetime.now()
    # 在线学习：预加载各 regime 滚动权重
    current_weights = {}
    for idx, td in enumerate(trade_dates, 1):
        period_start = datetime.now()
        print(f"[walkforward] [{idx}/{len(trade_dates)}] {td} ...", end=" ", flush=True)

        # 解析该期 regime / 实际策略
        actual_strategy, regime_info = resolve_strategy(td, args.strategy)
        regime = regime_info["regime"] if regime_info else None
        strategy_key = actual_strategy

        # 在线学习：加载该 regime 的滚动权重
        pass1_weights, pass2_weights = None, None
        if args.online_update and regime:
            wkey = (strategy_key, regime)
            if wkey not in current_weights:
                current_weights[wkey] = load_rolling_weights(strategy_key, regime)
            pass1_weights = current_weights[wkey].get("pass1")
            pass2_weights = current_weights[wkey].get("pass2")

        try:
            res = run_single_period(
                td, args.strategy, args.horizon, args.top_n,
                resume=not args.no_resume,
                online_update=args.online_update,
                pass1_weights=pass1_weights,
                pass2_weights=pass2_weights,
            )
            elapsed = (datetime.now() - period_start).total_seconds()
            if "error" in res:
                print(f"ERROR: {res['error']} ({elapsed:.0f}s)")
            else:
                status = "resumed" if res.get("resumed") else "done"
                strategy_tag = f" strategy={res.get('strategy', '?')}"
                regime_tag = f" regime={res.get('regime', '?')}" if res.get('regime') else ""
                weight_tag = " online" if args.online_update else ""
                print(f"portfolio={res['portfolio_return']:+.2%} excess={res['excess_return']:+.2%} hit={res['direction_accuracy']:.0%}[{status}]{strategy_tag}{regime_tag}{weight_tag} ({elapsed:.0f}s)")
        except Exception as e:
            elapsed = (datetime.now() - period_start).total_seconds()
            print(f"EXCEPTION: {e} ({elapsed:.0f}s)")
            res = {"date": td, "error": str(e)}
        results.append(res)

        # 在线学习：用该期结果更新该 regime 滚动权重（仅用于下一期）
        if args.online_update and regime and "error" not in res:
            try:
                updated = update_regime_weights(
                    strategy_key, regime, args.horizon, td,
                    max_lookback=args.online_lookback,
                    learning_rate=args.online_lr,
                )
                current_weights[(strategy_key, regime)] = updated
            except Exception as e:
                print(f"[walkforward] online update failed for {td} {regime}: {e}")

        if "error" not in res:
            trace_event(
                "walkforward.period",
                {
                    "outputs": {
                        "portfolio_return": res.get("portfolio_return"),
                        "excess_return": res.get("excess_return"),
                        "direction_accuracy": res.get("direction_accuracy"),
                        "top3_hit_rate": res.get("top3_hit_rate"),
                        "benchmark_return": res.get("benchmark_return"),
                        "candidates": res.get("candidates"),
                        "actual_strategy": res.get("strategy"),
                        "regime": res.get("regime"),
                        "resumed": res.get("resumed", False),
                    }
                },
                date=td,
                strategy=res.get("strategy", args.strategy),
            )

        # 定期写出进度
        if args.progress_file:
            partial = aggregate_results(results)
            partial["start_date"] = args.start
            partial["end_date"] = args.end
            partial["current_period"] = td
            Path(args.progress_file).write_text(json.dumps(partial, ensure_ascii=False, indent=2))

    total_elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[walkforward] Total elapsed: {total_elapsed/60:.1f} minutes")

    summary = aggregate_results(results)
    summary["start_date"] = args.start
    summary["end_date"] = args.end
    summary["strategy"] = args.strategy
    summary["horizon"] = args.horizon
    summary["top_n"] = args.top_n

    out_path = OUTPUT_DIR / f"walkforward_{args.start}_{args.end}_{args.strategy}_h{args.horizon}_{args.freq}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[walkforward] Summary saved to {out_path}")

    trace_event(
        "walkforward.summary",
        {
            "outputs": {
                "periods": summary.get("periods"),
                "skipped": summary.get("skipped"),
                "avg_portfolio_return": summary.get("avg_portfolio_return"),
                "avg_excess_return": summary.get("avg_excess_return"),
                "avg_direction_accuracy": summary.get("avg_direction_accuracy"),
                "avg_top3_hit_rate": summary.get("avg_top3_hit_rate"),
                "cumulative_excess_return": summary.get("cumulative_excess_return"),
                "summary_path": str(out_path),
            }
        },
        date=args.end,
        strategy=args.strategy,
    )

    print("\n=== Walk-forward Summary ===")
    print(f"Periods: {summary.get('periods', 0)} (skipped: {summary.get('skipped', 0)})")
    print(f"Avg portfolio return: {summary.get('avg_portfolio_return', 0):+.2%}")
    print(f"Avg excess return: {summary.get('avg_excess_return', 0):+.2%}")
    print(f"Avg direction accuracy: {summary.get('avg_direction_accuracy', 0):.1%}")
    print(f"Avg Top3 hit rate: {summary.get('avg_top3_hit_rate', 0):.1%}")
    print(f"Win rate (excess > 0): {summary.get('win_rate_excess', 0):.1%}")
    print(f"Cumulative portfolio return: {summary.get('cumulative_portfolio_return', 0):+.2%}")
    print(f"Cumulative benchmark return: {summary.get('cumulative_benchmark_return', 0):+.2%}")
    print(f"Cumulative excess return: {summary.get('cumulative_excess_return', 0):+.2%}")


if __name__ == "__main__":
    main()
