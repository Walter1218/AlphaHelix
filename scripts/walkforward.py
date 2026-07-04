"""
AlphaHelix Walk-forward 回测脚本（预测型持仓版）

核心变化：
- 每 Δ 个交易日做一次再平衡；
- 用分 regime 在线二分类模型预测候选股/持仓股未来 Δ 天上涨概率；
- 概率 > threshold 才入选，否则空仓；
- 模型在标签暴露后逐样本在线更新。
"""
import sys
import os
import json
import argparse
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import get_trade_calendar, get_trade_date_after
from screen import screen
from evaluate import get_close_price
from market_regime import classify_regime
from online_predictor import RegimeModelManager, features_from_record, DEFAULT_FEATURE_NAMES
from _trace import trace_event, new_run

OUTPUT_DIR = Path("memory/eval")
SNAPSHOT_DIR = Path("memory/stock")
MODEL_DIR = Path("memory/models")

import warnings
warnings.filterwarnings("ignore")


def get_rebalance_dates(start_date: str, end_date: str, delta_days: int) -> list:
    """从 start_date 开始，每隔 delta_days 个交易日取一个再平衡日。"""
    cal = get_trade_calendar("SSE", start_date, end_date)
    cal = cal[cal["is_open"].astype(int) == 1].copy()
    dates = sorted(cal["cal_date"].astype(str).tolist())
    if not dates:
        return []

    rebalance = []
    idx = 0
    while idx < len(dates):
        t = dates[idx]
        # 必须能拿到 T+Δ 的收盘价才纳入
        try:
            t_next = get_trade_date_after(t, days=delta_days)
        except Exception:
            break
        if t_next > end_date:
            break
        rebalance.append(t)
        idx += delta_days
    return rebalance


def compute_period_return(holdings: list, start_date: str, end_date: str) -> dict:
    """计算持仓组合在 [start_date, end_date] 的收益与方向准确率。"""
    returns = []
    valid_holdings = []
    for ts_code in holdings:
        try:
            p0 = get_close_price(ts_code, start_date)
            p1 = get_close_price(ts_code, end_date)
            r = p1 / p0 - 1
            returns.append(r)
            valid_holdings.append(ts_code)
        except Exception:
            continue

    if not returns:
        return {"portfolio_return": 0.0, "direction_accuracy": float("nan"), "n": 0}

    returns = np.array(returns)
    return {
        "portfolio_return": float(np.mean(returns)),
        "direction_accuracy": float(np.mean(returns > 0)),
        "n": len(returns),
        "holdings": valid_holdings,
        "returns": returns.tolist(),
    }


def update_model_with_period(manager: RegimeModelManager, regime: str,
                             df_pass2_full: pd.DataFrame, start_date: str, end_date: str):
    """用本期所有 Pass2 候选样本更新对应 regime 的模型。"""
    if df_pass2_full is None or df_pass2_full.empty:
        return

    for _, row in df_pass2_full.iterrows():
        ts_code = row.get("ts_code")
        if not ts_code:
            continue
        try:
            p0 = get_close_price(ts_code, start_date)
            p1 = get_close_price(ts_code, end_date)
            label = 1 if p1 > p0 else 0
        except Exception:
            continue

        x = features_from_record(row.to_dict(), manager.feature_names)
        manager.update(regime, x, label)


def aggregate_results(results: list) -> dict:
    """汇总多期回测结果。"""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "No valid evaluation results"}

    traded = [r for r in valid if r.get("n", 0) > 0]
    cash_periods = len(valid) - len(traded)

    portfolio_returns = [r["portfolio_return"] for r in valid]
    excess_returns = [r["excess_return"] for r in valid]
    direction_accuracies = [r["direction_accuracy"] for r in traded if not math.isnan(r["direction_accuracy"])]

    cumulative_portfolio = float(np.prod([1 + x for x in portfolio_returns]) - 1)
    benchmark_returns = [r["benchmark_return"] for r in valid]
    cumulative_benchmark = float(np.prod([1 + x for x in benchmark_returns]) - 1)
    cumulative_excess = cumulative_portfolio - cumulative_benchmark

    turnovers = [r.get("turnover", 0.0) for r in valid]

    return {
        "periods": len(valid),
        "cash_periods": cash_periods,
        "traded_periods": len(traded),
        "avg_portfolio_return": round(float(np.mean(portfolio_returns)), 6),
        "avg_excess_return": round(float(np.mean(excess_returns)), 6),
        "avg_direction_accuracy": round(float(np.mean(direction_accuracies)) if direction_accuracies else 0.0, 4),
        "win_rate_excess": round(float(np.mean([e > 0 for e in excess_returns])), 4),
        "cumulative_portfolio_return": round(cumulative_portfolio, 6),
        "cumulative_benchmark_return": round(cumulative_benchmark, 6),
        "cumulative_excess_return": round(cumulative_excess, 6),
        "avg_turnover": round(float(np.mean(turnovers)), 4),
        "periods_detail": valid,
    }


def main():
    parser = argparse.ArgumentParser(description="AlphaHelix predictive walk-forward backtest")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--strategy", default="regime", help="Screening strategy")
    parser.add_argument("--rebalance-days", type=int, default=5, help="Rebalance interval in trading days")
    parser.add_argument("--max-positions", type=int, default=10, help="Max number of holdings")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum predicted up probability to hold")
    parser.add_argument("--burn-in-samples", type=int, default=200, help="Burn-in samples per regime before trading")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Online learning rate")
    parser.add_argument("--l2-reg", type=float, default=1e-1, help="L2 regularization")
    parser.add_argument("--universe-size", type=int, default=None, help="Override screen.py UNIVERSE_SAMPLE")
    parser.add_argument("--skip-st-check", action="store_true", help="Skip historical ST check for speed")
    parser.add_argument("--model-path", default=None, help="Path to load existing model state")
    parser.add_argument("--save-model", default=None, help="Path to save final model state")
    parser.add_argument("--progress-file", default=None, help="Write ongoing progress JSON")
    args = parser.parse_args()

    if args.universe_size is not None:
        os.environ["AH_UNIVERSE_SAMPLE"] = str(args.universe_size)
    if args.skip_st_check:
        os.environ["AH_SKIP_ST_CHECK"] = "1"
    os.environ["AH_BACKTEST_MODE"] = "1"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    run_id = new_run()
    rebalance_dates = get_rebalance_dates(args.start, args.end, args.rebalance_days)
    if len(rebalance_dates) < 2:
        print("[walkforward] ERROR: Not enough rebalance dates in range")
        return

    trace_event(
        "walkforward.start",
        {
            "inputs": {
                "start": args.start,
                "end": args.end,
                "strategy": args.strategy,
                "rebalance_days": args.rebalance_days,
                "max_positions": args.max_positions,
                "threshold": args.threshold,
                "burn_in_samples": args.burn_in_samples,
                "learning_rate": args.learning_rate,
                "l2_reg": args.l2_reg,
                "run_id": run_id,
                "rebalance_dates": rebalance_dates,
            }
        },
        date=rebalance_dates[0],
        strategy=args.strategy,
    )

    print(f"[walkforward] Running {len(rebalance_dates)} rebalance periods from {args.start} to {args.end}")
    print(f"[walkforward] Strategy={args.strategy}, Δ={args.rebalance_days}, max_positions={args.max_positions}, threshold={args.threshold}")

    # 初始化或加载模型
    model_path = args.model_path or str(MODEL_DIR / "online_regime_models.json")
    if args.model_path and Path(args.model_path).exists():
        manager = RegimeModelManager.load(args.model_path)
        print(f"[walkforward] Loaded model state from {args.model_path}")
    else:
        manager = RegimeModelManager(
            feature_names=DEFAULT_FEATURE_NAMES,
            window_size=120,
            lr=args.learning_rate,
            l2_reg=args.l2_reg,
            burn_in_samples=args.burn_in_samples,
        )

    results = []
    current_holdings = []
    start_time = datetime.now()

    for idx, t in enumerate(rebalance_dates, 1):
        period_start = datetime.now()
        t_next = get_trade_date_after(t, days=args.rebalance_days)
        print(f"[walkforward] [{idx}/{len(rebalance_dates)}] {t} -> {t_next} ...", end=" ", flush=True)

        try:
            regime_info = classify_regime(t)
            regime = regime_info.get("regime", "range")

            # 选股：返回所有超过 threshold 的候选（不限制数量，由 walkforward 做最终截断）
            records, df_pass2_full = screen(
                t, args.strategy,
                top_n=999,
                return_full=True,
                manager=manager,
                threshold=args.threshold,
                max_positions=999,
            )

            # 只有在该 regime burn-in 完成后才交易
            trading_enabled = manager.is_ready(regime)

            if trading_enabled:
                candidate_dict = {r["ts_code"]: r.get("predicted_up_prob", 0.0) for r in records}
                old_set = set(current_holdings)

                # 保留仍被模型看好的持仓 + 新候选
                combined = []
                for code in current_holdings:
                    if code in candidate_dict:
                        combined.append((code, candidate_dict[code]))
                for r in records:
                    code = r["ts_code"]
                    if code not in old_set:
                        combined.append((code, r.get("predicted_up_prob", 0.0)))

                combined.sort(key=lambda x: x[1], reverse=True)
                new_holdings = [code for code, _ in combined[:args.max_positions]]
            else:
                new_holdings = []

            sold = set(current_holdings) - set(new_holdings)
            bought = set(new_holdings) - set(current_holdings)
            turnover = (len(sold) + len(bought)) / (2 * args.max_positions) if args.max_positions > 0 else 0.0

            # 评估本期持仓收益
            eval_res = compute_period_return(new_holdings, t, t_next)
            portfolio_return = eval_res["portfolio_return"]

            try:
                bench_t0 = get_close_price("000300.SH", t)
                bench_t1 = get_close_price("000300.SH", t_next)
                benchmark_return = bench_t1 / bench_t0 - 1
            except Exception:
                benchmark_return = 0.0

            excess_return = portfolio_return - benchmark_return

            res = {
                "date": t,
                "next_date": t_next,
                "regime": regime,
                "n": eval_res["n"],
                "holdings": eval_res.get("holdings", []),
                "portfolio_return": round(portfolio_return, 6),
                "benchmark_return": round(benchmark_return, 6),
                "excess_return": round(excess_return, 6),
                "direction_accuracy": round(eval_res["direction_accuracy"], 4) if eval_res["n"] > 0 else float("nan"),
                "turnover": round(turnover, 4),
                "trading_enabled": trading_enabled,
                "sample_counts": manager.sample_counts.copy(),
            }

            # 用本期全部 Pass2 候选更新模型（无论是否交易）
            update_model_with_period(manager, regime, df_pass2_full, t, t_next)

            # 保存模型状态（每 5 期）
            if idx % 5 == 0:
                save_path = args.save_model or str(MODEL_DIR / "online_regime_models.json")
                manager.save(save_path)

            current_holdings = new_holdings
            elapsed = (datetime.now() - period_start).total_seconds()
            print(f"portfolio={portfolio_return:+.2%} excess={excess_return:+.2%} dir={res['direction_accuracy']:.1%} holdings={eval_res['n']} ({elapsed:.0f}s)")

        except Exception as e:
            elapsed = (datetime.now() - period_start).total_seconds()
            print(f"EXCEPTION: {e} ({elapsed:.0f}s)")
            res = {"date": t, "next_date": t_next, "error": str(e)}

        results.append(res)

        if args.progress_file and idx % 5 == 0:
            Path(args.progress_file).parent.mkdir(parents=True, exist_ok=True)
            Path(args.progress_file).write_text(json.dumps({
                "periods": len(results),
                "latest": results[-1],
            }, ensure_ascii=False, indent=2))

    # 最终保存模型
    final_model_path = args.save_model or str(MODEL_DIR / "online_regime_models.json")
    manager.save(final_model_path)

    summary = aggregate_results(results)
    summary["run_id"] = run_id
    summary["inputs"] = vars(args)

    summary_path = OUTPUT_DIR / f"walkforward_{args.start}_{args.end}_{args.strategy}_d{args.rebalance_days}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    elapsed_total = (datetime.now() - start_time).total_seconds()
    print(f"\n[walkforward] Total elapsed: {elapsed_total/60:.1f} minutes")
    print(f"[walkforward] Summary saved to {summary_path}")

    print("\n=== Walk-forward Summary ===")
    print(f"Periods: {summary.get('periods', 0)} (cash: {summary.get('cash_periods', 0)}, traded: {summary.get('traded_periods', 0)})")
    print(f"Avg portfolio return: {summary.get('avg_portfolio_return', 0):+.2%}")
    print(f"Avg excess return: {summary.get('avg_excess_return', 0):+.2%}")
    print(f"Avg direction accuracy: {summary.get('avg_direction_accuracy', 0):.1%}")
    print(f"Win rate (excess > 0): {summary.get('win_rate_excess', 0):.1%}")
    print(f"Avg turnover: {summary.get('avg_turnover', 0):.1%}")
    print(f"Cumulative portfolio return: {summary.get('cumulative_portfolio_return', 0):+.2%}")
    print(f"Cumulative benchmark return: {summary.get('cumulative_benchmark_return', 0):+.2%}")
    print(f"Cumulative excess return: {summary.get('cumulative_excess_return', 0):+.2%}")

    trace_event(
        "walkforward.end",
        {"outputs": summary},
        date=rebalance_dates[-1],
        strategy=args.strategy,
    )


if __name__ == "__main__":
    main()
