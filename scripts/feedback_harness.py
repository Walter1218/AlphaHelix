"""
AlphaHelix Feedback Harness
把 walk-forward / 每日选股的时效性结果反哺回系统：
1. 计算 factor IC
2. 跟踪策略滚动表现
3. 优化因子权重
4. 生成 prompt 自适应风险/风格提示

用法：
    python scripts/feedback_harness.py \
        --dates 20250127,20250228,20250331,20250430,20250530,20260430,20260529,20260615 \
        --start 20250101 --end 20260615 \
        --horizon 10
"""
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_ic import compute_ic, load_snapshot, load_eval
from strategy_tracker import load_strategy_summary, compute_strategy_weights
from weight_optimizer import optimize_weights, DEFAULT_BASE_WEIGHTS

WEIGHTS_DIR = Path("memory/weights")
PROMPT_DIR = Path("memory/prompt_adaptations")


def generate_prompt_adaptation(dates: list, strategy: str, horizon: int, start: str, end: str) -> str:
    """根据近期表现和因子 IC 生成 prompt 自适应提示。"""
    lines = ["# AlphaHelix Prompt Adaptations", f"\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f"Based on dates: {', '.join(dates)}", ""]

    # 1. 最近一期的表现
    latest_date = max(dates)
    try:
        eval_data = load_eval(latest_date, strategy, horizon)
        if "error" not in eval_data:
            lines.append(f"## Latest Period ({latest_date})")
            lines.append(f"- Portfolio return: {eval_data['portfolio_return']:+.2%}")
            lines.append(f"- Excess return: {eval_data['excess_return']:+.2%}")
            lines.append(f"- Direction accuracy: {eval_data['direction_accuracy']:.0%}")
            lines.append(f"- Max drawdown: {eval_data.get('portfolio_max_drawdown', 0):.2%}")
            lines.append("")
    except Exception:
        pass

    # 2. 因子 IC 提示
    ic_values = {}
    for d in dates:
        path = Path("memory/factor_ic") / f"{d}_{strategy}_h{horizon}.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if k in ("n", "date", "error"):
                continue
            ic_values.setdefault(k, []).append(v)

    if ic_values:
        avg_ic = {k: float(np.mean(v)) for k, v in ic_values.items() if v}
        positive = sorted([(k, v) for k, v in avg_ic.items() if v > 0.05], key=lambda x: -x[1])
        negative = sorted([(k, v) for k, v in avg_ic.items() if v < -0.05], key=lambda x: x[1])

        lines.append("## Factor Effectiveness (rank IC)")
        if positive:
            lines.append("近期有效的因子（可适当侧重）：")
            for k, v in positive[:3]:
                lines.append(f"- {k}: IC={v:+.3f}")
        if negative:
            lines.append("近期失效的因子（应谨慎使用）：")
            for k, v in negative[:3]:
                lines.append(f"- {k}: IC={v:+.3f}")
        lines.append("")

    # 3. 策略权重提示
    strategy_summary_path = Path("memory/strategy_tracker") / f"weights_{start}_{end}_h{horizon}.json"
    if strategy_summary_path.exists():
        with open(strategy_summary_path, "r", encoding="utf-8") as f:
            st = json.load(f)
        weights = st.get("strategy_weights", {}).get("weights", {})
        if weights:
            lines.append("## Strategy Allocation")
            for k, v in sorted(weights.items(), key=lambda x: -x[1]):
                lines.append(f"- {k}: {v:.1%}")
            top = max(weights, key=weights.get)
            lines.append(f"\n当前建议优先使用策略：{top}")
            lines.append("")

    # 4. 风险/仓位提示
    recent_returns = []
    recent_mdds = []
    for d in dates[-3:]:
        try:
            ed = load_eval(d, strategy, horizon)
            if "error" not in ed:
                recent_returns.append(ed["portfolio_return"])
                recent_mdds.append(ed.get("portfolio_max_drawdown", 0))
        except Exception:
            pass

    if recent_mdds and min(recent_mdds) < -0.08:
        lines.append("## Risk Alert")
        lines.append(f"- 近期最大回撤 {min(recent_mdds):.1%}，建议降低仓位、收紧止损、避免高波动标的。")
        lines.append("")

    if recent_returns and sum(1 for r in recent_returns if r < 0) >= 2:
        lines.append("- 近期连续亏损，建议减少激进动量暴露，优先基本面稳健的标的。")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="AlphaHelix feedback harness")
    parser.add_argument("--dates", required=True, help="Comma-separated backtest dates YYYYMMDD")
    parser.add_argument("--start", required=True, help="Walk-forward start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="Walk-forward end date YYYYMMDD")
    parser.add_argument("--horizon", type=int, default=10, help="Holding horizon")
    parser.add_argument("--strategy", default="momentum_value_hybrid", help="Base strategy for factor IC and prompt")
    parser.add_argument("--lr", type=float, default=0.5, help="Weight optimization learning rate")
    args = parser.parse_args()

    dates = sorted([d.strip() for d in args.dates.split(",") if d.strip()])
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    Path("memory/factor_ic").mkdir(parents=True, exist_ok=True)
    Path("memory/strategy_tracker").mkdir(parents=True, exist_ok=True)

    # 1. 计算 factor IC（使用 pooled 名称，作为通用因子有效性反馈）
    print("[harness] Computing factor IC ...")
    for d in dates:
        try:
            snapshot = load_snapshot(d)
            eval_result = load_eval(d, args.strategy, args.horizon)
            ic = compute_ic(snapshot, eval_result)
            out = Path("memory/factor_ic") / f"{d}_pooled_h{args.horizon}.json"
            out.write_text(json.dumps(ic, ensure_ascii=False, indent=2))
            print(f"  {d}: {len([k for k in ic if k not in ('n','date','error')])} factors")
        except Exception as e:
            print(f"  {d}: ERROR {e}")

    # 2. 策略跟踪（支持跨多个区间合并）
    print("[harness] Tracking strategy performance ...")
    summaries = {}
    for s in ["momentum_value_hybrid", "quality_growth", "contrarian", "event_driven", "regime"]:
        # 先尝试合并 2025 和 2026 两个已跑区间
        merged = {"monthly": []}
        for (s0, e0) in [("20250101", "20250531"), ("20260401", "20260615")]:
            part = load_strategy_summary(s0, e0, s, args.horizon)
            if "error" not in part:
                merged["monthly"].extend(part.get("monthly", []))
        if not merged["monthly"]:
            merged = load_strategy_summary(args.start, args.end, s, args.horizon)
        summaries[s] = merged
    st_result = {
        "start": args.start,
        "end": args.end,
        "horizon": args.horizon,
        "strategy_weights": compute_strategy_weights(summaries, lookback=None, temperature=1.0),
    }
    st_path = Path("memory/strategy_tracker") / f"weights_{args.start}_{args.end}_h{args.horizon}.json"
    st_path.write_text(json.dumps(st_result, ensure_ascii=False, indent=2))
    print(f"  Strategy weights saved to {st_path}")

    # 3. 优化因子权重（为所有策略都生成）
    print("[harness] Optimizing factor weights ...")
    for s in DEFAULT_BASE_WEIGHTS.keys():
        weights = optimize_weights(dates, s, args.horizon, args.lr, ic_strategy="pooled")
        if weights:
            out = WEIGHTS_DIR / f"{s}_latest.json"
            out.write_text(json.dumps({
                "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "based_on_dates": dates,
                "strategy": s,
                "horizon": args.horizon,
                "weights": weights,
            }, ensure_ascii=False, indent=2))
            print(f"  {s}: {out}")

    # 4. 生成 prompt 自适应提示
    print("[harness] Generating prompt adaptations ...")
    prompt_md = generate_prompt_adaptation(dates, args.strategy, args.horizon, args.start, args.end)
    prompt_path = PROMPT_DIR / "latest.md"
    prompt_path.write_text(prompt_md, encoding="utf-8")
    print(f"  Prompt adaptations saved to {prompt_path}")

    print("[harness] Done")


if __name__ == "__main__":
    main()
