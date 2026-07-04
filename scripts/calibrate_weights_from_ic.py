"""
AlphaHelix 基于 walk-forward 结果的因子 IC 权重校准

从 memory/eval/ 和 memory/stock/ 中读取历史选股与收益，
计算每个因子的 rank IC，然后生成新权重：
- 负 IC 因子权重置 0（A 股无法做空）
- 正 IC 因子按 IC 占比归一化

输出：memory/weights/{strategy}_ic.json，可被 screen.py 动态加载。
"""
import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weight_optimizer import DEFAULT_BASE_WEIGHTS

SNAPSHOT_DIR = Path("memory/stock")
WEIGHTS_DIR = Path("memory/weights")


def load_results(strategy: str = None):
    """加载 walk-forward 评估与快照。"""
    records = []
    for eval_path in sorted(Path("memory/eval").glob("202*_*_h*.json")):
        # 跳过汇总文件
        if "walkforward_" in eval_path.name:
            continue
        try:
            ev = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "error" in ev or ev.get("cash"):
            continue
        # regime 模式下实际策略会变化，这里不过滤实际策略，而是把所有期汇总
        if strategy and strategy != "regime" and ev.get("strategy") != strategy:
            continue
        date = ev["date"]
        snap_path = SNAPSHOT_DIR / f"{date}.json"
        if not snap_path.exists():
            continue
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        picks = snap.get("picks", [])
        ret_map = {det["ts_code"]: det["abs_return"] for det in ev.get("details", []) if "error" not in det}
        for p in picks:
            ts = p["ts_code"]
            if ts not in ret_map:
                continue
            rec = {"date": date, "ts_code": ts, "rank": p.get("rank"), "return": ret_map[ts], "strategy": ev.get("strategy")}
            for k, v in p.items():
                if k in ("ts_code", "name", "rationale", "confidence", "stop_loss", "rank", "score"):
                    continue
                try:
                    rec[k] = float(v)
                except Exception:
                    pass
            records.append(rec)
    return pd.DataFrame(records)


def compute_ic(df: pd.DataFrame) -> dict:
    """计算各因子平均 rank IC（按日期聚合）。"""
    factors = [c for c in df.columns if c not in ("date", "ts_code", "rank", "return", "strategy")]
    ic_mean = {}
    for f in factors:
        per_ic = []
        for _, g in df.groupby("date"):
            x = g[f].dropna()
            if len(x) < 3:
                continue
            y = g.loc[x.index, "return"]
            try:
                ic, _ = spearmanr(x, y)
                if not np.isnan(ic):
                    per_ic.append(ic)
            except Exception:
                continue
        if per_ic:
            ic_mean[f] = float(np.mean(per_ic))
    return ic_mean


def build_weights(ic_mean: dict, min_ic: float = 0.0) -> dict:
    """根据 IC 生成非负权重。"""
    positive = {f: max(0.0, ic - min_ic) for f, ic in ic_mean.items() if ic > min_ic}
    total = sum(positive.values())
    if total == 0:
        return {}
    return {f: round(w / total, 4) for f, w in positive.items()}


def _load_period(date: str, horizon: int = 10):
    """加载单期的选股与收益记录。"""
    eval_path = Path("memory/eval") / f"{date}_regime_h{horizon}.json"
    if not eval_path.exists():
        matches = list(Path("memory/eval").glob(f"{date}_*_h{horizon}.json"))
        if not matches:
            return None, None
        eval_path = matches[0]
    try:
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    if "error" in ev or ev.get("cash"):
        return None, None
    snap_path = SNAPSHOT_DIR / f"{date}.json"
    if not snap_path.exists():
        return None, None
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    return ev, snap


def compute_pass2_weights_for_dates(dates: list, min_ic: float = 0.0, horizon: int = 10, per_regime: bool = False):
    """
    根据给定日期列表的历史选股与收益，生成 pass2 权重。
    若 per_regime=True，返回 {regime: weights}；否则返回单个 weights dict。
    """
    records = []
    for date in dates:
        ev, snap = _load_period(date, horizon)
        if ev is None:
            continue
        regime = ev.get("regime")
        picks = snap.get("picks", [])
        ret_map = {det["ts_code"]: det["abs_return"] for det in ev.get("details", []) if "error" not in det}
        for p in picks:
            ts = p["ts_code"]
            if ts not in ret_map:
                continue
            rec = {"date": date, "ts_code": ts, "rank": p.get("rank"), "return": ret_map[ts], "regime": regime}
            for k, v in p.items():
                if k in ("ts_code", "name", "rationale", "confidence", "stop_loss", "rank", "score"):
                    continue
                try:
                    rec[k] = float(v)
                except Exception:
                    pass
            records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        return {} if not per_regime else {}

    if not per_regime:
        ic_mean = compute_ic(df)
        return build_weights(ic_mean, min_ic=min_ic)

    # per regime
    result = {}
    for regime, g in df.groupby("regime"):
        if len(g) < 10:
            continue
        ic_mean = compute_ic(g)
        w = build_weights(ic_mean, min_ic=min_ic)
        if w:
            result[regime] = w
    return result


def main():
    parser = argparse.ArgumentParser(description="Calibrate strategy weights from historical rank IC")
    parser.add_argument("--strategy", default=None, help="Filter by strategy; if None, pool all strategies")
    parser.add_argument("--min-ic", type=float, default=0.0, help="Ignore factors with IC below this threshold")
    parser.add_argument("--output", default=None, help="Output weights file name; default {strategy}_ic.json")
    parser.add_argument("--per-regime", action="store_true", help="Generate separate weights per market regime")
    parser.add_argument("--horizon", type=int, default=10, help="Holding horizon used in eval filenames")
    args = parser.parse_args()

    print("=" * 60)
    print("WARNING: This script produces weights based on a given set of")
    print("historical results. Using these weights to backtest the SAME")
    print("periods violates AGENTS.md C38 (no time travel / no in-sample)")
    print("optimization). Only use the output in walk-forward mode, where")
    print("each period's weights come from strictly prior data.")
    print("=" * 60)

    df = load_results(strategy=args.strategy)
    if df.empty:
        print("No data found")
        return

    print(f"Loaded {len(df)} picks across {df['date'].nunique()} dates")
    if args.strategy:
        print(f"Strategy filter: {args.strategy}")

    if args.per_regime:
        regime_weights = {}
        for regime, g in df.groupby("regime"):
            if len(g) < 10:
                continue
            ic_mean = compute_ic(g)
            new_weights = build_weights(ic_mean, min_ic=args.min_ic)
            regime_weights[regime] = {"ic_mean": ic_mean, "weights": new_weights}
            print(f"\nRegime: {regime} ({len(g)} picks)")
            print("Factor mean rank IC:")
            for f, ic in sorted(ic_mean.items(), key=lambda kv: kv[1], reverse=True):
                print(f"  {f:30s} {ic:+.4f}")
            print("New weights:")
            for f, w in sorted(new_weights.items(), key=lambda kv: kv[1], reverse=True):
                print(f"  {f:30s} {w:.4f}")

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        strategy_key = args.strategy or "regime"
        base = DEFAULT_BASE_WEIGHTS.get(strategy_key, DEFAULT_BASE_WEIGHTS.get("momentum_value_hybrid", {}))
        for regime, data in regime_weights.items():
            out_name = args.output or f"{strategy_key}_{regime}_ic.json"
            # 如果 output 指定了单一名称，则添加 regime 后缀
            if args.output and len(regime_weights) > 1:
                out_name = out_name.replace(".json", f"_{regime}.json")
            out_path = WEIGHTS_DIR / out_name
            output_weights = {
                "pass1": dict(base.get("pass1", {})),
                "pass2": data["weights"],
            }
            out_path.write_text(json.dumps({
                "strategy": strategy_key,
                "regime": regime,
                "weights": output_weights,
                "ic_mean": data["ic_mean"],
                "min_ic": args.min_ic,
                "based_on_picks": len(df[df["regime"] == regime]),
                "based_on_dates": df[df["regime"] == regime]["date"].nunique(),
                "diagnostic_only": True,
                "warning": "In-sample weights. Do not backtest the same dates. Use only in walk-forward mode.",
            }, ensure_ascii=False, indent=2))
            print(f"\nSaved to {out_path}")
        print("WARNING: outputs marked as diagnostic_only=True.")
        return

    ic_mean = compute_ic(df)
    print("\nFactor mean rank IC:")
    for f, ic in sorted(ic_mean.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {f:30s} {ic:+.4f}")

    new_weights = build_weights(ic_mean, min_ic=args.min_ic)
    print("\nNew weights:")
    for f, w in sorted(new_weights.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {f:30s} {w:.4f}")

    # 组装成 pass2 权重 JSON（保留原 pass1）
    strategy_key = args.strategy or "regime"
    base = DEFAULT_BASE_WEIGHTS.get(strategy_key, DEFAULT_BASE_WEIGHTS.get("momentum_value_hybrid", {}))
    output_weights = {
        "pass1": dict(base.get("pass1", {})),
        "pass2": new_weights,
    }

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.output or f"{strategy_key}_ic.json"
    out_path = WEIGHTS_DIR / out_name
    out_path.write_text(json.dumps({
        "strategy": strategy_key,
        "weights": output_weights,
        "ic_mean": ic_mean,
        "min_ic": args.min_ic,
        "based_on_picks": len(df),
        "based_on_dates": df["date"].nunique(),
        "diagnostic_only": True,
        "warning": "In-sample weights. Do not backtest the same dates. Use only in walk-forward mode.",
    }, ensure_ascii=False, indent=2))
    print(f"\nSaved to {out_path}")
    print("WARNING: output marked as diagnostic_only=True.")


if __name__ == "__main__":
    main()
