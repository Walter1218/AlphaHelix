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
EVAL_DIR = Path("eval")
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


def compute_pass2_weights_for_dates(dates: list, min_ic: float = 0.0) -> dict:
    """根据给定日期列表的历史选股与收益，生成 pass2 权重。"""
    records = []
    for date in dates:
        eval_path = Path("memory/eval") / f"{date}_regime_h10.json"
        if not eval_path.exists():
            # 兼容非 regime 文件名
            matches = list(Path("memory/eval").glob(f"{date}_*_h10.json"))
            if not matches:
                continue
            eval_path = matches[0]
        try:
            ev = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "error" in ev or ev.get("cash"):
            continue
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
            rec = {"date": date, "ts_code": ts, "rank": p.get("rank"), "return": ret_map[ts]}
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
        return {}
    ic_mean = compute_ic(df)
    return build_weights(ic_mean, min_ic=min_ic)


def main():
    parser = argparse.ArgumentParser(description="Calibrate strategy weights from historical rank IC")
    parser.add_argument("--strategy", default=None, help="Filter by strategy; if None, pool all strategies")
    parser.add_argument("--min-ic", type=float, default=0.0, help="Ignore factors with IC below this threshold")
    parser.add_argument("--output", default=None, help="Output weights file name; default {strategy}_ic.json")
    args = parser.parse_args()

    df = load_results(strategy=args.strategy)
    if df.empty:
        print("No data found")
        return

    print(f"Loaded {len(df)} picks across {df['date'].nunique()} dates")
    if args.strategy:
        print(f"Strategy filter: {args.strategy}")

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
    }, ensure_ascii=False, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
