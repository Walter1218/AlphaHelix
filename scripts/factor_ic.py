"""
AlphaHelix 因子 IC 计算模块
基于选股当日的因子值与持有期实际收益，计算各因子的 rank IC（信息系数）。
"""
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = Path("memory/factor_ic")

FACTOR_FIELDS = [
    "mom_5", "mom_20", "mom_60", "pe", "pb", "ps", "dv_ratio",
    "roe", "revenue_growth", "profit_growth", "ocf_growth",
    "net_mf_5d", "net_mf_20d", "net_mf_ratio",
    "avg_amount_20", "amount_ratio_5d", "volatility_20", "total_mv",
    "reversal_score", "sector_momentum", "relative_to_sector", "sector_mom5", "sector_amount_ratio",
    "forecast_type_score", "forecast_pchange_mid", "express_diluted_roe", "express_diluted_eps",
]


def load_snapshot(date: str) -> dict:
    path = Path("memory/stock") / f"{date}.json"
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_eval(date: str, strategy: str, horizon: int) -> dict:
    if strategy == "regime":
        path = Path("memory/eval") / f"{date}_regime_h{horizon}.json"
    else:
        path = Path("memory/eval") / f"{date}_{strategy}_h{horizon}.json"
    if not path.exists():
        raise FileNotFoundError(f"Eval not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_ic(snapshot: dict, eval_result: dict) -> dict:
    """计算每个因子的 rank IC。"""
    picks = snapshot.get("picks", [])
    details = {d["ts_code"]: d for d in eval_result.get("details", []) if "error" not in d}

    rows = []
    for pick in picks:
        ts_code = pick["ts_code"]
        if ts_code not in details:
            continue
        ret = details[ts_code].get("abs_return")
        if ret is None:
            continue
        row = {"ts_code": ts_code, "return": ret}
        for f in FACTOR_FIELDS:
            row[f] = pick.get(f)
        rows.append(row)

    if len(rows) < 4:
        return {"error": "Too few valid picks to compute IC", "n": len(rows)}

    df = pd.DataFrame(rows)
    ic_results = {"n": len(df), "date": snapshot.get("date")}

    for f in FACTOR_FIELDS:
        if f not in df.columns or df[f].isna().all():
            continue
        # 估值/波动率类因子方向反转：低 PE/PB/波动率应带来高收益
        ascending = f in ("pe", "pb", "ps", "volatility_20")
        rank_factor = df[f].rank(ascending=ascending, pct=True)
        rank_return = df["return"].rank(pct=True)
        valid = rank_factor.notna() & rank_return.notna()
        if valid.sum() < 4:
            continue
        x = rank_factor[valid].values
        y = rank_return[valid].values
        if np.std(x) == 0 or np.std(y) == 0:
            ic = 0.0
        else:
            ic = float(np.corrcoef(x, y)[0, 1])
        if pd.isna(ic):
            ic = 0.0
        ic_results[f] = round(ic, 4)

    return ic_results


def main():
    parser = argparse.ArgumentParser(description="AlphaHelix factor IC calculator")
    parser.add_argument("date", help="Trade date YYYYMMDD")
    parser.add_argument("--strategy", default="momentum_value_hybrid", help="Strategy name")
    parser.add_argument("--horizon", type=int, default=10, help="Holding horizon")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    snapshot = load_snapshot(args.date)
    eval_result = load_eval(args.date, args.strategy, args.horizon)
    ic = compute_ic(snapshot, eval_result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else OUTPUT_DIR / f"{args.date}_{args.strategy}_h{args.horizon}.json"
    out_path.write_text(json.dumps(ic, ensure_ascii=False, indent=2))
    print(json.dumps(ic, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
