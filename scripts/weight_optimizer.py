"""
AlphaHelix 权重优化模块
基于 factor IC 和策略表现，输出下一期可用的动态权重配置。
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

WEIGHTS_DIR = Path("memory/weights")
DEFAULT_BASE_WEIGHTS = {
    "momentum_value_hybrid": {
        "pass1": {
            "mom_20": 0.30, "mom_60": 0.20,
            "ep": 0.15, "bp": 0.15,
            "size": 0.10, "liquidity": 0.10,
        },
        "pass2": {
            "mom_20": 0.25, "mom_60": 0.10,
            "ep": 0.10, "bp": 0.10, "sp": 0.05,
            "roe": 0.10, "profit_growth": 0.05, "revenue_growth": 0.05,
            "net_mf_5d": 0.10, "net_mf_ratio": 0.05,
            "liquidity": 0.05,
        },
    },
    "quality_growth": {
        "pass1": {
            "mom_20": 0.20, "mom_60": 0.10,
            "ep": 0.10, "bp": 0.10,
            "size": 0.05, "liquidity": 0.15,
            "sp": 0.15, "dividend": 0.15,
        },
        "pass2": {
            "roe": 0.25, "profit_growth": 0.20, "revenue_growth": 0.10, "ocf_growth": 0.10,
            "ep": 0.10, "bp": 0.05,
            "net_mf_5d": 0.05, "net_mf_ratio": 0.05,
            "liquidity": 0.05, "mom_20": 0.05,
        },
    },
    "contrarian": {
        "pass1": {
            "bp": 0.18, "ep": 0.13, "sp": 0.08,
            "dividend": 0.08, "liquidity": 0.08,
            "mom_20": -0.10, "mom_60": -0.05,
            "mom_5": 0.05, "reversal_score": 0.15,
            "sector_momentum": 0.05, "relative_to_sector": 0.05,
        },
        "pass2": {
            "bp": 0.18, "ep": 0.13, "roe": 0.08, "profit_growth": 0.08,
            "net_mf_5d": 0.10,
            "mom_20": -0.10, "mom_60": -0.05,
            "mom_5": 0.05, "reversal_score": 0.10,
            "sector_momentum": 0.05, "relative_to_sector": 0.05,
            "liquidity": 0.05,
        },
    },
    "event_driven": {
        "pass1": {
            "mom_20": 0.15, "mom_60": 0.05,
            "ep": 0.10, "bp": 0.10,
            "liquidity": 0.15, "size": 0.05,
            "net_mf_ratio": 0.10,
            "sector_momentum": 0.05,
        },
        "pass2": {
            "forecast_type_score": 0.28,
            "forecast_pchange_mid": 0.18,
            "express_diluted_roe": 0.12,
            "net_mf_5d": 0.10,
            "net_mf_ratio": 0.10,
            "mom_20": 0.05,
            "ep": 0.05,
            "liquidity": 0.05,
            "sector_momentum": 0.05,
        },
    },
}


def load_ic_files(dates: list, strategy: str, horizon: int, ic_strategy: str = None) -> pd.DataFrame:
    """读取多个日期的 factor IC 结果。ic_strategy 用于指定 IC 文件名（可与目标策略不同）。"""
    rows = []
    ic_name = ic_strategy or strategy
    for d in dates:
        path = Path("memory/factor_ic") / f"{d}_{ic_name}_h{horizon}.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "error" in data:
            continue
        row = {"date": d}
        row.update({k: v for k, v in data.items() if k not in ("n", "date")})
        rows.append(row)
    return pd.DataFrame(rows)


def adjust_factor_weights(base_weights: dict, ic_df: pd.DataFrame, learning_rate: float = 0.5) -> dict:
    """
    根据 IC 均值调整因子权重。
    - IC > 0：权重提升
    - IC < 0：权重降低
    - 保持符号不变（负权重保持负向）
    - 归一化使绝对值之和比例尽量稳定
    """
    if ic_df.empty:
        return base_weights

    ic_mean = ic_df.select_dtypes(include=[np.number]).mean().to_dict()
    adjusted = {}

    for factor, w in base_weights.items():
        ic = ic_mean.get(factor, 0.0)
        # 对 IC 做裁剪，避免权重剧烈变化
        ic = np.clip(ic, -0.5, 0.5)
        new_w = w * (1 + learning_rate * ic)
        adjusted[factor] = new_w

    # 归一化：正权重归一化为原正权重和，负权重归一化为原负权重和
    pos_sum_old = sum(w for w in base_weights.values() if w > 0)
    neg_sum_old = sum(abs(w) for w in base_weights.values() if w < 0)

    pos_adj = {k: v for k, v in adjusted.items() if v > 0}
    neg_adj = {k: v for k, v in adjusted.items() if v < 0}

    pos_sum_new = sum(pos_adj.values()) or 1.0
    neg_sum_new = sum(abs(v) for v in neg_adj.values()) or 1.0

    result = {}
    for k, v in pos_adj.items():
        result[k] = round(v / pos_sum_new * pos_sum_old, 4)
    for k, v in neg_adj.items():
        result[k] = round(-(abs(v) / neg_sum_new * neg_sum_old), 4)

    return result


def optimize_weights(dates: list, strategy: str, horizon: int, learning_rate: float = 0.5, ic_strategy: str = None) -> dict:
    """对指定策略优化 pass1/pass2 权重。"""
    base = DEFAULT_BASE_WEIGHTS.get(strategy, {})
    if not base:
        return {}

    ic_df = load_ic_files(dates, strategy, horizon, ic_strategy)
    return {
        "pass1": adjust_factor_weights(base.get("pass1", {}), ic_df, learning_rate),
        "pass2": adjust_factor_weights(base.get("pass2", {}), ic_df, learning_rate),
    }


def main():
    parser = argparse.ArgumentParser(description="AlphaHelix weight optimizer")
    parser.add_argument("--dates", required=True, help="Comma-separated dates YYYYMMDD")
    parser.add_argument("--strategy", default="momentum_value_hybrid", help="Strategy to optimize")
    parser.add_argument("--horizon", type=int, default=10, help="Holding horizon")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate for weight adjustment")
    parser.add_argument("--ic-strategy", default=None, help="Use IC files from another strategy name (e.g. pooled)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    weights = optimize_weights(dates, args.strategy, args.horizon, args.lr, args.ic_strategy)

    result = {
        "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "based_on_dates": dates,
        "strategy": args.strategy,
        "horizon": args.horizon,
        "weights": weights,
    }

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else WEIGHTS_DIR / f"{args.strategy}_latest.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
