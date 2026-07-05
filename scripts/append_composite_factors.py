"""
把 factor_miner 筛出的高 IC 复合因子追加到已有 dataset。

这些因子均由现有特征算术组合而成，无需重新调用 Tushare，因此可以低成本增量更新。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COMPOSITE_FACTORS = {
    "defensive_quality": "roe / (volatility_20 + 1e-9)",
    "smart_money_per_risk": "net_mf_ratio / (volatility_20 + 1e-9)",
    "quality_growth": "roe * profit_growth",
    "value_quality": "dv_ratio * roe",
    "earnings_surprise_momentum": "(forecast_pchange_mid - mom_20) / (volatility_20 + 1e-9)",
    "growth_consistency": "revenue_growth * profit_growth * ocf_growth",
    "risk_adj_momentum_20": "mom_20 / (volatility_20 + 1e-9)",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="memory/dataset/features_h10.parquet")
    parser.add_argument("--output", default="memory/dataset/features_h10_composite.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    print(f"[append_composite] Loaded {df.shape} from {args.dataset}")

    for name, expr in COMPOSITE_FACTORS.items():
        df[name] = pd.eval(expr, local_dict=df, engine="python")
        # 极值截断，避免异常值
        df[name] = df[name].replace([np.inf, -np.inf], np.nan)
        lower = df[name].quantile(0.01)
        upper = df[name].quantile(0.99)
        df[name] = df[name].clip(lower, upper)
        print(f"[append_composite] Added {name}: mean={df[name].mean():.4f}, std={df[name].std():.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[append_composite] Saved {df.shape} to {args.output}")


if __name__ == "__main__":
    main()
