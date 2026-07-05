"""
AlphaHelix 因子挖掘器（RD-Agent 风格最小化实现）

目标：自动评估候选因子表达式，筛选出对 excess_return 有显著 IC 的新因子。

两种输入方式：
1. --expressions-file：YAML 文件，内含多个因子表达式；
2. --expression：单个表达式，用于快速验证。

表达式语法：使用 dataset 中已有列名的 pandas 算术表达式，例如：
  - mom_20 / volatility_20
  - roe * profit_growth
  - net_mf_ratio / amount_ratio_5d

评估方式：
- 每个交易日截面上计算因子值与未来 excess_return 的 Spearman IC；
- 输出平均 IC、ICIR、正 IC 比例、top-bottom 20% 收益差。

LLM 扩展：
- 若设置了 OPENAI_API_KEY，可通过 --llm-prompt 让 LLM 根据文本描述生成表达式；
- 默认未设置时，LLM 生成步骤跳过，仅做本地评估。
"""
import sys
import os
import argparse
import yaml
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_trainer import load_dataset


def _safe_eval(expr: str, df: pd.DataFrame) -> pd.Series:
    """在 df 上安全地评估表达式，返回 Series。"""
    try:
        # 只允许使用 df 列名和基本运算符
        result = pd.eval(expr, local_dict=df, engine="python")
        if isinstance(result, pd.Series):
            return result
        return pd.Series(result, index=df.index)
    except Exception as e:
        raise ValueError(f"表达式 '{expr}' 评估失败: {e}")


def evaluate_factor(df: pd.DataFrame, expr: str, name: str = None) -> Dict:
    """评估单个因子的 IC 和分位数收益。"""
    name = name or expr
    df = df.copy()
    df["_factor"] = _safe_eval(expr, df)

    # 过滤非有限值
    valid = df[np.isfinite(df["_factor"]) & np.isfinite(df["excess_return"])]
    if valid.empty:
        return {"name": name, "expr": expr, "error": "no valid values"}

    ics = []
    spreads = []
    for d, g in valid.groupby("date"):
        if len(g) < 5:
            continue
        ic, _ = stats.spearmanr(g["_factor"], g["excess_return"])
        ics.append(ic)

        g_sorted = g.sort_values("_factor", ascending=False)
        n = max(1, len(g_sorted) // 5)
        top_mean = g_sorted.iloc[:n]["excess_return"].mean()
        bot_mean = g_sorted.iloc[-n:]["excess_return"].mean()
        spreads.append(top_mean - bot_mean)

    ics = np.array(ics)
    spreads = np.array(spreads)
    return {
        "name": name,
        "expr": expr,
        "mean_ic": float(np.nanmean(ics)),
        "ic_std": float(np.nanstd(ics)),
        "icir": float(np.nanmean(ics) / np.nanstd(ics) * np.sqrt(len(ics))) if np.nanstd(ics) > 0 else 0.0,
        "positive_ic_ratio": float(np.nanmean(ics > 0)),
        "top_bottom_spread": float(np.nanmean(spreads)),
        "n_dates": int(len(ics)),
    }


def generate_expressions_with_llm(prompt: str, n: int = 5) -> List[str]:
    """使用 LLM 根据 prompt 生成因子表达式。需要 OPENAI_API_KEY。"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[factor_miner] OPENAI_API_KEY 未设置，跳过 LLM 生成。")
        return []
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        system_msg = (
            "你是一个量化因子工程师。根据用户描述，生成 1~5 个基于 pandas 的因子表达式。"
            "可用列名包括：mom_5, mom_20, mom_60, mom_120, volatility_20, amount_ratio_5d, "
            "reversal_score, risk_adj_mom, dv_ratio, total_mv, sector_momentum, relative_to_sector, "
            "sector_breadth, relative_strength, roe, revenue_growth, profit_growth, ocf_growth, "
            "net_mf_ratio, net_mf_divergence, forecast_type_score, forecast_pchange_mid, express_diluted_roe。"
            "只返回表达式列表，每行一个，不要解释。"
        )
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        text = resp.choices[0].message.content
        expressions = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
        return expressions[:n]
    except Exception as e:
        print(f"[factor_miner] LLM 生成失败: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="parquet 数据集路径")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--expressions-file", default="scripts/factor_candidates.yaml",
                        help="候选因子表达式 YAML")
    parser.add_argument("--expression", default=None, help="单个表达式，覆盖文件")
    parser.add_argument("--llm-prompt", default=None, help="LLM 生成因子的提示")
    parser.add_argument("--output", default="memory/factors/factor_eval_results.yaml",
                        help="评估结果输出路径")
    args = parser.parse_args()

    df = load_dataset(args.horizon, args.dataset)
    print(f"[factor_miner] Loaded dataset: {len(df)} rows, {df['date'].nunique()} dates")

    candidates = []
    if args.expression:
        candidates.append({"name": "cli_expr", "expr": args.expression})
    elif args.llm_prompt:
        exprs = generate_expressions_with_llm(args.llm_prompt)
        for i, expr in enumerate(exprs):
            candidates.append({"name": f"llm_{i}", "expr": expr})
    else:
        path = Path(args.expressions_file)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            candidates = data.get("expressions", [])
        else:
            print(f"[factor_miner] Expressions file not found: {path}")
            return

    if not candidates:
        print("[factor_miner] No candidate expressions to evaluate.")
        return

    results = []
    for item in candidates:
        name = item.get("name", item["expr"])
        expr = item["expr"]
        print(f"[factor_miner] Evaluating {name}: {expr}")
        res = evaluate_factor(df, expr, name=name)
        results.append(res)
        if "error" not in res:
            print(f"  meanIC={res['mean_ic']:+.4f}, ICIR={res['icir']:+.2f}, "
                  f"posIC={res['positive_ic_ratio']:.1%}, spread={res['top_bottom_spread']:+.4f}")
        else:
            print(f"  ERROR: {res['error']}")

    # 按 |mean_ic| 排序
    results_sorted = sorted(results, key=lambda x: -abs(x.get("mean_ic", 0)))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump({"results": results_sorted}, f, allow_unicode=True, sort_keys=False)
    print(f"\n[factor_miner] Results saved to {out_path}")

    print("\n=== Top Factors ===")
    for r in results_sorted[:5]:
        if "error" not in r:
            print(f"{r['name']:20s} IC={r['mean_ic']:+.4f} ICIR={r['icir']:+.2f} "
                  f"spread={r['top_bottom_spread']:+.4f}  {r['expr']}")


if __name__ == "__main__":
    main()
