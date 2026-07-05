"""
Factor Agent：LLM + 遗传式因子挖掘

支持三种输入：
1. --prompt：直接给 LLM 文本提示生成因子表达式；
2. --url：抓取一篇论文/研报，让 LLM 提炼因子；
3. --seed-file：本地 YAML 种子表达式，做交叉/变异生成第二代。

流程：
- 生成候选表达式；
- 用 factor_miner.evaluate_factor 在数据集上计算 IC；
- 保留 |mean_ic| > threshold 的因子，写回 memory/factors/winner_factors.yaml；
- 对 top 因子做两两组合（乘、除、加）生成第二代，再评估。

LLM 默认走 OpenAI API（OPENAI_API_KEY）。未配置时启用 fallback：
根据 prompt 关键词输出一组经验表达式，方便在无 API 环境也能跑通流程。
"""
import sys
import os
import argparse
import re
import yaml
from pathlib import Path
from itertools import combinations
from typing import List, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_miner import evaluate_factor, _safe_eval
from model_trainer import load_dataset

try:
    from webfetch import fetch_url  # 如果未来有本地 webfetch 工具
except Exception:
    fetch_url = None


BASE_COLUMNS = [
    "mom_5", "mom_20", "mom_60", "mom_120", "volatility_20", "amount_ratio_5d",
    "reversal_score", "risk_adj_mom", "dv_ratio", "total_mv", "sector_momentum",
    "relative_to_sector", "sector_breadth", "relative_strength", "roe",
    "revenue_growth", "profit_growth", "ocf_growth", "net_mf_ratio", "net_mf_divergence",
    "forecast_type_score", "forecast_pchange_mid", "express_diluted_roe",
    "defensive_quality", "smart_money_per_risk", "quality_growth", "value_quality",
    "earnings_surprise_momentum", "growth_consistency", "risk_adj_momentum_20",
]


def _parse_expressions(text: str) -> List[str]:
    """从 LLM 输出中提取形如 expr 的行。"""
    exprs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-", "*", "`", "【", "```")):
            # 去掉 markdown 标记
            line = line.strip("-`* 【】")
        if not line:
            continue
        # 只保留包含已知列名或运算符的行
        if any(c in line for c in ["/", "*", "+", "-"]) or any(col in line for col in BASE_COLUMNS):
            exprs.append(line)
    return list(dict.fromkeys(exprs))[:20]


def _fallback_expressions(prompt: str) -> List[str]:
    """无 OpenAI API 时的经验表达式生成器。"""
    prompt_l = prompt.lower()
    candidates = []
    if any(k in prompt_l for k in ["动量", "momentum", "趋势", "trend"]):
        candidates += [
            "mom_20 / (volatility_20 + 1e-9)",
            "mom_5 * (mom_60 - mom_20)",
            "(mom_20 - mom_60) / volatility_20",
        ]
    if any(k in prompt_l for k in ["质量", "quality", "roe", "盈利"]):
        candidates += [
            "roe * profit_growth / (volatility_20 + 1e-9)",
            "roe / (1 + abs(ocf_growth - profit_growth))",
            "(revenue_growth + profit_growth + ocf_growth) / 3",
        ]
    if any(k in prompt_l for k in ["资金", "smart money", "flow", "money", "北向", "龙虎榜", "top list"]):
        candidates += [
            "net_mf_ratio / (volatility_20 + 1e-9)",
            "net_mf_divergence * mom_20",
            "smart_money_per_risk * quality_growth",
        ]
    if any(k in prompt_l for k in ["价值", "value", "估值", "ep", "bp", "股息"]):
        candidates += [
            "dv_ratio * roe",
            "ep / (volatility_20 + 1e-9)",
        ]
    if any(k in prompt_l for k in ["事件", "event", "业绩", "披露", "公告"]):
        candidates += [
            "forecast_pchange_mid / (volatility_20 + 1e-9)",
            "earnings_surprise_momentum * net_mf_ratio",
            "forecast_type_score * mom_20",
        ]
    if any(k in prompt_l for k in ["波动", "volatility", "风险", "risk"]):
        candidates += [
            "risk_adj_momentum_20 * quality_growth",
            "mom_20 / (amount_ratio_5d + 1e-9)",
        ]
    # 通用组合：高 IC 复合因子的再组合
    candidates += [
        "defensive_quality * smart_money_per_risk",
        "quality_growth / (volatility_20 + 1e-9)",
        "(earnings_surprise_momentum + smart_money_per_risk) / 2",
        "value_quality * growth_consistency",
    ]
    return list(dict.fromkeys(candidates))


def generate_with_llm(prompt: str, n: int = 10, url: str = None) -> List[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[factor_agent] OPENAI_API_KEY 未设置，使用 fallback 表达式生成器。")
        return _fallback_expressions(prompt)

    content = prompt
    if url:
        try:
            import requests
            resp = requests.get(url, timeout=20)
            text = resp.text[:8000]
            content = f"论文/研报内容摘要：\n{text[:4000]}\n\n请基于以上内容生成量化因子表达式：\n{prompt}"
        except Exception as e:
            print(f"[factor_agent] 抓取 URL 失败: {e}")

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        system = (
            "你是资深量化因子工程师。请根据提示生成最多 10 个 A股截面因子表达式。"
            "表达式使用 pandas 语法，仅使用以下列名：\n" + ", ".join(BASE_COLUMNS) +
            "\n要求：1）无未来函数；2）不得使用 if/else；3）分母加 1e-9；4）只返回表达式，每行一个。"
        )
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            temperature=0.7,
            max_tokens=600,
        )
        text = resp.choices[0].message.content
        return _parse_expressions(text)[:n]
    except Exception as e:
        print(f"[factor_agent] LLM 调用失败: {e}，使用 fallback。")
        return _fallback_expressions(prompt)


def _safe_name(expr: str) -> str:
    """把表达式转为合法文件名/列名。"""
    name = re.sub(r"[\/\*\+\-\(\)\s]+", "_", expr)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:80] or "expr"


def mutate_factors(df: pd.DataFrame, top_exprs: List[str], max_pairs: int = 20) -> List[str]:
    """对 top 因子做两两组合生成第二代。"""
    if len(top_exprs) < 2:
        return []
    second_gen = []
    for a, b in combinations(top_exprs[:max_pairs], 2):
        second_gen.append(f"({a}) * ({b})")
        second_gen.append(f"({a}) / (({b}) + 1e-9)")
        second_gen.append(f"({a}) + ({b})")
    # 去重
    seen = set()
    out = []
    for e in second_gen:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=None, help="LLM 生成因子的提示")
    parser.add_argument("--url", default=None, help="论文/研报 URL（可选）")
    parser.add_argument("--seed-file", default=None, help="本地 YAML 种子文件（可选）")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--ic-threshold", type=float, default=0.02, help="保留 |mean_ic| 的阈值")
    parser.add_argument("--max-pairs", type=int, default=20, help="第二代组合数量上限")
    parser.add_argument("--output", default="memory/factors/winner_factors.yaml")
    parser.add_argument("--second-gen", action="store_true", help="是否对 top 因子做第二代组合")
    args = parser.parse_args()

    df = load_dataset(args.horizon, args.dataset)
    print(f"[factor_agent] Dataset: {len(df)} rows, {df['date'].nunique()} dates")

    candidates: List[Dict[str, str]] = []

    if args.seed_file:
        path = Path(args.seed_file)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            for item in data.get("expressions", []):
                candidates.append({"name": item.get("name", item["expr"]), "expr": item["expr"]})

    if args.prompt or args.url:
        exprs = generate_with_llm(args.prompt or "生成 A股截面选股因子", n=10, url=args.url)
        for i, expr in enumerate(exprs):
            candidates.append({"name": f"llm_{i}_{_safe_name(expr)}", "expr": expr})

    if not candidates:
        print("[factor_agent] 没有候选因子，退出。")
        return

    print(f"[factor_agent] Evaluating {len(candidates)} candidates...")
    results = []
    for c in candidates:
        res = evaluate_factor(df, c["expr"], name=c["name"])
        results.append(res)
        if "error" not in res:
            print(f"  {res['name']:30s} IC={res['mean_ic']:+.4f} ICIR={res['icir']:+.2f} spread={res['top_bottom_spread']:+.4f}")
        else:
            print(f"  {res['name']:30s} ERROR: {res['error']}")

    # 第一代赢家
    winners = [r for r in results if "error" not in r and abs(r["mean_ic"]) >= args.ic_threshold]
    winners = sorted(winners, key=lambda x: -abs(x["mean_ic"]))
    print(f"\n[factor_agent] 第一代赢家: {len(winners)} / {len(candidates)}")

    # 第二代组合
    if args.second_gen and winners:
        top_exprs = [r["expr"] for r in winners[:10]]
        second_exprs = mutate_factors(df, top_exprs, max_pairs=args.max_pairs)
        print(f"[factor_agent] Evaluating {len(second_exprs)} second-generation combinations...")
        second_results = []
        for i, expr in enumerate(second_exprs):
            res = evaluate_factor(df, expr, name=f"gen2_{i}_{_safe_name(expr)}")
            second_results.append(res)
            if "error" not in res:
                print(f"  {res['name']:30s} IC={res['mean_ic']:+.4f} ICIR={res['icir']:+.2f} spread={res['top_bottom_spread']:+.4f}")
        second_winners = [r for r in second_results if "error" not in r and abs(r["mean_ic"]) >= args.ic_threshold]
        winners.extend(second_winners)
        winners = sorted(winners, key=lambda x: -abs(x["mean_ic"]))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump({"winners": winners}, f, allow_unicode=True, sort_keys=False)
    print(f"\n[factor_agent] Saved {len(winners)} winners to {out_path}")

    if winners:
        print("\n=== Top 5 Winners ===")
        for r in winners[:5]:
            print(f"{r['name']:30s} IC={r['mean_ic']:+.4f} ICIR={r['icir']:+.2f}  {r['expr']}")


if __name__ == "__main__":
    main()
