"""
Factor Agent v2：LLM + 遗传式因子挖掘 + 正交化筛选

改进点：
1. 支持 --exclude-composites：只用原始列生成因子，避免与已有 composite 因子共线；
2. 支持 --orthogonal：剔除与现有特征平均截面 rank 相关度过高的因子；
3. 支持 --min-icir / --min-positive-ic-ratio：从 IC 稳定性角度筛选；
4. 支持 --append-to-dataset：把最终选中的因子直接拼到数据集，省掉手工步骤；
5. 第二代组合仍保留，但默认只组合通过正交/稳定性筛选的因子。

LLM 默认走 OpenAI API（OPENAI_API_KEY）。未配置时启用 fallback：
根据 prompt 关键词输出一组基于原始列的经验表达式。
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
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_miner import evaluate_factor, _safe_eval
from model_trainer import load_dataset

RAW_COLUMNS = [
    "mom_5", "mom_20", "mom_60", "mom_120", "volatility_20", "amount_ratio_5d",
    "reversal_score", "risk_adj_mom", "dv_ratio", "total_mv", "sector_momentum",
    "relative_to_sector", "sector_breadth", "relative_strength", "roe",
    "revenue_growth", "profit_growth", "ocf_growth", "net_mf_ratio", "net_mf_divergence",
    "forecast_type_score", "forecast_pchange_mid", "express_diluted_roe",
]

COMPOSITE_COLUMNS = [
    "defensive_quality", "smart_money_per_risk", "quality_growth", "value_quality",
    "earnings_surprise_momentum", "growth_consistency", "risk_adj_momentum_20",
]


def get_allowed_columns(exclude_composites: bool = False) -> List[str]:
    if exclude_composites:
        return RAW_COLUMNS[:]
    return RAW_COLUMNS + COMPOSITE_COLUMNS


def _parse_expressions(text: str, allowed: List[str]) -> List[str]:
    """从 LLM 输出中提取形如 expr 的行，并过滤包含未知列名的表达式。"""
    exprs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # 去掉 markdown 列表符号
        line = line.lstrip("-`*.。 \t")
        if not line:
            continue
        # 只允许使用 allowed 列名
        try:
            # 简单解析：把非字母数字下划线替换为空格，再 split
            tokens = re.sub(r"[^a-zA-Z0-9_]", " ", line).split()
            used_cols = [t for t in tokens if t in allowed]
            if not used_cols:
                continue
            exprs.append(line)
        except Exception:
            continue
    return list(dict.fromkeys(exprs))[:30]


def _fallback_expressions(prompt: str, allowed: List[str]) -> List[str]:
    """无 OpenAI API 时的经验表达式生成器。仅使用 allowed 列。"""
    prompt_l = prompt.lower()
    candidates = []
    has_momentum = any(k in prompt_l for k in ["动量", "momentum", "趋势", "trend"])
    has_quality = any(k in prompt_l for k in ["质量", "quality", "roe", "盈利"])
    has_flow = any(k in prompt_l for k in ["资金", "smart money", "flow", "money", "北向", "龙虎榜", "top list"])
    has_value = any(k in prompt_l for k in ["价值", "value", "估值", "ep", "bp", "股息"])
    has_event = any(k in prompt_l for k in ["事件", "event", "业绩", "披露", "公告"])
    has_vol = any(k in prompt_l for k in ["波动", "volatility", "风险", "risk"])

    # 如果用户没有指定任何关键词，默认全部主题都给一些
    if not any([has_momentum, has_quality, has_flow, has_value, has_event, has_vol]):
        has_momentum = has_quality = has_flow = has_value = has_event = has_vol = True

    if has_momentum:
        candidates += [
            "mom_20 / (volatility_20 + 1e-9)",
            "mom_5 * (mom_60 - mom_20)",
            "(mom_20 - mom_60) / (volatility_20 + 1e-9)",
            "mom_120 / (mom_60 + 1e-9)",
        ]
    if has_quality:
        candidates += [
            "roe * profit_growth / (volatility_20 + 1e-9)",
            "roe / (1 + abs(ocf_growth - profit_growth))",
            "(revenue_growth + profit_growth + ocf_growth) / 3",
            "roe * (1 + revenue_growth)",
        ]
    if has_flow:
        candidates += [
            "net_mf_ratio / (volatility_20 + 1e-9)",
            "net_mf_divergence * mom_20",
            "net_mf_ratio * relative_strength",
        ]
    if has_value:
        candidates += [
            "dv_ratio * roe",
            "dv_ratio / (volatility_20 + 1e-9)",
        ]
    if has_event:
        candidates += [
            "forecast_pchange_mid / (volatility_20 + 1e-9)",
            "forecast_type_score * mom_20",
            "forecast_pchange_mid * net_mf_ratio",
        ]
    if has_vol:
        candidates += [
            "risk_adj_mom * roe",
            "mom_20 / (amount_ratio_5d + 1e-9)",
            "reversal_score * net_mf_ratio",
        ]

    # 通用：只允许使用 allowed 的原始列
    candidates += [
        "roe * net_mf_ratio / (volatility_20 + 1e-9)",
        "(mom_20 + forecast_pchange_mid) / (volatility_20 + 1e-9)",
        "relative_strength * dv_ratio",
    ]

    # 过滤掉包含非 allowed 列的表达式
    def _is_number(tok: str) -> bool:
        try:
            float(tok.replace("e", "").replace("E", ""))
            return True
        except Exception:
            return False

    filtered = []
    for expr in candidates:
        tokens = re.sub(r"[^a-zA-Z0-9_]", " ", expr).split()
        used = [t for t in tokens if t in allowed]
        unknown = [t for t in tokens if t not in allowed and not _is_number(t) and t not in ("e", "E", "abs")]
        if used and not unknown:
            filtered.append(expr)
    return list(dict.fromkeys(filtered))


def generate_with_llm(prompt: str, allowed: List[str], n: int = 10, url: str = None) -> List[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[factor_agent] OPENAI_API_KEY 未设置，使用 fallback 表达式生成器。")
        return _fallback_expressions(prompt, allowed)

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
            "你是资深量化因子工程师。请根据用户提示生成最多 15 个 A股截面因子表达式。"
            "表达式使用 pandas 语法，仅允许使用以下列名：\n" + ", ".join(allowed) +
            "\n要求：1）无未来函数；2）不得使用 if/else/for/while；3）分母必须加 1e-9；"
            "4）只返回表达式，每行一个，不要解释；5）尽量生成不同主题、彼此正交的因子。"
        )
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            temperature=0.8,
            max_tokens=800,
        )
        text = resp.choices[0].message.content
        return _parse_expressions(text, allowed)[:n]
    except Exception as e:
        print(f"[factor_agent] LLM 调用失败: {e}，使用 fallback。")
        return _fallback_expressions(prompt, allowed)


def _safe_name(expr: str) -> str:
    """把表达式转为合法文件名/列名。"""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", expr)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:80] or "expr"


def _eval_factor_series(df: pd.DataFrame, expr: str) -> pd.Series:
    """安全评估表达式并返回 Series，非有限值置为 NaN。"""
    try:
        s = _safe_eval(expr, df)
        s = s.replace([np.inf, -np.inf], np.nan)
        return s
    except Exception:
        return pd.Series(np.nan, index=df.index)


def max_avg_rank_correlation(df: pd.DataFrame, expr: str,
                             existing_cols: List[str],
                             min_dates: int = 10) -> float:
    """
    计算候选因子与现有特征的最大平均截面 rank 相关系数。
    返回 0~1，越接近 1 说明越不独立。
    """
    s = _eval_factor_series(df, expr)
    if s.isna().all():
        return 1.0
    df = df.copy()
    df["_candidate"] = s

    max_corr = 0.0
    for col in existing_cols:
        if col not in df.columns:
            continue
        corrs = []
        for _, g in df.groupby("date"):
            if len(g) < 5:
                continue
            x = g["_candidate"].rank(pct=True, na_option="keep")
            y = g[col].rank(pct=True, na_option="keep")
            valid = x.notna() & y.notna()
            if valid.sum() < 5:
                continue
            c, _ = stats.spearmanr(x[valid], y[valid])
            if np.isfinite(c):
                corrs.append(abs(c))
        if len(corrs) >= min_dates:
            avg_corr = np.mean(corrs)
            if avg_corr > max_corr:
                max_corr = avg_corr
    return max_corr


def mutate_factors(top_exprs: List[str], max_pairs: int = 20) -> List[str]:
    """对 top 因子做两两组合生成第二代。"""
    if len(top_exprs) < 2:
        return []
    second_gen = []
    for a, b in combinations(top_exprs[:max_pairs], 2):
        second_gen.append(f"({a}) * ({b})")
        second_gen.append(f"({a}) / (({b}) + 1e-9)")
        second_gen.append(f"({a}) + ({b})")
    seen = set()
    out = []
    for e in second_gen:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def append_winners_to_dataset(df: pd.DataFrame, winners: List[Dict],
                              output_path: str,
                              winsorize_quantiles: tuple = (0.01, 0.99)) -> pd.DataFrame:
    """把 winner 表达式计算成列并拼到数据集，输出 parquet。"""
    df = df.copy()
    for r in winners:
        name = r["name"]
        expr = r["expr"]
        try:
            df[name] = _eval_factor_series(df, expr)
            lo, hi = df[name].quantile(winsorize_quantiles[0]), df[name].quantile(winsorize_quantiles[1])
            df[name] = df[name].clip(lo, hi)
        except Exception as e:
            print(f"[factor_agent] 跳过因子 {name}: {e}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[factor_agent] Appended {len(winners)} factors -> {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=None, help="LLM 生成因子的提示")
    parser.add_argument("--url", default=None, help="论文/研报 URL（可选）")
    parser.add_argument("--seed-file", default=None, help="本地 YAML 种子文件（可选）")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--ic-threshold", type=float, default=0.03, help="保留 |mean_ic| 的阈值")
    parser.add_argument("--min-icir", type=float, default=0.5, help="最小 ICIR（年化根号天数后）")
    parser.add_argument("--min-positive-ic-ratio", type=float, default=0.55, help="最小正 IC 比例")
    parser.add_argument("--max-avg-corr", type=float, default=0.85, help="与现有特征最大平均 rank 相关系数上限")
    parser.add_argument("--max-pairs", type=int, default=20, help="第二代组合数量上限")
    parser.add_argument("--exclude-composites", action="store_true", help="只用原始列生成因子")
    parser.add_argument("--orthogonal", action="store_true", help="做正交性筛选")
    parser.add_argument("--second-gen", action="store_true", help="是否对 top 因子做第二代组合")
    parser.add_argument("--output", default="memory/factors/winner_factors.yaml")
    parser.add_argument("--append-to-dataset", default=None, help="把赢家因子拼到数据集并输出路径")
    args = parser.parse_args()

    df = load_dataset(args.horizon, args.dataset)
    allowed = get_allowed_columns(args.exclude_composites)
    existing_cols = [c for c in df.columns if c in (RAW_COLUMNS + COMPOSITE_COLUMNS)]
    print(f"[factor_agent] Dataset: {len(df)} rows, {df['date'].nunique()} dates")
    print(f"[factor_agent] Allowed columns ({len(allowed)}): {', '.join(allowed[:10])}...")

    candidates: List[Dict[str, str]] = []

    if args.seed_file:
        path = Path(args.seed_file)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            for item in data.get("expressions", []):
                candidates.append({"name": item.get("name", item["expr"]), "expr": item["expr"]})

    if args.prompt or args.url:
        exprs = generate_with_llm(args.prompt or "生成 A股截面选股因子", allowed, n=15, url=args.url)
        for i, expr in enumerate(exprs):
            candidates.append({"name": f"llm_{i}_{_safe_name(expr)}", "expr": expr})

    if not candidates:
        print("[factor_agent] 没有候选因子，退出。")
        return

    print(f"[factor_agent] Evaluating {len(candidates)} candidates...")
    results = []
    for c in candidates:
        res = evaluate_factor(df, c["expr"], name=c["name"])
        if "error" not in res:
            print(f"  {res['name']:40s} IC={res['mean_ic']:+.4f} ICIR={res['icir']:+.2f} pos={res['positive_ic_ratio']:.1%} spread={res['top_bottom_spread']:+.4f}")
            if args.orthogonal:
                corr = max_avg_rank_correlation(df, c["expr"], existing_cols)
                res["max_avg_corr"] = corr
                print(f"      max_avg_rank_corr={corr:.3f}")
        else:
            print(f"  {res['name']:40s} ERROR: {res['error']}")
        results.append(res)

    # 稳定性 + IC 阈值筛选
    stable = []
    for r in results:
        if "error" in r:
            continue
        if abs(r["mean_ic"]) < args.ic_threshold:
            continue
        if r["icir"] < args.min_icir:
            continue
        if r["positive_ic_ratio"] < args.min_positive_ic_ratio:
            continue
        if args.orthogonal and r.get("max_avg_corr", 0) > args.max_avg_corr:
            continue
        stable.append(r)

    stable = sorted(stable, key=lambda x: -abs(x["mean_ic"]))
    print(f"\n[factor_agent] 稳定且独立的第一代因子: {len(stable)} / {len(candidates)}")

    winners = stable[:]

    # 第二代组合：只在第一代稳定因子里组合
    if args.second_gen and stable:
        top_exprs = [r["expr"] for r in stable[:10]]
        second_exprs = mutate_factors(top_exprs, max_pairs=args.max_pairs)
        print(f"[factor_agent] Evaluating {len(second_exprs)} second-generation combinations...")
        second_results = []
        for i, expr in enumerate(second_exprs):
            res = evaluate_factor(df, expr, name=f"gen2_{i}_{_safe_name(expr)}")
            if "error" not in res:
                if abs(res["mean_ic"]) >= args.ic_threshold and res["icir"] >= args.min_icir and res["positive_ic_ratio"] >= args.min_positive_ic_ratio:
                    if args.orthogonal:
                        corr = max_avg_rank_correlation(df, expr, existing_cols)
                        res["max_avg_corr"] = corr
                        if corr > args.max_avg_corr:
                            continue
                    second_results.append(res)
                    print(f"  {res['name']:40s} IC={res['mean_ic']:+.4f} ICIR={res['icir']:+.2f} pos={res['positive_ic_ratio']:.1%} spread={res['top_bottom_spread']:+.4f}")
        winners.extend(second_results)
        winners = sorted(winners, key=lambda x: -abs(x["mean_ic"]))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump({"winners": winners}, f, allow_unicode=True, sort_keys=False)
    print(f"\n[factor_agent] Saved {len(winners)} winners to {out_path}")

    if winners:
        print("\n=== Top 10 Winners ===")
        for r in winners[:10]:
            extra = f" corr={r.get('max_avg_corr', -1):.3f}" if "max_avg_corr" in r else ""
            print(f"{r['name']:40s} IC={r['mean_ic']:+.4f} ICIR={r['icir']:+.2f} pos={r['positive_ic_ratio']:.1%}{extra}")
            print(f"    expr: {r['expr']}")

    if args.append_to_dataset and winners:
        append_winners_to_dataset(df, winners, args.append_to_dataset)


if __name__ == "__main__":
    main()
