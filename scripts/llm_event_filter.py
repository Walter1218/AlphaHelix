"""
LLM 事件/舆情过滤器（ scaffold ）

功能：对选股结果中的 top-k 候选，基于最近一段时间的新闻/公告/龙虎榜席位，
用 LLM 判断是否存在重大利空或高风险事件，输出一个风险分数（-1~1），
供下游组合构造时降权或剔除。

当前限制：
- Tushare 免费/基础版通常没有个股新闻接口；如调用失败则返回中性分。
- 真正发挥作用需要接入可稳定获取的个股文本数据源（如 Tushare 付费资讯、
  财联社、东方财富公告、公司公告 PDF 等）。
- 本脚本先给出可运行的脚手架和接口约定。

用法：
  python scripts/llm_event_filter.py --date 20250402 --ts-codes 000001.SZ,600519.SH --lookback-days 10
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import get_trade_date_before


try:
    from _tushare_utils import tushare_call
except Exception:
    tushare_call = None


def _get_news_text(ts_code: str, start_date: str, end_date: str) -> List[str]:
    """尝试拉取个股新闻标题/摘要。Tushare 接口可能不可用。"""
    snippets = []
    if tushare_call is None:
        return snippets

    # 1. 尝试 major_news（若账号有权限）
    try:
        df = tushare_call("major_news", {"ts_code": ts_code, "start_date": start_date, "end_date": end_date})
        if not df.empty and "title" in df.columns:
            for _, row in df.iterrows():
                text = str(row.get("title", ""))
                if "content" in row:
                    text += " " + str(row.get("content", ""))
                snippets.append(text)
    except Exception:
        pass

    # 2. 尝试 company_news（部分付费接口）
    try:
        df = tushare_call("news", {"ts_code": ts_code, "start_date": start_date, "end_date": end_date})
        if not df.empty and "title" in df.columns:
            for _, row in df.iterrows():
                text = str(row.get("title", ""))
                if "content" in row:
                    text += " " + str(row.get("content", ""))
                snippets.append(text)
    except Exception:
        pass

    return snippets


def _score_with_llm(ts_code: str, texts: List[str]) -> Dict:
    """用 LLM 对新闻文本打分。未配置 API 时返回中性分。"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not texts:
        return {
            "ts_code": ts_code,
            "event_risk_score": 0.0,
            "event_summary": "",
            "has_text": bool(texts),
        }

    prompt = (
        "你是 A股事件风险分析师。请根据以下近期新闻/公告，判断该股票未来 10 个交易日"
        "是否存在显著利空或高风险。输出一个 -1（强烈利空）到 +1（明显利好）之间的分数，"
        "并给出一句话总结。只返回 JSON：{'score': float, 'summary': str}\n\n" +
        "\n".join(texts[:10])
    )
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        content = resp.choices[0].message.content
        # 简单解析 JSON
        import json, re
        m = re.search(r"\{.*?\}", content, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            return {
                "ts_code": ts_code,
                "event_risk_score": float(obj.get("score", 0.0)),
                "event_summary": obj.get("summary", ""),
                "has_text": True,
            }
    except Exception as e:
        print(f"[llm_event_filter] LLM 评分失败 {ts_code}: {e}")

    return {
        "ts_code": ts_code,
        "event_risk_score": 0.0,
        "event_summary": "",
        "has_text": bool(texts),
    }


def filter_event_risk(ts_codes: List[str], date: str, lookback_days: int = 10,
                      risk_threshold: float = 0.5) -> pd.DataFrame:
    """
    对候选股票进行事件风险过滤。
    返回 DataFrame：ts_code, event_risk_score, event_summary, has_text, filtered_out。
    """
    end_date = date
    start = datetime.strptime(date, "%Y%m%d") - timedelta(days=lookback_days + 30)
    start_date = get_trade_date_before(date, days=lookback_days) if lookback_days > 0 else start.strftime("%Y%m%d")

    rows = []
    for code in ts_codes:
        texts = _get_news_text(code, start_date, end_date)
        rec = _score_with_llm(code, texts)
        rec["filtered_out"] = rec["event_risk_score"] >= risk_threshold
        rows.append(rec)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--ts-codes", required=True, help="逗号分隔的 ts_code")
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ts_codes = [c.strip() for c in args.ts_codes.split(",")]
    df = filter_event_risk(ts_codes, args.date, args.lookback_days, args.risk_threshold)
    print(df.to_string(index=False))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"[llm_event_filter] saved to {args.output}")


if __name__ == "__main__":
    main()
