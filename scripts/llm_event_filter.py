"""
LLM / 规则 事件风险过滤器

功能：对选股结果中的 top-k 候选，基于最近一段时间的个股公告标题，
用 LLM（若配置了 OPENAI_API_KEY）或规则关键词判断是否存在重大利空或高风险事件，
输出一个风险分数，供下游组合构造时降权或剔除。

数据来源优先级：
1. AKShare `stock_individual_notice_report`：免费、可回测、支持历史日期区间；
2. Tushare `major_news` / `news`：若账号有付费权限；
3. 未获取到文本时返回中性分。

用法：
  python scripts/llm_event_filter.py --date 20250402 --ts-codes 000001.SZ,600519.SH --lookback-days 10
"""
import sys
import os
import argparse
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import get_trade_date_before

NEGATIVE_KEYWORDS = [
    "诉讼", "仲裁", "处罚", "罚款", "监管", "警示", "立案", "调查", "退市",
    "亏损", "预亏", "减持", "质押", "违约", "债务", "查封", "冻结", "清算",
    "破产", "重整", "收购失败", "撤销", "终止", "关注函", "问询函", "监管函",
    "警示函", "责令改正", "内幕交易", "操纵市场", "行政处罚", "市场禁入",
    "失信被执行人", "高风险", "重大风险", "业绩下滑",
]

POSITIVE_KEYWORDS = [
    "增持", "回购", "预增", "中标", "签约", "重大合同", "收购", "重组",
    "股权激励", "分红", "派息", "转正", "扭亏", "批复", "核准", "通过",
    "合作协议", "战略合作", "重大项目",
]


try:
    from _tushare_utils import tushare_call
except Exception:
    tushare_call = None


try:
    import akshare as ak
    HAS_AKSHARE = True
except Exception:
    HAS_AKSHARE = False


def ts_code_to_symbol(ts_code: str) -> str:
    return ts_code.split(".")[0]


def _heuristic_score(titles: List[str]) -> tuple:
    """基于标题关键词计算 (risk_score, summary)。"""
    neg = pos = 0
    matched = []
    for t in titles:
        for kw in NEGATIVE_KEYWORDS:
            if kw in t:
                neg += 1
                matched.append(f"- {kw}")
        for kw in POSITIVE_KEYWORDS:
            if kw in t:
                pos += 1
                matched.append(f"+ {kw}")
    score = neg - pos
    summary = "; ".join(matched[:5]) if matched else ""
    return score, summary


def _get_akshare_announcements(ts_code: str, start_date: str, end_date: str) -> List[str]:
    """用 AKShare 抓取个股公告标题。"""
    if not HAS_AKSHARE:
        return []
    try:
        symbol = ts_code_to_symbol(ts_code)
        df = ak.stock_individual_notice_report(
            security=symbol, symbol="全部",
            begin_date=start_date, end_date=end_date,
        )
        if df.empty or "公告标题" not in df.columns:
            return []
        df["公告日期"] = pd.to_datetime(df["公告日期"], errors="coerce")
        end_dt = pd.to_datetime(end_date, format="%Y%m%d")
        df = df[df["公告日期"] <= end_dt]
        return df["公告标题"].astype(str).tolist()
    except Exception:
        return []


def _get_tushare_news_text(ts_code: str, start_date: str, end_date: str) -> List[str]:
    """尝试拉取 Tushare 个股新闻标题/摘要。Tushare 接口可能不可用。"""
    snippets = []
    if tushare_call is None:
        return snippets

    for api in ["major_news", "news"]:
        try:
            df = tushare_call(api, {"ts_code": ts_code, "start_date": start_date, "end_date": end_date})
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
    """用 LLM 对新闻文本打分。未配置 API 时使用规则关键词打分。"""
    if not texts:
        return {
            "ts_code": ts_code,
            "event_risk_score": 0.0,
            "event_summary": "",
            "has_text": False,
        }

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        score, summary = _heuristic_score(texts)
        return {
            "ts_code": ts_code,
            "event_risk_score": float(score),
            "event_summary": summary,
            "has_text": True,
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
        import json
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

    score, summary = _heuristic_score(texts)
    return {
        "ts_code": ts_code,
        "event_risk_score": float(score),
        "event_summary": summary,
        "has_text": True,
    }


def _get_texts(ts_code: str, start_date: str, end_date: str) -> List[str]:
    """聚合所有文本源。"""
    texts = _get_akshare_announcements(ts_code, start_date, end_date)
    if not texts:
        texts = _get_tushare_news_text(ts_code, start_date, end_date)
    return texts


def filter_event_risk(ts_codes: List[str], date: str, lookback_days: int = 10,
                      risk_threshold: float = 0.5) -> pd.DataFrame:
    """
    对候选股票进行事件风险过滤。
    返回 DataFrame：ts_code, event_risk_score, event_summary, has_text, filtered_out。
    """
    start = datetime.strptime(date, "%Y%m%d") - timedelta(days=lookback_days + 30)
    start_date = get_trade_date_before(date, days=lookback_days) if lookback_days > 0 else start.strftime("%Y%m%d")
    end_date = date

    rows = []
    for code in ts_codes:
        texts = _get_texts(code, start_date, end_date)
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
