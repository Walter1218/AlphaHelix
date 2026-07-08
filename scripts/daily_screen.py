"""
AlphaHelix 每日选股脚本（纯 Python 版本）

不依赖 HelixAgent，直接调用 screen.py 获取候选股，
生成 Markdown 报告和 JSON 快照。

用法：
    python scripts/daily_screen.py
    python scripts/daily_screen.py --date 20240102 --top-n 20
"""
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen import screen

STOCK_DIR = Path("memory/stock")
LOG_DIR = Path("memory/log")


def generate_report(date: str, picks: list, strategy: str, actual_strategy: str) -> str:
    """生成 Markdown 报告"""
    lines = [
        f"# AlphaHelix 每日选股报告",
        f"",
        f"> 日期：{date[:4]}-{date[4:6]}-{date[6:]}",
        f"> 策略：{actual_strategy}（请求：{strategy}）",
        f"> 候选数：{len(picks)}",
        f"",
        f"## Top {len(picks)} 候选股",
        f"",
        f"| 排名 | 代码 | 名称 | 行业 | GBDT得分 | 逻辑 |",
        f"|---|---|---|---|---|---|",
    ]
    
    for i, pick in enumerate(picks[:10], 1):
        ts_code = pick.get("ts_code", "")
        name = pick.get("name", "")
        industry = pick.get("industry", "")
        gbdt_score = pick.get("gbdt_score", pick.get("total_score", 0))
        rationale = pick.get("rationale", "动量+估值+质量")
        
        lines.append(f"| {i} | {ts_code} | {name} | {industry} | {gbdt_score:.4f} | {rationale} |")
    
    lines.extend([
        "",
        "## 风险提示",
        "",
        "- 市场存在不确定性，模型输出不保证未来收益",
        "- 建议结合基本面分析和市场情绪判断",
        "- 止损纪律：跌破支撑位及时离场",
        "",
        "AlphaHelix 研究团队对研究方法和数据质量负责，但不承诺收益。用户应基于自身判断做出投资决策。过往业绩不代表未来表现。",
    ])
    
    return "\n".join(lines)


def generate_json(date: str, picks: list, strategy: str, actual_strategy: str) -> dict:
    """生成 JSON 快照"""
    return {
        "date": date,
        "data_date": date,
        "strategy": strategy,
        "actual_strategy": actual_strategy,
        "picks": [
            {
                "ts_code": p.get("ts_code", ""),
                "name": p.get("name", ""),
                "industry": p.get("industry", ""),
                "score": p.get("gbdt_score", p.get("total_score", 0)),
                "rank": i + 1,
                "confidence": "medium",
                "stop_loss": 0,
                "rationale": p.get("rationale", "动量+估值+质量"),
            }
            for i, p in enumerate(picks[:10])
        ],
        "generated_at": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="AlphaHelix daily stock screening")
    parser.add_argument("--date", default=None, help="Trade date YYYYMMDD (default: today)")
    parser.add_argument("--strategy", default="regime", help="Screening strategy")
    parser.add_argument("--top-n", type=int, default=10, help="Number of picks")
    parser.add_argument("--use-gbdt", action="store_true", default=True, help="Use GBDT model")
    args = parser.parse_args()
    
    date = args.date or datetime.now().strftime("%Y%m%d")
    
    STOCK_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"[daily_screen] Running screen for {date}, strategy={args.strategy}, top_n={args.top_n}")
    
    try:
        picks = screen(
            date=date,
            strategy=args.strategy,
            top_n=args.top_n,
            use_gbdt_model=args.use_gbdt,
        )
        
        print(f"[daily_screen] Got {len(picks)} picks")
        
        # 生成报告
        md_report = generate_report(date, picks, args.strategy, args.strategy)
        md_path = STOCK_DIR / f"{date}.md"
        md_path.write_text(md_report, encoding="utf-8")
        print(f"[daily_screen] Report saved to {md_path}")
        
        # 生成 JSON 快照
        json_data = generate_json(date, picks, args.strategy, args.strategy)
        json_path = STOCK_DIR / f"{date}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"[daily_screen] Snapshot saved to {json_path}")
        
        # 打印摘要
        print(f"\n=== Top {min(10, len(picks))} Picks ===")
        for i, pick in enumerate(picks[:10], 1):
            ts_code = pick.get("ts_code", "")
            name = pick.get("name", "")
            score = pick.get("gbdt_score", pick.get("total_score", 0))
            print(f"  {i}. {ts_code} {name} (score={score:.4f})")
        
    except Exception as e:
        print(f"[daily_screen] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
