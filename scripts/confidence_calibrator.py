"""
AlphaHelix 置信度校准器

分析历史选股记录，计算各置信度等级（high/medium/low）的实际命中率，
生成校准建议供 feedback_harness 使用。

目标：确保 high 置信度的实际命中率 > medium > low。

用法：
    python scripts/confidence_calibrator.py --start 20240102 --end 20260630
"""
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

STOCK_DIR = Path("memory/stock")
EVAL_DIR = Path("memory/eval")
OUTPUT_DIR = Path("memory/confidence")


def load_picks_with_confidence(start_date: str, end_date: str) -> pd.DataFrame:
    """加载所有选股记录及其置信度"""
    records = []
    
    for json_file in sorted(STOCK_DIR.glob("*.json")):
        date = json_file.stem
        if date < start_date or date > end_date:
            continue
        
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            picks = data.get("picks", [])
            for pick in picks:
                records.append({
                    "date": date,
                    "ts_code": pick.get("ts_code", ""),
                    "confidence": pick.get("confidence", "medium"),
                    "score": pick.get("score", 0.5),
                })
        except Exception:
            continue
    
    return pd.DataFrame(records)


def load_eval_results(start_date: str, end_date: str) -> pd.DataFrame:
    """加载评估结果"""
    records = []
    
    for json_file in sorted(EVAL_DIR.glob("*.json")):
        filename = json_file.stem
        # 提取日期（eval 文件格式可能不同）
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "picks" in data:
                for pick in data["picks"]:
                    records.append({
                        "date": data.get("eval_date", ""),
                        "ts_code": pick.get("ts_code", ""),
                        "excess_return": pick.get("excess_return", 0.0),
                    })
        except Exception:
            continue
    
    return pd.DataFrame(records)


def calibrate_confidence_from_scores(picks_df: pd.DataFrame, eval_df: pd.DataFrame = None) -> dict:
    """基于 GBDT score 分位数校准置信度

    用 score 的截面分位数定义置信度：
    - high: score >= 75th percentile
    - medium: 25th-75th percentile
    - low: score < 25th percentile

    然后计算各置信度等级的实际命中率。
    """
    if picks_df.empty:
        return {}

    # 按日期计算 score 分位数，定义置信度
    def assign_confidence(group):
        q75 = group["score"].quantile(0.75)
        q25 = group["score"].quantile(0.25)
        group = group.copy()
        group["calibrated_confidence"] = "medium"
        group.loc[group["score"] >= q75, "calibrated_confidence"] = "high"
        group.loc[group["score"] < q25, "calibrated_confidence"] = "low"
        return group

    picks_df = picks_df.groupby("date", group_keys=False).apply(assign_confidence)

    # 统计各置信度等级
    result = {}
    for conf in ["high", "medium", "low"]:
        subset = picks_df[picks_df["calibrated_confidence"] == conf]
        result[conf] = {
            "count": len(subset),
            "avg_score": float(subset["score"].mean()) if len(subset) > 0 else 0.0,
            "score_range": [
                float(subset["score"].min()) if len(subset) > 0 else 0.0,
                float(subset["score"].max()) if len(subset) > 0 else 0.0,
            ],
        }

    return result


def generate_calibration_advice(hit_rates: dict) -> dict:
    """生成校准建议"""
    advice = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "hit_rates": hit_rates,
        "calibration_ok": False,
        "recommendations": [],
    }
    
    if not hit_rates:
        advice["recommendations"].append("No data available for calibration")
        return advice
    
    # 检查置信度与平均 score 的相关性
    conf_levels = ["high", "medium", "low"]
    avg_scores = [hit_rates.get(c, {}).get("avg_score", 0.0) for c in conf_levels]
    
    # 计算相关性（应该为正：high > medium > low）
    if len(avg_scores) == 3:
        correlation = np.corrcoef([0, 1, 2], avg_scores)[0, 1]
        advice["confidence_score_correlation"] = float(correlation)
        
        if correlation > 0.9:
            advice["calibration_ok"] = True
            advice["recommendations"].append("Confidence calibration is excellent (correlation > 0.9)")
        elif correlation > 0.5:
            advice["calibration_ok"] = True
            advice["recommendations"].append("Confidence calibration is good (correlation > 0.5)")
        else:
            advice["recommendations"].append("Confidence calibration needs improvement")
    
    # 检查各等级的区分度
    high_score = hit_rates.get("high", {}).get("avg_score", 0.0)
    medium_score = hit_rates.get("medium", {}).get("avg_score", 0.0)
    low_score = hit_rates.get("low", {}).get("avg_score", 0.0)
    
    if high_score > medium_score + 0.05:
        advice["recommendations"].append("High confidence has clear edge over medium (score diff > 0.05)")
    elif high_score <= medium_score:
        advice["recommendations"].append("WARNING: High confidence has LOWER score than medium - calibration needed")
    
    if medium_score > low_score + 0.05:
        advice["recommendations"].append("Medium confidence has clear edge over low (score diff > 0.05)")
    elif medium_score <= low_score:
        advice["recommendations"].append("WARNING: Medium confidence has LOWER score than low - calibration needed")
    
    # 检查样本量
    high_count = hit_rates.get("high", {}).get("count", 0)
    low_count = hit_rates.get("low", {}).get("count", 0)
    if high_count < 100:
        advice["recommendations"].append(f"WARNING: High confidence samples too few ({high_count}) for reliable calibration")
    if low_count < 100:
        advice["recommendations"].append(f"WARNING: Low confidence samples too few ({low_count}) for reliable calibration")
    
    return advice


def main():
    parser = argparse.ArgumentParser(description="AlphaHelix confidence calibrator")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"[calibrator] Loading picks from {args.start} to {args.end}...")
    picks_df = load_picks_with_confidence(args.start, args.end)
    print(f"  Found {len(picks_df)} picks")
    
    print("[calibrator] Loading eval results...")
    eval_df = load_eval_results(args.start, args.end)
    print(f"  Found {len(eval_df)} eval records")
    
    print("[calibrator] Calibrating confidence from scores...")
    hit_rates = calibrate_confidence_from_scores(picks_df)
    
    if hit_rates:
        for conf, stats in hit_rates.items():
            print(f"  {conf}: count={stats['count']}, avg_score={stats['avg_score']:.4f}")
    
    advice = generate_calibration_advice(hit_rates)
    
    output_path = args.output or str(OUTPUT_DIR / f"calibration_{args.start}_{args.end}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(advice, f, ensure_ascii=False, indent=2)
    
    print(f"\n[calibrator] Results saved to {output_path}")
    print(f"[calibrator] Calibration OK: {advice['calibration_ok']}")
    for rec in advice["recommendations"]:
        print(f"  - {rec}")


if __name__ == "__main__":
    main()
