"""
AlphaHelix Trace 模块

提供轻量级、结构化的 trace 采集与持久化能力。
每个 trace 事件写入 memory/trace/YYYYMMDD.jsonl，便于后续 case 分析、策略迭代和 DPO 数据集构建。

设计原则：
- 不依赖 HelixAgent 内部 trace，完全在 AlphaHelix 侧可控。
- 对核心脚本零侵入或极小侵入：通过 trace_event() 记录关键节点。
- JSONL 格式，方便追加和按行读取。
"""
import os
import json
import uuid
from datetime import datetime
from pathlib import Path

TRACE_DIR = Path("memory/trace")

# 单次选股/回测会话 ID，用于关联同一 run 内的多个事件
_RUN_ID = str(uuid.uuid4())[:8]


def _sanitize(value, max_len: int = 2000):
    """对 trace 字段做安全截断，避免单个事件过大。"""
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + f"...({len(value) - max_len} chars omitted)"
    if isinstance(value, list) and len(value) > 200:
        return value[:200] + [f"...({len(value) - 200} items omitted)"]
    if isinstance(value, dict):
        return {k: _sanitize(v, max_len) for k, v in value.items()}
    return value


def trace_event(step: str, data: dict, date: str = None, strategy: str = None):
    """记录一个 trace 事件。

    Args:
        step: 事件类型，如 screen.start, screen.end, evaluate, feedback.update, walkforward.period
        data: 任意可 JSON 序列化的字典，通常包含 inputs/outputs/metadata
        date: 交易日期（YYYYMMDD），用于决定文件名；未提供则使用当前日期
        strategy: 策略名，可选，会注入 metadata
    """
    if not date:
        date = datetime.now().strftime("%Y%m%d")

    event = {
        "timestamp": datetime.now().isoformat(),
        "run_id": _RUN_ID,
        "step": step,
        "date": date,
        "strategy": strategy,
    }
    # 把 data 放到 payload，先做 sanitize
    event["payload"] = _sanitize(data)

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_file = TRACE_DIR / f"{date}.jsonl"
    with open(trace_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(obj):
    """处理 numpy/pandas 等不可直接序列化的类型。"""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def new_run():
    """显式开启一次新 run，生成新的 run_id。"""
    global _RUN_ID
    _RUN_ID = str(uuid.uuid4())[:8]
    return _RUN_ID
