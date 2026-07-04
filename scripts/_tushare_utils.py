"""
AlphaHelix Tushare 共享工具模块
提供交易日历、数据缓存、限流等基础能力。
"""
import os
import time
import json
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import tushare as ts
import pandas as pd

# 缓存目录
CACHE_DIR = Path(os.environ.get("ALPHAHELIX_CACHE_DIR", ".cache/tushare"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 限流：每秒最多 1 次调用（免费用户保守设置）
RATE_LIMIT_INTERVAL = float(os.environ.get("ALPHAHELIX_RATE_LIMIT", "1.0"))
_last_call_time = 0.0
_rate_limit_lock = threading.Lock()

# 并发控制
MAX_WORKERS = int(os.environ.get("ALPHAHELIX_MAX_WORKERS", "4"))

# 受窗口隔离约束的数据接口（价格、估值、财务、资金、事件）
_DATA_APIS = {
    "daily", "daily_basic", "moneyflow", "fina_indicator", "forecast", "express",
    "index_daily", "index_weight", "index_classify",
    "margin", "moneyflow_hsgt", "top_list", "disclosure_date",
}

# 延迟初始化的 Tushare pro_api 实例
_pro = None


def _get_pro():
    """延迟初始化并返回 Tushare pro_api 实例。"""
    global _pro
    if _pro is None:
        token = os.environ.get("TUSHARE_TOKEN")
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is not set")
        ts.set_token(token)
        _pro = ts.pro_api()
    return _pro


def _rate_limit():
    global _last_call_time
    with _rate_limit_lock:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < RATE_LIMIT_INTERVAL:
            time.sleep(RATE_LIMIT_INTERVAL - elapsed)
        _last_call_time = time.time()


def _parse_date(d) -> str:
    """统一把 YYYYMMDD 或 Timestamp 转成字符串。"""
    if d is None:
        return None
    if isinstance(d, str):
        return d
    if isinstance(d, (int, float)):
        return str(int(d))
    if hasattr(d, "strftime"):
        return d.strftime("%Y%m%d")
    return str(d)


def _enforce_data_window(api_name: str, params: dict):
    """若设置了数据窗口，禁止数据接口请求窗口外的日期。"""
    if api_name not in _DATA_APIS:
        return
    start = os.environ.get("ALPHAHELIX_DATA_WINDOW_START")
    end = os.environ.get("ALPHAHELIX_DATA_WINDOW_END")
    if not start or not end:
        return

    def _check(field: str):
        val = _parse_date(params.get(field))
        if val is None:
            return
        if val < start or val > end:
            raise RuntimeError(
                f"Data context isolation: {api_name}.{field}={val} is outside "
                f"allowed window [{start}, {end}]."
            )

    # trade_date 必须落在窗口内；start_date/end_date 必须完全在窗口内
    _check("trade_date")
    req_start = _parse_date(params.get("start_date"))
    req_end = _parse_date(params.get("end_date"))
    if req_start and req_start < start:
        raise RuntimeError(
            f"Data context isolation: {api_name}.start_date={req_start} is before "
            f"allowed window start {start}."
        )
    if req_end and req_end > end:
        raise RuntimeError(
            f"Data context isolation: {api_name}.end_date={req_end} is after "
            f"allowed window end {end}."
        )


def concurrent_map(func, items, max_workers: int = None):
    """并发执行 func(item)，保留顺序返回结果。任一任务异常会立即抛出。"""
    if max_workers is None:
        max_workers = MAX_WORKERS
    if max_workers <= 1 or len(items) <= 1:
        return [func(x) for x in items]

    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
    return results


def _cache_key(api_name: str, params: dict) -> str:
    payload = json.dumps({"api": api_name, "params": params}, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()


def _cache_path(api_name: str, params: dict) -> Path:
    return CACHE_DIR / f"{api_name}_{_cache_key(api_name, params)}.json"


def tushare_call(api_name: str, params: dict, use_cache: bool = True) -> pd.DataFrame:
    """带缓存、限流和数据上下文隔离的 Tushare 调用，返回 DataFrame"""
    _enforce_data_window(api_name, params)
    cache_path = _cache_path(api_name, params)
    if use_cache and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached and "data" in cached:
            return pd.DataFrame(cached["data"].get("items", []), columns=cached["data"].get("fields", []))

    _rate_limit()
    resp = _get_pro().query(api_name=api_name, **params)

    # tushare pro_api().query 通常直接返回 DataFrame
    if resp is None:
        resp = pd.DataFrame()

    if use_cache and not resp.empty:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"data": {"fields": list(resp.columns), "items": resp.values.tolist()}}, f, ensure_ascii=False)

    return resp


_trade_cal_cache: dict = {}
_name_history_cache: dict = {}


def get_name_history(ts_code: str) -> pd.DataFrame:
    """获取股票历史名称变更记录，带内存缓存。"""
    if ts_code not in _name_history_cache:
        df = tushare_call("namechange", {"ts_code": ts_code})
        if not df.empty:
            df["start_date"] = df["start_date"].astype(str)
            df["end_date"] = df["end_date"].fillna("99991231").astype(str)
            df["ann_date"] = df["ann_date"].fillna(df["start_date"]).astype(str)
        _name_history_cache[ts_code] = df
    return _name_history_cache[ts_code]


def is_st_historical(ts_code: str, trade_date: str) -> bool:
    """判断 trade_date 当天股票是否为 ST/*ST/退市。
    依据 namechange 接口的历史名称记录，避免用当前名字判断历史状态。"""
    df = get_name_history(ts_code)
    if df.empty:
        # 拿不到历史名称时保守处理：按当前名称判断（仅用于 live，backtest 应人工复核）
        return False
    td = trade_date
    mask = (df["start_date"] <= td) & (df["end_date"] >= td)
    names = df.loc[mask, "name"].tolist()
    if not names:
        return False
    # 任一匹配时段名称为 ST/*ST/退，即视为 ST
    return any("ST" in n or "退" in n for n in names)


def get_trade_calendar(exchange: str = "SSE", start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """获取交易日历，带缓存"""
    key = f"{exchange}:{start_date}:{end_date}"
    if key not in _trade_cal_cache:
        df = tushare_call("trade_cal", {
            "exchange": exchange,
            "start_date": start_date or "20150101",
            "end_date": end_date or datetime.now().strftime("%Y%m%d"),
        })
        _trade_cal_cache[key] = df
    return _trade_cal_cache[key]


def get_trade_date_before(date: str, days: int = 0, exchange: str = "SSE") -> str:
    """获取指定日期前第 N 个交易日（按真实交易日历）"""
    target = datetime.strptime(date, "%Y%m%d")
    # 取足够宽的范围，覆盖超长假期
    start = (target - timedelta(days=days * 3 + 60)).strftime("%Y%m%d")
    end = date
    cal = get_trade_calendar(exchange, start, end)
    cal = cal[cal["is_open"].astype(int) == 1].sort_values("cal_date", ascending=False)
    if len(cal) <= days:
        return cal.iloc[-1]["cal_date"] if not cal.empty else date
    return cal.iloc[days]["cal_date"]


def get_trade_date_after(date: str, days: int = 0, exchange: str = "SSE") -> str:
    """获取指定日期后第 N 个交易日（按真实交易日历）"""
    target = datetime.strptime(date, "%Y%m%d")
    start = date
    end = (target + timedelta(days=days * 3 + 60)).strftime("%Y%m%d")
    cal = get_trade_calendar(exchange, start, end)
    cal = cal[cal["is_open"].astype(int) == 1].sort_values("cal_date", ascending=True)
    if len(cal) <= days:
        return cal.iloc[-1]["cal_date"] if not cal.empty else date
    return cal.iloc[days]["cal_date"]


# -----------------------------------------------------------------------------
# Phase 2 另类数据获取（融资融券、北向资金、龙虎榜、披露日）
# -----------------------------------------------------------------------------

_margin_cache: dict = {}
_northbound_cache: dict = {}
_top_list_cache: dict = {}
_disclosure_cache: dict = {}


def fetch_margin_daily(trade_date: str, use_cache: bool = True) -> pd.DataFrame:
    """获取全市场融资融券余额（交易所维度），返回按交易所汇总。"""
    key = trade_date
    if key not in _margin_cache:
        df = tushare_call("margin", {"trade_date": trade_date}, use_cache=use_cache)
        _margin_cache[key] = df
    return _margin_cache[key]


def fetch_northbound_daily(trade_date: str, use_cache: bool = True) -> pd.DataFrame:
    """获取北向资金（沪深港通）每日净流入。"""
    key = trade_date
    if key not in _northbound_cache:
        df = tushare_call("moneyflow_hsgt", {"trade_date": trade_date}, use_cache=use_cache)
        _northbound_cache[key] = df
    return _northbound_cache[key]


def fetch_top_list(trade_date: str, use_cache: bool = True) -> pd.DataFrame:
    """获取龙虎榜数据（全市场），返回 DataFrame。"""
    key = trade_date
    if key not in _top_list_cache:
        df = tushare_call("top_list", {"trade_date": trade_date}, use_cache=use_cache)
        _top_list_cache[key] = df
    return _top_list_cache[key]


def fetch_disclosure_schedule(years: list = None, use_cache: bool = True) -> pd.DataFrame:
    """获取年报/季报预约披露时间表，按 ann_date 过滤。

    返回列：ts_code, ann_date, end_date, pre_date, actual_date。
    对于 2024/2025 年报告期，分别拉取对应 ann_date 区间并合并。
    """
    if years is None:
        years = [2024, 2025]
    key = tuple(years)
    if key not in _disclosure_cache:
        parts = []
        for y in years:
            df = tushare_call("disclosure_date", {
                "start_date": f"{y}0101",
                "end_date": f"{y}1231",
            }, use_cache=use_cache)
            if not df.empty:
                parts.append(df)
        _disclosure_cache[key] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return _disclosure_cache[key]
