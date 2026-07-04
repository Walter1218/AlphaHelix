"""
AlphaHelix 数据预取脚本

在跑 walk-forward 前，把指定窗口内需要用到的 Tushare 数据批量拉取到本地缓存，
避免选股时逐只逐日请求导致效率低下。同时通过 ALPHAHELIX_DATA_WINDOW 做上下文隔离，
确保预取范围就是脚本/智能体被允许看到的范围。

预取内容：
- 交易日历（全窗口）
- 沪深300 指数日线（含 lookback）
- stock_basic（当前上市列表）
- 每个交易日的 daily / daily_basic / moneyflow（按 trade_date 全截面）
- 每只股票的 fina_indicator / forecast / express（按 ts_code 全历史，用于窗口内公告过滤）

用法：
    ALPHAHELIX_RATE_LIMIT=0.02 ALPHAHELIX_MAX_WORKERS=8 \
    TUSHARE_TOKEN=xxx python scripts/prefetch_data.py \
        --start 20240101 --end 20260615 --lookback 120
"""
import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import tushare_call, get_trade_calendar, concurrent_map, get_trade_date_before


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d")


def format_date(d: datetime) -> str:
    return d.strftime("%Y%m%d")


def prefetch_calendar(start_date: str, end_date: str):
    """交易日历：一次调用覆盖足够宽范围。"""
    tushare_call("trade_cal", {
        "exchange": "SSE",
        "start_date": get_trade_date_before(start_date, days=180),
        "end_date": end_date,
    })
    print(f"[prefetch] calendar ok")


def prefetch_index(start_date: str, end_date: str):
    """沪深300 日线。"""
    tushare_call("index_daily", {
        "ts_code": "000300.SH",
        "start_date": start_date,
        "end_date": end_date,
    })
    print(f"[prefetch] index_daily ok")


def prefetch_stock_basic():
    """股票基础信息。"""
    df = tushare_call("stock_basic", {"exchange": "", "list_status": "L"})
    print(f"[prefetch] stock_basic ok, {len(df)} rows")
    return df


def prefetch_daily_by_date(date: str):
    """按 trade_date 获取全市场日线。"""
    try:
        tushare_call("daily", {"trade_date": date})
    except Exception as e:
        print(f"[prefetch] daily {date} failed: {e}")


def prefetch_daily_basic_by_date(date: str):
    """按 trade_date 获取全市场每日指标。"""
    try:
        tushare_call("daily_basic", {"trade_date": date})
    except Exception as e:
        print(f"[prefetch] daily_basic {date} failed: {e}")


def prefetch_moneyflow_by_date(date: str):
    """按 trade_date 获取全市场资金流向。"""
    try:
        tushare_call("moneyflow", {"trade_date": date})
    except Exception as e:
        print(f"[prefetch] moneyflow {date} failed: {e}")


def prefetch_by_dates(dates: list):
    """并发预取截面数据。"""
    print(f"[prefetch] fetching daily/daily_basic/moneyflow for {len(dates)} dates ...")
    concurrent_map(prefetch_daily_by_date, dates)
    concurrent_map(prefetch_daily_basic_by_date, dates)
    concurrent_map(prefetch_moneyflow_by_date, dates)


def prefetch_fina(ts_code: str):
    try:
        tushare_call("fina_indicator", {"ts_code": ts_code})
    except Exception:
        pass


def prefetch_forecast(ts_code: str):
    try:
        tushare_call("forecast", {"ts_code": ts_code})
    except Exception:
        pass


def prefetch_express(ts_code: str):
    try:
        tushare_call("express", {"ts_code": ts_code})
    except Exception:
        pass


def prefetch_by_stock(ts_codes: list):
    """并发预取个股财务/事件数据。"""
    print(f"[prefetch] fetching fina/forecast/express for {len(ts_codes)} stocks ...")
    concurrent_map(prefetch_fina, ts_codes)
    concurrent_map(prefetch_forecast, ts_codes)
    concurrent_map(prefetch_express, ts_codes)


def main():
    parser = argparse.ArgumentParser(description="Prefetch Tushare data for a walk-forward window")
    parser.add_argument("--start", required=True, help="Window start YYYYMMDD")
    parser.add_argument("--end", required=True, help="Window end YYYYMMDD")
    parser.add_argument("--lookback", type=int, default=120, help="Extra lookback days before start for momentum/indicator history")
    parser.add_argument("--max-stocks", type=int, default=None, help="Limit number of stocks for fina/forecast/express prefetch")
    args = parser.parse_args()

    # 数据上下文：窗口起点前推 lookback，终点不变
    window_start = format_date(parse_date(get_trade_date_before(args.start, days=args.lookback)) - timedelta(days=5))
    window_end = args.end
    os.environ["ALPHAHELIX_DATA_WINDOW_START"] = window_start
    os.environ["ALPHAHELIX_DATA_WINDOW_END"] = window_end
    print(f"[prefetch] data window set: {window_start} ~ {window_end}")

    prefetch_calendar(args.start, args.end)
    prefetch_index(window_start, window_end)
    df_basic = prefetch_stock_basic()

    # 生成窗口内所有交易日
    cal = get_trade_calendar("SSE", window_start, window_end)
    trade_dates = cal[cal["is_open"].astype(int) == 1]["cal_date"].astype(str).tolist()
    print(f"[prefetch] {len(trade_dates)} trade dates to prefetch")

    prefetch_by_dates(trade_dates)

    ts_codes = df_basic["ts_code"].tolist()
    if args.max_stocks:
        ts_codes = ts_codes[:args.max_stocks]
    prefetch_by_stock(ts_codes)

    print("[prefetch] done")


if __name__ == "__main__":
    main()
