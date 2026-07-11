"""
个股 GBDT 预测脚本

用法：
    python predict_stock.py 600036.SH 20260709
    python predict_stock.py 600036.SH 300750.SZ 20260709

输出每只股票的预测得分和全市场排名百分位。
"""
import sys
import os
import json
import argparse
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tushare_utils import tushare_call, get_trade_date_before
from feature_engineering import build_numeric_features
from gbdt_predictor import GBDTScorePredictor, find_latest_model
from dividend_features import calculate_dividend_features

warnings.filterwarnings("ignore")

# 缓存目录
CACHE_DIR = Path("memory/cache/predict_stock")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 最优配置
OPTIMAL_FEATURES = [
    'relative_to_sector', 'mom_120', 'mom_60', 'sector_momentum', 'mom_20',
    'volatility_20', 'liquidity', 'sector_breadth', 'margin_total_balance',
    'risk_adj_mom', 'northbound_net_5d', 'bp', 'mom_5', 'defensive_quality',
    'top_list_flag', 'top_list_turnover_rate', 'risk_adj_momentum_20',
    'top_list_amount_rate', 'value_quality', 'amount_ratio_5d', 'dv_ratio',
    'northbound_net', 'reversal_score', 'sp', 'top_list_pct_change', 'roe',
    'forecast_type_score', 'days_to_disclosure', 'revenue_growth',
    'earnings_surprise_momentum'
]
OPTIMAL_ALPHA = 5.0


def _cache_key(ts_code: str, date: str) -> str:
    return f"{ts_code}_{date}"


def _load_cache(ts_code: str, date: str) -> dict:
    path = CACHE_DIR / f"{_cache_key(ts_code, date)}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(ts_code: str, date: str, data: dict):
    path = CACHE_DIR / f"{_cache_key(ts_code, date)}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_stock_data(ts_code: str, date: str, lookback_days: int = 150) -> pd.DataFrame:
    """加载个股历史数据。"""
    start_date = get_trade_date_before(date, days=lookback_days)
    df = tushare_call("daily", {
        "ts_code": ts_code,
        "start_date": start_date,
        "end_date": date,
    })
    if df.empty:
        return df
    df["trade_date"] = df["trade_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    return df


def load_moneyflow(ts_code: str, date: str, lookback_days: int = 30) -> pd.DataFrame:
    """加载个股资金流向。"""
    start_date = get_trade_date_before(date, days=lookback_days)
    df = tushare_call("moneyflow", {
        "ts_code": ts_code,
        "start_date": start_date,
        "end_date": date,
    })
    if not df.empty:
        df["trade_date"] = df["trade_date"].astype(str)
        df["ts_code"] = df["ts_code"].astype(str)
    return df


def _rsi(close, period=14):
    if len(close) < period + 1:
        return 50
    deltas = np.diff(close[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _ema(data, span):
    if len(data) < span:
        return data[-1] if len(data) > 0 else 0
    alpha = 2 / (span + 1)
    ema = data[0]
    for v in data[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema


def _atr(df, period=14):
    if len(df) < period + 1:
        return 0
    high = df["high"].values[-(period+1):]
    low = df["low"].values[-(period+1):]
    close = df["close"].values[-(period+1):]
    trs = []
    for i in range(1, len(high)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        trs.append(tr)
    return np.mean(trs) if trs else 0


def _reversal_score(close, period=20):
    """反转得分：近期跌幅越大，反转得分越高。"""
    if len(close) < period + 1:
        return 0
    ret = close[-1] / close[-period - 1] - 1
    return -ret  # 跌得越多，反转得分越高


def _risk_adj_mom(close, period=20):
    """风险调整动量：动量/波动率。"""
    if len(close) < period + 1:
        return 0
    mom = close[-1] / close[-period - 1] - 1
    vol = np.std(np.diff(np.log(close[-period:])))
    if vol == 0:
        return 0
    return mom / vol


def _relative_strength(close, period=60):
    """相对强度：近期动量 vs 长期动量。"""
    if len(close) < period + 1:
        return 0
    short_mom = close[-1] / close[-21] - 1 if len(close) >= 21 else 0
    long_mom = close[-1] / close[-period - 1] - 1
    if long_mom == 0:
        return 0
    return short_mom / abs(long_mom)


def _vol_regime(close, period=60):
    """波动率状态：当前波动率 vs 历史波动率。"""
    if len(close) < period + 1:
        return 0
    current_vol = np.std(np.diff(np.log(close[-21:]))) if len(close) >= 21 else 0
    hist_vol = np.std(np.diff(np.log(close[-period:])))
    if hist_vol == 0:
        return 0
    return current_vol / hist_vol


def compute_basic_features(ts_code: str, date: str, use_cache: bool = True) -> dict:
    """为单只股票计算基础因子特征。"""
    # 检查缓存
    if use_cache:
        cached = _load_cache(ts_code, date)
        if cached is not None:
            return cached

    daily = load_stock_data(ts_code, date, lookback_days=150)
    if daily.empty or len(daily) < 60:
        return None

    daily = daily.sort_values("trade_date").reset_index(drop=True)
    latest = daily.iloc[-1]

    close = daily["close"].values
    vol = daily["vol"].values
    amount = daily["amount"].values

    # 获取行业信息
    industry = ""
    try:
        stock_basic = tushare_call("stock_basic", {"ts_code": ts_code})
        if not stock_basic.empty:
            industry = stock_basic.iloc[0].get("industry", "未知")
    except Exception:
        pass

    result = {
        "ts_code": ts_code,
        "date": date,
        "close": float(latest["close"]),
        "industry": industry,
        "total_mv": float(latest.get("total_mv", 0)) if pd.notna(latest.get("total_mv")) else 0,

        # 动量
        "mom_5": float((close[-1] / close[-6] - 1)) if len(close) >= 6 else 0,
        "mom_20": float((close[-1] / close[-21] - 1)) if len(close) >= 21 else 0,
        "mom_60": float((close[-1] / close[-61] - 1)) if len(close) >= 61 else 0,
        "mom_120": float((close[-1] / close[-121] - 1)) if len(close) >= 121 else 0,

        # 波动率
        "volatility_20": float(np.std(np.diff(np.log(close[-21:])))) if len(close) >= 21 else 0,

        # 量比
        "amount_ratio_5d": float(amount[-1] / np.mean(amount[-6:-1])) if len(amount) >= 6 and np.mean(amount[-6:-1]) > 0 else 1,

        # RSI
        "rsi": float(_rsi(close, 14)),

        # MACD
        "macd": float(_ema(close, 12) - _ema(close, 26)),
        "macd_signal": float(_ema(close, 9)),
        "macd_hist": float(_ema(close, 12) - _ema(close, 26) - _ema(close, 9)),

        # ATR
        "atr": float(_atr(daily, 14)),

        # 可计算的特征
        "reversal_score": float(_reversal_score(close, 20)),
        "risk_adj_mom": float(_risk_adj_mom(close, 20)),
        "relative_strength": float(_relative_strength(close, 60)),
        "risk_adj_momentum_20": float(_risk_adj_mom(close, 20)),
        "vol_regime": float(_vol_regime(close, 60)),

        # 交互特征
        "mom_x_vol": float(abs(close[-1] / close[-21] - 1) * np.std(np.diff(np.log(close[-21:])))) if len(close) >= 21 else 0,
        "flow_x_vol": 0,  # 需要资金流数据

        # 需要基本面数据的特征（设为中性值）
        "dv_ratio": 0,
        "sector_momentum": 0,
        "relative_to_sector": 0,
        "sector_breadth": 0,
        "roe": 0,
        "revenue_growth": 0,
        "profit_growth": 0,
        "ocf_growth": 0,
        "forecast_type_score": 0,
        "forecast_pchange_mid": 0,
        "express_diluted_roe": 0,
        "defensive_quality": 0,
        "smart_money_per_risk": 0,
        "quality_growth": 0,
        "value_quality": 0,
        "earnings_surprise_momentum": 0,
        "growth_consistency": 0,
        "quality_x_growth": 0,
        "value_x_size": 0,

        # 截面排名特征（单股票用中性值）
        "mom_20_sector_rank": 0.5,
        "volatility_20_sector_rank": 0.5,
        "total_mv_sector_rank": 0.5,
        "roe_sector_rank": 0.5,
        "market_breadth": 0.5,
        "market_volatility": 0.02,
    }

    # 资金流向特征（可选，失败不影响）
    try:
        mf = load_moneyflow(ts_code, date, lookback_days=30)
        if not mf.empty and len(mf) >= 5:
            net = mf["net_mf_amount"].values
            result["net_mf_ratio"] = float(net[-1] / (abs(net[-1]) + 1e-6)) if not np.isnan(net[-1]) else 0
            result["net_mf_divergence"] = float((net[-1] - np.mean(net[-5:])) / (np.std(net[-5:]) + 1e-6)) if np.std(net[-5:]) > 0 else 0
    except Exception:
        pass

    # 分红特征（可选，失败不影响）
    try:
        div_features = calculate_dividend_features(ts_code, date, latest["close"])
        result.update(div_features)
    except Exception:
        pass

    # 保存缓存
    if use_cache:
        _save_cache(ts_code, date, result)

    return result


def predict_stocks(ts_codes: list, date: str, use_cache: bool = True):
    """预测指定股票的 GBDT 得分。"""
    # 尝试从预测文件加载（最优模型）
    pred_file = Path("memory/predictions/predictions_h10_stacking_5model.parquet")
    pred_df = None
    if pred_file.exists():
        pred_df = pd.read_parquet(pred_file)
        pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.strftime("%Y%m%d")

    # 检查哪些股票在预测文件中
    date_preds = None
    if pred_df is not None:
        date_preds = pred_df[pred_df["date"] == date]

    # 分类：在候选池中的 vs 不在的
    in_pool = []
    not_in_pool = []
    for code in ts_codes:
        if date_preds is not None and not date_preds.empty:
            stock_pred = date_preds[date_preds["ts_code"] == code]
            if not stock_pred.empty:
                in_pool.append(code)
                continue
        not_in_pool.append(code)

    # 输出在候选池中的股票（最优模型）
    if in_pool:
        print(f"使用最优模型预测（Ridge alpha=5.0, Top-30 特征）")
        print(f"\n{'='*60}")
        print(f"个股预测  基准日: {date}")
        print(f"模型: Ridge alpha=5.0, Top-30 特征, 12个月训练窗口")
        print(f"{'='*60}")

        date_preds_copy = date_preds.copy()
        date_preds_copy["rank"] = date_preds_copy["predicted"].rank(ascending=False)

        for code in in_pool:
            stock_pred = date_preds[date_preds["ts_code"] == code]
            row = stock_pred.iloc[0]
            rank_row = date_preds_copy[date_preds_copy["ts_code"] == code].iloc[0]

            print(f"\n{code} ({row.get('industry', '')}):")
            print(f"  预测得分: {row['predicted']:.6f}")
            print(f"  排名: {int(rank_row['rank'])} / {len(date_preds)}")

            direction = "看多 ↑" if row["predicted"] > 0 else "看空 ↓"
            print(f"  信号方向: {direction}")
            strength = abs(row["predicted"])
            if strength > 0.01:
                level = "强"
            elif strength > 0.005:
                level = "中"
            else:
                level = "弱"
            print(f"  信号强度: {level}")

    # 输出不在候选池中的股票（单模型预测）
    if not_in_pool:
        print(f"\n以下股票不在候选池中，使用 LightGBM 单模型预测：")
        model_path = find_latest_model()
        if model_path is None:
            print("错误：找不到 GBDT 模型文件")
            return
        predictor = GBDTScorePredictor(str(model_path))

        records = []
        for code in not_in_pool:
            print(f"正在计算 {code} 的特征...")
            feat = compute_basic_features(code, date, use_cache=use_cache)
            if feat is None:
                print(f"警告：{code} 数据不足，跳过")
                continue
            records.append(feat)
            time.sleep(1)

        if records:
            df = pd.DataFrame(records)
            scores = predictor.predict(df, skip_rank=True)
            df["gbdt_score"] = scores

            print(f"\n{'='*60}")
            print(f"个股预测  基准日: {date}")
            print(f"模型: {model_path.name} (单模型)")
            print(f"{'='*60}")

            for _, row in df.iterrows():
                print(f"\n{row['ts_code']} ({row.get('industry', '')}):")
                print(f"  收盘价: {row['close']:.2f}")
                print(f"  GBDT 得分: {row['gbdt_score']:.6f}")
                direction = "看多 ↑" if row["gbdt_score"] > 0 else "看空 ↓"
                print(f"  信号方向: {direction}")
                strength = abs(row["gbdt_score"])
                if strength > 0.01:
                    level = "强"
                elif strength > 0.005:
                    level = "中"
                else:
                    level = "弱"
                print(f"  信号强度: {level}")

    print(f"\n{'='*60}")
    print("说明：")
    print("  - 候选池内: 使用 Ridge alpha=5.0, Top-30 特征模型")
    print("  - 候选池外: 使用 LightGBM 单模型预测")
    print("  - 得分 > 0 表示预期正收益（看多）")
    print("  - 得分 < 0 表示预期负收益（看空）")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="个股 GBDT 预测")
    parser.add_argument("codes", nargs="+", help="股票代码，如 600036.SH 300750.SZ")
    parser.add_argument("--date", "-d", default="", help="基准日 YYYYMMDD，默认最近交易日")
    parser.add_argument("--no-cache", action="store_true", help="不使用缓存")
    args = parser.parse_args()

    date = args.date
    if not date:
        from datetime import datetime, timedelta
        # 往前找最近交易日
        d = datetime.now()
        for _ in range(10):
            ds = d.strftime("%Y%m%d")
            test = tushare_call("daily", {"trade_date": ds, "limit": 1})
            if not test.empty:
                date = ds
                break
            d -= timedelta(days=1)
        if not date:
            print("错误：无法确定最近交易日")
            return

    predict_stocks(args.codes, date, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
