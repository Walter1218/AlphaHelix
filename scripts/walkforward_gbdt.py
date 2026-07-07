"""
AlphaHelix GBDT 滚动回测主链路

流程：
1. 读取或生成 walk-forward 预测（date / ts_code / predicted）；
2. 每个再平衡日按 GBDT 得分选 top-k，应用行业集中度约束；
3. 等权持有到下一再平衡日，扣除交易成本；
4. 输出累计绩效，并把每期评估写入 memory/eval/。

本脚本对应生产主链路：screen.py --use-gbdt 选股 → 等权持仓 → evaluate.py 评估。
"""
import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_trainer import load_dataset, get_feature_cols, walk_forward_predict
from portfolio_backtest import run_backtest as _run_portfolio_backtest
from walkforward_threshold import calibrate_and_mask

OUTPUT_DIR = Path("memory/eval")
PRED_DIR = Path("memory/predictions")


def generate_predictions(dataset_path: str, horizon: int, model_type: str,
                         train_window_months: int, target: str) -> pd.DataFrame:
    """如果未提供预测文件，则运行 walk-forward 生成。"""
    df = load_dataset(horizon, dataset_path)
    feature_cols = get_feature_cols(df)
    pred_df = walk_forward_predict(df, feature_cols,
                                   train_window_months=train_window_months,
                                   model_type=model_type,
                                   target=target)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"predictions_h{horizon}_walkforward_{target}_{model_type}.parquet"
    output_path = PRED_DIR / output_name
    pred_df.to_parquet(output_path, index=False)
    print(f"[walkforward_gbdt] Saved predictions to {output_path}")
    return pred_df


def run_walkforward_gbdt(pred_path: str = None,
                         dataset_path: str = None,
                         horizon: int = 10,
                         model_type: str = "lightgbm",
                         train_window_months: int = 12,
                         target: str = "excess_return",
                         max_positions: int = 20,
                         max_sector_pct: float = 0.4,
                         commission: float = 0.0002,
                         stamp_tax: float = 0.001,
                         slippage: float = 0.001,
                         pred_threshold: float = None,
                         stop_loss_pct: float = None,
                         start_date: str = None,
                         end_date: str = None,
                         use_wf_threshold: bool = False,
                         wf_train_periods: int = 12,
                         wf_metric: str = "win_rate",
                         weight_scheme: str = "equal",
                         max_sector_weight: float = 1.0,
                         neutralize_market_cap: bool = False,
                          macro_dataset: str = None,
                          macro_regime_threshold: float = None) -> dict:
    if macro_dataset:
        print("[WARNING] Macro timing has been shown to HURT performance (-60% cumulative excess).")
        print("[WARNING] Consider removing --macro-dataset for better results.")
    if pred_path:
        pred_df = pd.read_parquet(pred_path)
    elif dataset_path:
        pred_df = generate_predictions(dataset_path, horizon, model_type,
                                       train_window_months, target)
    else:
        raise ValueError("必须提供 --pred-path 或 --dataset")

    pred_df["date"] = pd.to_datetime(pred_df["date"])
    if start_date:
        pred_df = pred_df[pred_df["date"] >= pd.to_datetime(start_date, format="%Y%m%d")]
    if end_date:
        pred_df = pred_df[pred_df["date"] <= pd.to_datetime(end_date, format="%Y%m%d")]

    # Walk-forward 阈值校准后处理
    if use_wf_threshold:
        print(f"[walkforward_gbdt] Applying walk-forward threshold calibration (train_periods={wf_train_periods}, metric={wf_metric})")
        pred_tmp = PRED_DIR / f"_tmp_pred_for_wfthresh_{datetime.now().strftime('%H%M%S')}.parquet"
        pred_df.to_parquet(pred_tmp, index=False)
        masked_path = PRED_DIR / f"predictions_h{horizon}_walkforward_{target}_{model_type}_wfthresh.parquet"
        calibrate_and_mask(
            pred_path=str(pred_tmp),
            output_path=str(masked_path),
            train_periods=wf_train_periods,
            max_positions=max_positions,
            metric=wf_metric,
        )
        pred_tmp.unlink(missing_ok=True)
        pred_df = pd.read_parquet(masked_path)

    # 保存临时预测文件供 portfolio_backtest 读取
    tmp_path = PRED_DIR / f"_tmp_walkforward_gbdt_{datetime.now().strftime('%H%M%S')}.parquet"
    pred_df.to_parquet(tmp_path, index=False)

    summary = _run_portfolio_backtest(
        str(tmp_path),
        max_positions=max_positions,
        max_sector_pct=max_sector_pct,
        max_sector_weight=max_sector_weight,
        commission=commission,
        stamp_tax=stamp_tax,
        slippage=slippage,
        pred_threshold=pred_threshold,
        stop_loss_pct=stop_loss_pct,
        weight_scheme=weight_scheme,
        neutralize_market_cap=neutralize_market_cap,
        macro_dataset=macro_dataset,
        macro_regime_threshold=macro_regime_threshold,
    )
    tmp_path.unlink(missing_ok=True)

    # 写入 memory/eval/ 评估文件
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary["pred_path"] = str(pred_path) if pred_path else str(dataset_path)
    summary["max_positions"] = max_positions
    summary["max_sector_pct"] = max_sector_pct
    summary["commission"] = commission
    summary["stamp_tax"] = stamp_tax
    summary["slippage"] = slippage
    summary["pred_threshold"] = pred_threshold
    summary["stop_loss_pct"] = stop_loss_pct
    summary["use_wf_threshold"] = use_wf_threshold
    summary["wf_train_periods"] = wf_train_periods
    summary["wf_metric"] = wf_metric
    summary["weight_scheme"] = weight_scheme
    summary["max_sector_weight"] = max_sector_weight
    summary["neutralize_market_cap"] = neutralize_market_cap
    summary["macro_dataset"] = macro_dataset

    summary_path = OUTPUT_DIR / f"gbdt_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        # records 里可能包含 numpy 类型，先转 python 原生类型
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[walkforward_gbdt] Summary saved to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", default=None, help="预生成的 predictions parquet")
    parser.add_argument("--dataset", default=None, help="dataset parquet（未提供 pred-path 时自动生成预测）")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--model-type", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--train-window-months", type=int, default=12)
    parser.add_argument("--target", choices=["excess_return", "stock_return"], default="excess_return")
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--max-sector-pct", type=float, default=0.4)
    parser.add_argument("--commission", type=float, default=0.0002)
    parser.add_argument("--stamp-tax", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--pred-threshold", type=float, default=None)
    parser.add_argument("--stop-loss-pct", type=float, default=None,
                        help="个股止损比例，例如 0.05 表示跌 5% 止损")
    parser.add_argument("--use-wf-threshold", action="store_true",
                        help="启用 walk-forward 阈值校准后处理")
    parser.add_argument("--wf-train-periods", type=int, default=12,
                        help="阈值校准训练窗口期数")
    parser.add_argument("--wf-metric", choices=["avg_excess", "win_rate", "sharpe"], default="win_rate",
                        help="阈值校准优化目标")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--weight-scheme", type=str, default="equal",
                        choices=["equal", "score", "risk_parity", "score_risk"],
                        help="持仓权重方案：equal 等权，score 按预测得分，risk_parity 按波动率倒数，score_risk 结合得分和风险")
    parser.add_argument("--max-sector-weight", type=float, default=1.0,
                        help="行业市值权重上限，例如 0.4 表示单行业不超过 40%；1.0 表示不启用")
    parser.add_argument("--neutralize-market-cap", action="store_true",
                        help="选股前对预测得分做市值中性化（截面回归去除 log(总市值) 暴露）")
    parser.add_argument("--macro-dataset", type=str, default=None,
                        help="包含宏观特征（margin/northbound）的数据集路径，用于宏观择时仓位缩放")
    parser.add_argument("--macro-regime-threshold", type=float, default=None,
                        help="宏观 regime 阈值，当 regime_score <= 阈值时空仓；未指定时使用连续缩放")
    args = parser.parse_args()

    if not args.pred_path and not args.dataset:
        parser.error("必须提供 --pred-path 或 --dataset")

    summary = run_walkforward_gbdt(
        pred_path=args.pred_path,
        dataset_path=args.dataset,
        horizon=args.horizon,
        model_type=args.model_type,
        train_window_months=args.train_window_months,
        target=args.target,
        max_positions=args.max_positions,
        max_sector_pct=args.max_sector_pct,
        commission=args.commission,
        stamp_tax=args.stamp_tax,
        slippage=args.slippage,
        pred_threshold=args.pred_threshold,
        stop_loss_pct=args.stop_loss_pct,
        start_date=args.start_date,
        end_date=args.end_date,
        use_wf_threshold=args.use_wf_threshold,
        wf_train_periods=args.wf_train_periods,
        wf_metric=args.wf_metric,
        weight_scheme=args.weight_scheme,
        max_sector_weight=args.max_sector_weight,
        neutralize_market_cap=args.neutralize_market_cap,
        macro_dataset=args.macro_dataset,
        macro_regime_threshold=args.macro_regime_threshold,
    )

    print("\n=== Walk-forward GBDT Backtest Summary ===")
    print(f"Periods: {summary['periods']}")
    print(f"Avg portfolio return: {summary['avg_portfolio_return']:+.2%}")
    print(f"Avg excess return: {summary['avg_excess_return']:+.2%}")
    print(f"Cumulative portfolio return: {summary['cumulative_portfolio_return']:+.2%}")
    print(f"Cumulative excess return: {summary['cumulative_excess_return']:+.2%}")
    print(f"Win rate (excess > 0): {summary['win_rate_excess']:.1%}")
    print(f"Avg turnover: {summary['avg_turnover']:+.1%}")


if __name__ == "__main__":
    main()
