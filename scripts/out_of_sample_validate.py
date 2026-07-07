"""
AlphaHelix 样本外验证脚本

严格划分训练期与测试期，训练期只用来生成 pass2 权重，测试期只能用该权重评估，
避免任何时间穿越和信息泄露。

用法：
    python scripts/out_of_sample_validate.py \
        --train-start 20240101 --train-end 20241231 \
        --test-start 20250101 --test-end 20260615 \
        --strategy regime --horizon 10 --top-n 10 \
        --universe-size 200 --skip-st-check

流程：
1. 在训练期跑 walk-forward，生成 snapshots/evals；
2. 用训练期结果计算 IC，生成 pass2 权重；
3. 在测试期用该固定权重跑 walk-forward；
4. 输出训练期 vs 测试期绩效对比。
"""
import sys
import os
import json
import subprocess
import shutil
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_weights_from_ic import compute_pass2_weights_for_dates

WEIGHTS_DIR = Path("memory/weights")


def run_walkforward(start: str, end: str, strategy: str, horizon: int, top_n: int,
                    universe_size: int, skip_st_check: bool, pass2_weights: Path = None,
                    regime_weights_dir: Path = None, extra_label: str = "") -> Path:
    """调用 walkforward.py，返回汇总文件路径。"""
    cmd = [
        "python", "scripts/walkforward.py",
        "--start", start,
        "--end", end,
        "--strategy", strategy,
        "--horizon", str(horizon),
        "--top-n", str(top_n),
        "--universe-size", str(universe_size),
        "--no-resume",
    ]
    if skip_st_check:
        cmd.append("--skip-st-check")
    if pass2_weights:
        cmd.extend(["--pass2-weights", str(pass2_weights)])
    if regime_weights_dir:
        cmd.extend(["--regime-weights-dir", str(regime_weights_dir)])

    label = f"{extra_label}_{start}_{end}_{strategy}_h{horizon}" if extra_label else f"{start}_{end}_{strategy}_h{horizon}"
    log_path = Path("memory/log") / f"oos_{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["ALPHAHELIX_MAX_WORKERS"] = env.get("ALPHAHELIX_MAX_WORKERS", "4")
    env["ALPHAHELIX_RATE_LIMIT"] = env.get("ALPHAHELIX_RATE_LIMIT", "0.02")

    label = "with train weights" if (pass2_weights or regime_weights_dir) else "baseline"
    print(f"[oos] Running walkforward {start}~{end} ({label}) ...")
    with open(log_path, "w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, env=env, stdout=f, stderr=subprocess.STDOUT)

    # 汇总文件路径
    summary_path = Path("memory/eval") / f"walkforward_{start}_{end}_{strategy}_h{horizon}_monthly.json"
    if not summary_path.exists():
        raise RuntimeError(f"Summary not found: {summary_path}; see {log_path}")
    return summary_path


def get_trade_dates(start: str, end: str) -> list:
    """获取训练期或测试期的月度选股日列表（从 walkforward 汇总中读取）。"""
    summary_path = Path("memory/eval") / f"walkforward_{start}_{end}_regime_h10_monthly.json"
    if not summary_path.exists():
        # 尝试非 regime
        candidates = list(Path("memory/eval").glob(f"walkforward_{start}_{end}_*_h*_monthly.json"))
        if candidates:
            summary_path = candidates[0]
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return [p["date"] for p in data.get("monthly", [])]


def main():
    parser = argparse.ArgumentParser(description="Out-of-sample validation for AlphaHelix")
    parser.add_argument("--train-start", required=True, help="Training period start YYYYMMDD")
    parser.add_argument("--train-end", required=True, help="Training period end YYYYMMDD")
    parser.add_argument("--test-start", required=True, help="Test period start YYYYMMDD")
    parser.add_argument("--test-end", required=True, help="Test period end YYYYMMDD")
    parser.add_argument("--strategy", default="regime", help="Screening strategy")
    parser.add_argument("--horizon", type=int, default=10, help="Holding horizon")
    parser.add_argument("--top-n", type=int, default=10, help="Number of picks")
    parser.add_argument("--universe-size", type=int, default=200, help="Universe size")
    parser.add_argument("--skip-st-check", action="store_true", help="Skip historical ST check")
    parser.add_argument("--min-ic", type=float, default=0.0, help="Min IC threshold for weight calibration")
    parser.add_argument("--per-regime", action="store_true", help="Train separate weights per market regime")
    args = parser.parse_args()

    # 回测模式：禁止读取未来权重（C01/C38 纪律）
    os.environ["AH_BACKTEST_MODE"] = "1"

    if args.test_start <= args.train_end:
        raise ValueError("Test period must start after training period ends")

    # 1. 训练期 walk-forward（用默认权重，只为生成 snapshots/evals）
    train_summary = run_walkforward(
        args.train_start, args.train_end, args.strategy, args.horizon, args.top_n,
        args.universe_size, args.skip_st_check, pass2_weights=None, extra_label="train"
    )

    # 2. 用训练期数据生成 pass2 权重
    train_dates = get_trade_dates(args.train_start, args.train_end)
    print(f"[oos] Calibrating pass2 weights from {len(train_dates)} train periods ...")

    weights_dir = WEIGHTS_DIR / f"oos_train_{args.train_start}_{args.train_end}"
    weights_dir.mkdir(parents=True, exist_ok=True)

    if args.per_regime:
        regime_weights = compute_pass2_weights_for_dates(train_dates, min_ic=args.min_ic, horizon=args.horizon, per_regime=True)
        if not regime_weights:
            raise RuntimeError("No per-regime weights generated from training period")
        for regime, w in regime_weights.items():
            wfile = weights_dir / f"regime_{regime}.json"
            wfile.write_text(json.dumps({
                "walk_forward": True,
                "train_start": args.train_start,
                "train_end": args.train_end,
                "regime": regime,
                "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "weights": {"pass2": w},
                "pass2": w,
            }, ensure_ascii=False, indent=2))
            print(f"[oos] Regime {regime} weights saved to {wfile}")
            print(f"  weights: {w}")
        test_summary_with_weights = run_walkforward(
            args.test_start, args.test_end, args.strategy, args.horizon, args.top_n,
            args.universe_size, args.skip_st_check, pass2_weights=None,
            regime_weights_dir=weights_dir, extra_label="test_weighted"
        )
    else:
        pass2_weights = compute_pass2_weights_for_dates(train_dates, min_ic=args.min_ic, horizon=args.horizon)
        if not pass2_weights:
            raise RuntimeError("No pass2 weights generated from training period")
        weights_file = weights_dir / "pass2.json"
        weights_file.write_text(json.dumps({
            "walk_forward": True,
            "train_start": args.train_start,
            "train_end": args.train_end,
            "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "weights": {"pass2": pass2_weights},
            "pass2": pass2_weights,
        }, ensure_ascii=False, indent=2))
        print(f"[oos] Train weights saved to {weights_file}")
        print("Train pass2 weights:")
        for f, w in sorted(pass2_weights.items(), key=lambda kv: -kv[1]):
            print(f"  {f}: {w:.4f}")
        test_summary_with_weights = run_walkforward(
            args.test_start, args.test_end, args.strategy, args.horizon, args.top_n,
            args.universe_size, args.skip_st_check, pass2_weights=weights_file, extra_label="test_weighted"
        )
    # 复制一份加权结果，避免 baseline 运行后被覆盖
    test_summary_with_weights_copy = Path("memory/eval") / f"walkforward_{args.test_start}_{args.test_end}_{args.strategy}_h{args.horizon}_monthly_weighted.json"
    shutil.copy(test_summary_with_weights, test_summary_with_weights_copy)

    # 4. 测试期 baseline（默认权重）用于对比
    test_summary_baseline = run_walkforward(
        args.test_start, args.test_end, args.strategy, args.horizon, args.top_n,
        args.universe_size, args.skip_st_check, pass2_weights=None, extra_label="test_baseline"
    )

    # 5. 输出对比
    train = json.loads(train_summary.read_text(encoding="utf-8"))
    test_w = json.loads(test_summary_with_weights_copy.read_text(encoding="utf-8"))
    test_b = json.loads(test_summary_baseline.read_text(encoding="utf-8"))

    print("\n=== Out-of-sample Validation Summary ===")
    print(f"Train period: {args.train_start} ~ {args.train_end}")
    print(f"Test period:  {args.test_start} ~ {args.test_end}")
    print()
    print(f"{'Metric':<30} {'Train':>12} {'Test (weighted)':>18} {'Test (baseline)':>18}")
    print("-" * 80)
    for key in ["avg_excess_return", "avg_direction_accuracy", "avg_top3_hit_rate", "cumulative_excess_return", "win_rate_excess"]:
        print(f"{key:<30} {train.get(key, 0):>12.4f} {test_w.get(key, 0):>18.4f} {test_b.get(key, 0):>18.4f}")

    # 保存最终报告
    report_path = Path("memory/eval") / f"oos_report_{args.train_start}_{args.train_end}_{args.test_start}_{args.test_end}.json"
    report = {
        "train_period": [args.train_start, args.train_end],
        "test_period": [args.test_start, args.test_end],
        "train_summary": train,
        "test_weighted_summary": test_w,
        "test_baseline_summary": test_b,
        "weights_dir": str(weights_dir),
        "per_regime": args.per_regime,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
