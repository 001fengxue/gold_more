from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gold_advisor.evaluation import evaluate_forward_returns, run_parameter_grid
from gold_advisor.data import load_prices
from gold_advisor.profiles import PROFILE_ORDER, apply_profile
from gold_advisor.strategy import StrategyConfig


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run signal evaluation and parameter grid.")
    parser.add_argument("--source", choices=["demo", "akshare", "csv"], default="demo")
    parser.add_argument("--symbol", default="Au99.99")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--initial-cash", type=float, default=100_000)
    parser.add_argument("--spread-bps", type=float, default=35)
    parser.add_argument("--profile", choices=PROFILE_ORDER, default="收益优先")
    args = parser.parse_args()

    prices, info = load_prices(args.source, symbol=args.symbol, csv_file=args.csv)
    config = apply_profile(StrategyConfig(spread_bps=args.spread_bps), args.profile)
    _, signal_summary = evaluate_forward_returns(prices, config)
    grid = run_parameter_grid(prices, config, initial_cash=args.initial_cash)

    print(f"数据源: {info.name} - {info.description}")
    print(f"策略风格: {args.profile}")
    print(f"区间: {prices['date'].iloc[0].date()} 至 {prices['date'].iloc[-1].date()}")
    print()
    print("买入动作事后表现:")
    for _, row in signal_summary.iterrows():
        print(
            f"{row['action']} ({row['avg_buy_scale']:.2f}x) / {int(row['horizon_days'])}日: "
            f"样本 {int(row['samples'])}, "
            f"平均 {_format_percent(row['avg_return'])}, "
            f"胜率 {_format_percent(row['win_rate'])}, "
            f"最差 {_format_percent(row['worst_return'])}"
        )

    print()
    print("参数组合前 10:")
    display = grid.head(10).copy()
    for column in ["deep_pullback_pct", "total_return", "annualized_return", "max_drawdown", "latest_target_position"]:
        display[column] = display[column].map(_format_percent)
    display["sharpe"] = display["sharpe"].map(lambda value: f"{value:.2f}")
    display["score"] = display["score"].map(lambda value: f"{value:.4f}")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
